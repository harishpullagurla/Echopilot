import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DATA_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), 'data');
const EMBEDDING_MODEL = process.env.LOCAL_EMBEDDING_MODEL || 'Xenova/bge-small-en-v1.5';
const RERANKER_MODEL = process.env.LOCAL_RERANKER_MODEL || 'Xenova/ms-marco-MiniLM-L-6-v2';
const VERIFIER_MODEL = process.env.LOCAL_NLI_MODEL || 'Xenova/mobilebert-uncased-mnli';
const EMBEDDING_DIMENSIONS = 384;
const MAX_CHUNK_CHARS = 1200;
const CHUNK_OVERLAP_CHARS = 140;
const EMBEDDING_BATCH_SIZE = 16;
const BKT = { initialKnowledge: 0.3, learning: 0.15, guess: 0.25, slip: 0.1 };
const RRF_K = 60;
const STOPWORDS = new Set([
  'the',
  'and',
  'for',
  'from',
  'with',
  'that',
  'this',
  'into',
  'what',
  'which',
  'about',
  'topic',
  'document',
  'material',
  'concept',
  'explain',
  'find',
  'using',
  'their',
  'there',
  'then',
  'than',
  'your',
  'have',
  'has',
  'are',
  'was',
  'were',
]);
let transformersPromise;
let embeddingPipelinePromise;
let rerankerPromise;
let verifierPromise;

// Load the ML runtime only when AI work is needed; it caches model files locally.
async function transformers() {
  if (!transformersPromise)
    transformersPromise = import('@huggingface/transformers').then((module) => {
      module.env.cacheDir = path.join(DATA_DIR, 'model-cache');
      return module;
    });
  return transformersPromise;
}

async function getEmbeddingPipeline() {
  if (!embeddingPipelinePromise)
    embeddingPipelinePromise = transformers().then(({ pipeline }) =>
      pipeline('feature-extraction', EMBEDDING_MODEL)
    );
  return embeddingPipelinePromise;
}

async function getPairModels(modelId, current) {
  if (current) return current;
  const { AutoTokenizer, AutoModelForSequenceClassification } = await transformers();
  return {
    tokenizer: await AutoTokenizer.from_pretrained(modelId),
    model: await AutoModelForSequenceClassification.from_pretrained(modelId),
  };
}

async function getReranker() {
  if (!rerankerPromise) rerankerPromise = getPairModels(RERANKER_MODEL, null);
  return rerankerPromise;
}

async function getVerifier() {
  if (!verifierPromise) verifierPromise = getPairModels(VERIFIER_MODEL, null);
  return verifierPromise;
}

function normalizeVector(values) {
  const magnitude = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
  if (!magnitude) throw new Error('The local embedding model returned an empty vector.');
  return values.map((value) => value / magnitude);
}

function splitLongBlock(block) {
  const pieces = [];
  let start = 0;
  while (start < block.length) {
    let end = Math.min(start + MAX_CHUNK_CHARS, block.length);
    if (end < block.length) {
      const boundary = block.lastIndexOf(' ', end);
      if (boundary > start + MAX_CHUNK_CHARS * 0.65) end = boundary;
    }
    const piece = block.slice(start, end).trim();
    if (piece) pieces.push(piece);
    if (end >= block.length) break;
    start = Math.max(start + 1, end - CHUNK_OVERLAP_CHARS);
  }
  return pieces;
}

export function chunkDocument(text) {
  const blocks = String(text || '')
    .replace(/\r/g, '')
    .replace(/\f/g, '\n\n')
    .split(/\n{2,}/)
    .map((block) => block.replace(/[\t ]+/g, ' ').trim())
    .filter(Boolean)
    // Split oversized paragraphs with overlap so a boundary does not remove context.
    .flatMap((block) => (block.length > MAX_CHUNK_CHARS ? splitLongBlock(block) : [block]));

  const chunks = [];
  let current = '';
  for (const block of blocks) {
    if (current && current.length + block.length + 1 > MAX_CHUNK_CHARS) {
      chunks.push(current);
      current = '';
    }
    current = current ? `${current}\n${block}` : block;
  }
  if (current) chunks.push(current);
  return chunks.map((chunk, index) => ({
    id: `C${String(index + 1).padStart(3, '0')}`,
    text: chunk,
  }));
}

async function embedTexts(texts, { query = false } = {}) {
  const extractor = await getEmbeddingPipeline();
  const prepared = texts.map((text) =>
    query ? `Represent this sentence for searching relevant passages: ${text}` : text
  );
  const output = await extractor(prepared, { pooling: 'mean', normalize: true });
  const vectors = output.tolist();
  if (
    !Array.isArray(vectors) ||
    vectors.length !== texts.length ||
    vectors.some((vector) => !Array.isArray(vector) || vector.length !== EMBEDDING_DIMENSIONS)
  ) {
    throw new Error('The local embedding model returned an incomplete vector batch.');
  }
  return vectors.map(normalizeVector);
}

function indexFile(indexDir, materialId) {
  return path.join(indexDir, `${materialId}.json`);
}

export async function ensureMaterialIndex(text, materialId, indexDir) {
  const chunks = chunkDocument(text);
  const contentHash = crypto
    .createHash('sha256')
    .update(String(text || ''))
    .digest('hex');
  await fs.mkdir(indexDir, { recursive: true });
  const file = indexFile(indexDir, materialId);
  try {
    const cached = JSON.parse(await fs.readFile(file, 'utf8'));
    // Reuse vectors only when both the PDF text and embedding model still match.
    if (
      cached.contentHash === contentHash &&
      cached.model === EMBEDDING_MODEL &&
      cached.dimensions === EMBEDDING_DIMENSIONS &&
      cached.chunks?.length === chunks.length &&
      cached.chunks.every((item) => item.embedding?.length === EMBEDDING_DIMENSIONS)
    )
      return cached;
  } catch {}

  const embedded = [];
  for (let offset = 0; offset < chunks.length; offset += EMBEDDING_BATCH_SIZE) {
    const batch = chunks.slice(offset, offset + EMBEDDING_BATCH_SIZE);
    const vectors = await embedTexts(batch.map((chunk) => chunk.text));
    embedded.push(...batch.map((chunk, index) => ({ ...chunk, embedding: vectors[index] })));
  }
  const result = {
    materialId,
    contentHash,
    model: EMBEDDING_MODEL,
    dimensions: EMBEDDING_DIMENSIONS,
    chunks: embedded,
  };
  const tempFile = `${file}.tmp`;
  await fs.writeFile(tempFile, JSON.stringify(result));
  await fs.rename(tempFile, file);
  return result;
}

function tokenize(text) {
  return (
    String(text || '')
      .toLowerCase()
      .normalize('NFKD')
      .replace(/[\u0300-\u036f]/g, '')
      .match(/[a-z0-9]+/g) || []
  );
}

export function canonicalTopicId(label) {
  const normalized = tokenize(label)
    .filter((word) => !STOPWORDS.has(word))
    .join('-')
    .slice(0, 100);
  return (
    normalized ||
    `topic-${crypto
      .createHash('sha1')
      .update(String(label || ''))
      .digest('hex')
      .slice(0, 10)}`
  );
}

export function bm25Rank(chunks, query) {
  const terms = [
    ...new Set(tokenize(query).filter((word) => word.length > 2 && !STOPWORDS.has(word))),
  ];
  if (!terms.length || !chunks.length) return [];
  const documents = chunks.map((chunk) => tokenize(chunk.text));
  const averageLength =
    documents.reduce((sum, words) => sum + words.length, 0) / documents.length || 1;
  const documentFrequencies = new Map(
    terms.map((term) => [
      term,
      documents.reduce((sum, words) => sum + (words.includes(term) ? 1 : 0), 0),
    ])
  );
  // BM25 balances term frequency, term rarity, and document length.
  const k1 = 1.5;
  const b = 0.75;
  return chunks
    .map((chunk, index) => {
      const words = documents[index];
      const counts = new Map();
      for (const word of words) counts.set(word, (counts.get(word) || 0) + 1);
      const score = terms.reduce((sum, term) => {
        const frequency = counts.get(term) || 0;
        if (!frequency) return sum;
        const docFrequency = documentFrequencies.get(term) || 0;
        const inverseFrequency = Math.log(
          1 + (chunks.length - docFrequency + 0.5) / (docFrequency + 0.5)
        );
        const normalizedFrequency =
          (frequency * (k1 + 1)) / (frequency + k1 * (1 - b + (b * words.length) / averageLength));
        return sum + inverseFrequency * normalizedFrequency;
      }, 0);
      return { ...chunk, lexicalScore: score, index };
    })
    .sort((a, b) => b.lexicalScore - a.lexicalScore || a.index - b.index);
}

export async function hybridRank(chunks, query, limit = 7) {
  if (!chunks.length) return [];
  const lexical = bm25Rank(chunks, query);
  try {
    const missing = chunks.some(
      (chunk) => !Array.isArray(chunk.embedding) || chunk.embedding.length !== EMBEDDING_DIMENSIONS
    );
    const vectors = missing
      ? await embedTexts(chunks.map((chunk) => chunk.text))
      : chunks.map((chunk) => chunk.embedding);
    const [queryVector] = await embedTexts([query], { query: true });
    const semantic = chunks
      .map((chunk, index) => ({
        ...chunk,
        embedding: vectors[index],
        semanticScore: cosine(vectors[index], queryVector),
      }))
      .sort((a, b) => b.semanticScore - a.semanticScore);
    // Fuse ranks first, then spend more compute reranking only promising candidates.
    const fused = reciprocalRankFusion([semantic, lexical]).slice(0, Math.max(24, limit));
    return (await rerankPassages(query, fused)).slice(0, limit);
  } catch (error) {
    console.warn('Local hybrid retrieval unavailable; using BM25.', error.message);
    return lexical.slice(0, limit);
  }
}

function cosine(a, b) {
  let product = 0;
  for (let i = 0; i < a.length; i++) product += a[i] * b[i];
  return product;
}

async function scorePairs(models, leftTexts, rightTexts) {
  if (!leftTexts.length) return [];
  const encoded = await models.tokenizer(leftTexts, {
    text_pair: rightTexts,
    padding: true,
    truncation: true,
    max_length: 512,
  });
  const output = await models.model(encoded);
  return output.logits.tolist();
}

async function rerankPassages(query, passages) {
  if (passages.length < 2) return passages;
  try {
    if (!rerankerPromise) rerankerPromise = getPairModels(RERANKER_MODEL, null);
    const model = await rerankerPromise;
    const logits = await scorePairs(
      model,
      passages.map(() => query),
      passages.map((passage) => passage.text)
    );
    return passages
      .map((passage, index) => ({ ...passage, rerankScore: Number(logits[index]?.[0] ?? 0) }))
      .sort((a, b) => b.rerankScore - a.rerankScore);
  } catch (error) {
    rerankerPromise = null;
    console.warn(
      `Local passage reranking unavailable; keeping hybrid retrieval order.`,
      error.message
    );
    return passages;
  }
}

function reciprocalRankFusion(rankings) {
  const fused = new Map();
  for (const ranking of rankings) {
    ranking.forEach((item, index) => {
      const current = fused.get(item.id) || { ...item, score: 0 };
      // RRF uses one-based rank and avoids comparing incompatible raw score scales.
      current.score += 1 / (RRF_K + index + 1);
      if (item.embedding) current.embedding = item.embedding;
      fused.set(item.id, current);
    });
  }
  return [...fused.values()].sort((a, b) => b.score - a.score);
}

export async function retrievePassages(text, materialId, indexDir, topic, limit = 7) {
  let index;
  try {
    index = await ensureMaterialIndex(text, materialId, indexDir);
  } catch (error) {
    console.warn(
      `Could not build local embeddings for ${materialId}; using BM25 retrieval.`,
      error.message
    );
    index = { chunks: chunkDocument(text) };
  }
  const query = topic
    ? `Find authoritative definitions, explanations, worked examples, and common confusions about the study topic: ${topic}.`
    : 'Find foundational definitions, central concepts, important rules, and relationships that best introduce this study document.';
  const reranked = await hybridRank(index.chunks, query, limit);
  const hybrid = reranked.some(
    (item) => Number.isFinite(item.rerankScore) || Number.isFinite(item.semanticScore)
  );
  return {
    mode: `${hybrid ? 'hybrid-bm25-dense-rrf' : 'bm25-fallback'}${reranked.some((item) => Number.isFinite(item.rerankScore)) ? '+local-cross-encoder' : ''}`,
    passages: reranked
      .slice(0, limit)
      .map(({ id, text: passageText, score, lexicalScore, semanticScore, rerankScore }) => ({
        id,
        text: passageText,
        score: Number((rerankScore ?? score ?? lexicalScore ?? semanticScore ?? 0).toFixed(4)),
      })),
  };
}

export function updateBkt(previous, correct, updatedAt = new Date().toISOString()) {
  const attempts = Number(previous?.attempts) || 0;
  const legacyScore = Number(previous?.score);
  const known = Number.isFinite(Number(previous?.pKnown))
    ? Number(previous.pKnown)
    : attempts && Number.isFinite(legacyScore)
      ? legacyScore / 100
      : BKT.initialKnowledge;
  // First update the belief using guess/slip likelihoods, then allow learning.
  const posterior = correct
    ? (known * (1 - BKT.slip)) / (known * (1 - BKT.slip) + (1 - known) * BKT.guess)
    : (known * BKT.slip) / (known * BKT.slip + (1 - known) * (1 - BKT.guess));
  const pKnown = Math.min(0.999, Math.max(0.001, posterior + (1 - posterior) * BKT.learning));
  const consecutiveCorrect = correct ? (Number(previous?.consecutiveCorrect) || 0) + 1 : 0;
  const intervalDays = correct
    ? Math.min(30, Math.max(1, Number(previous?.intervalDays) || 1) * (1.35 + 0.45 * pKnown))
    : 0;
  return {
    score: Math.round(pKnown * 100),
    pKnown: Number(pKnown.toFixed(4)),
    attempts: attempts + 1,
    updatedAt,
    consecutiveCorrect,
    intervalDays,
  };
}

function entropy(probability) {
  const p = Math.min(0.999, Math.max(0.001, probability));
  return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
}

function expectedInformationGain(pKnown) {
  const p = Math.min(0.999, Math.max(0.001, pKnown));
  const likelihoodCorrect = p * (1 - BKT.slip) + (1 - p) * BKT.guess;
  const afterCorrect = (p * (1 - BKT.slip)) / likelihoodCorrect;
  const likelihoodWrong = 1 - likelihoodCorrect;
  const afterWrong = (p * BKT.slip) / likelihoodWrong;
  // Expected uncertainty after either response is subtracted from current entropy.
  return Math.max(
    0,
    entropy(p) - likelihoodCorrect * entropy(afterCorrect) - likelihoodWrong * entropy(afterWrong)
  );
}

export function selectAdaptiveTopic(mastery = {}, attempts = [], now = Date.now()) {
  const entries = Object.entries(mastery);
  if (!entries.length) return null;
  const ranked = entries
    .map(([label, state]) => {
      const pKnown = Number.isFinite(Number(state?.pKnown))
        ? Number(state.pKnown)
        : (Number(state?.score) || 0) / 100;
      const lastUpdated = Date.parse(state?.updatedAt || '');
      const elapsedDays = Number.isFinite(lastUpdated)
        ? Math.max(0, (now - lastUpdated) / 86400000)
        : 30;
      const storedInterval = Number(state?.intervalDays);
      const due =
        storedInterval === 0 ? 2 : Math.min(2, elapsedDays / Math.max(1, storedInterval || 1));
      const recent = attempts
        .flatMap((attempt) => attempt.results || [])
        .filter((result) => result.topic === label)
        .slice(0, 10);
      const recentError = recent.length
        ? recent.filter((result) => !result.correct).length / recent.length
        : 0;
      // Uncertainty, review due-ness, recent errors, and lower mastery all matter.
      const score =
        expectedInformationGain(pKnown) * (1 + due) + recentError * 0.35 + (1 - pKnown) * 0.1;
      return { label, score };
    })
    .sort((a, b) => b.score - a.score || a.label.localeCompare(b.label));
  return ranked[0]?.label || null;
}

function normalizeEvidence(text) {
  return String(text || '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()
    .replace(/\s+/g, ' ');
}

export function validateQuiz(quiz, targetTopic = '') {
  if (!quiz || typeof quiz.title !== 'string' || !quiz.title.trim())
    return 'Return a short non-empty quiz title.';
  if (!Array.isArray(quiz.questions) || quiz.questions.length !== 5)
    return 'Return exactly five questions.';
  const seenQuestions = new Set();
  for (const [index, question] of quiz.questions.entries()) {
    if (
      typeof question.topic !== 'string' ||
      !question.topic.trim() ||
      typeof question.question !== 'string' ||
      question.question.trim().length < 12 ||
      typeof question.explanation !== 'string' ||
      question.explanation.trim().length < 12 ||
      !Array.isArray(question.options) ||
      question.options.length !== 4 ||
      question.options.some((option) => typeof option !== 'string' || !option.trim())
    )
      return `Question ${index + 1} must have a topic, clear question, explanation, and exactly four non-empty string options.`;
    if (!Number.isInteger(question.answer) || question.answer < 0 || question.answer > 3)
      return `Question ${index + 1} has an invalid answer index.`;
    if (!['gentle', 'steady', 'stretch'].includes(question.difficulty))
      return `Question ${index + 1} has an invalid difficulty label.`;
    const normalizedOptions = question.options.map((option) => normalizeEvidence(option));
    if (normalizedOptions.some((option) => !option) || new Set(normalizedOptions).size !== 4)
      return `Question ${index + 1} has empty or repeated options.`;
    const normalizedQuestion = normalizeEvidence(question.question);
    if (seenQuestions.has(normalizedQuestion))
      return `Question ${index + 1} repeats another question.`;
    seenQuestions.add(normalizedQuestion);
    if (targetTopic && String(question.topic).trim().toLowerCase() !== targetTopic.toLowerCase())
      return `Every question topic must be exactly "${targetTopic}".`;
  }
  return '';
}

function overlapScore(queryTerms, text) {
  const words = new Set(tokenize(text));
  const matches = queryTerms.filter((term) => words.has(term));
  return queryTerms.length ? matches.length / queryTerms.length : 0;
}

const evidenceStopwords = new Set([
  'the',
  'and',
  'for',
  'from',
  'with',
  'that',
  'this',
  'what',
  'which',
  'would',
  'could',
  'should',
  'does',
  'have',
  'has',
  'are',
  'was',
  'were',
  'into',
  'their',
  'there',
  'then',
  'than',
  'based',
  'following',
  'according',
  'provided',
]);
function evidenceTerms(question) {
  const correctOption = question.options[question.answer];
  return [
    ...new Set(
      tokenize(
        `${question.topic} ${question.question} ${correctOption} ${question.explanation}`
      ).filter((term) => term.length > 2 && !evidenceStopwords.has(term))
    ),
  ];
}

function bestEvidenceExcerpt(question, passage) {
  const queryTerms = evidenceTerms(question);
  const sentences = passage.text
    .split(/(?<=[.!?])\s+|\n+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
  const ranked = sentences
    .map((sentence, index) => ({ sentence, index, score: overlapScore(queryTerms, sentence) }))
    .sort((a, b) => b.score - a.score || a.index - b.index);
  let excerpt = ranked[0]?.sentence || passage.text.trim();
  if (!ranked[0]?.score) return '';
  if (excerpt.length > 360) {
    const end = excerpt.lastIndexOf(' ', 357);
    excerpt = excerpt.slice(0, end > 180 ? end : 357).trim();
  }
  return excerpt;
}

function softmax(values) {
  const maximum = Math.max(...values);
  const exponents = values.map((value) => Math.exp(value - maximum));
  const total = exponents.reduce((sum, value) => sum + value, 0) || 1;
  return exponents.map((value) => value / total);
}

async function verifyAnswerEvidence(question, passage) {
  try {
    if (!verifierPromise) verifierPromise = getPairModels(VERIFIER_MODEL, null);
    const models = await verifierPromise;
    const hypothesis = `The correct answer to this question is: ${question.options[question.answer]}. Explanation: ${question.explanation}`;
    const logits = await scorePairs(models, [passage.text], [hypothesis]);
    const probabilities = softmax(logits[0] || []);
    const labels = models.model.config?.id2label || {
      0: 'ENTAILMENT',
      1: 'NEUTRAL',
      2: 'CONTRADICTION',
    };
    const scores = {};
    for (const [index, probability] of probabilities.entries())
      scores[String(labels[index] || '').toUpperCase()] = probability;
    const entailment = scores.ENTAILMENT ?? 0;
    const contradiction = scores.CONTRADICTION ?? 0;
    const neutral = scores.NEUTRAL ?? Math.max(0, 1 - entailment - contradiction);
    // This is a review hint, not a guarantee that the answer is factually correct.
    const supported = entailment >= 0.55 && entailment > contradiction;
    return {
      status: supported ? 'supported' : 'review',
      entailment: Number(entailment.toFixed(3)),
      contradiction: Number(contradiction.toFixed(3)),
      neutral: Number(neutral.toFixed(3)),
    };
  } catch (error) {
    verifierPromise = null;
    console.warn(
      'Local evidence verification unavailable; marking source check unchecked.',
      error.message
    );
    return { status: 'unchecked', entailment: null, contradiction: null, neutral: null };
  }
}

export async function attachEvidence(quiz, passages) {
  for (const [index, question] of quiz.questions.entries()) {
    const queryTerms = evidenceTerms(question);
    let ranked = passages
      .map((passage, order) => ({ passage, order, score: overlapScore(queryTerms, passage.text) }))
      .sort((a, b) => b.score - a.score || a.order - b.order);
    try {
      if (!rerankerPromise) rerankerPromise = getPairModels(RERANKER_MODEL, null);
      const model = await rerankerPromise;
      const logits = await scorePairs(
        model,
        passages.map(
          () =>
            `${question.topic}: ${question.question} Correct answer: ${question.options[question.answer]}`
        ),
        passages.map((passage) => passage.text)
      );
      ranked = passages
        .map((passage, order) => ({ passage, order, score: Number(logits[order]?.[0] ?? 0) }))
        .sort((a, b) => b.score - a.score || a.order - b.order);
    } catch (error) {
      rerankerPromise = null;
      console.warn(
        'Could not rerank evidence for a quiz question; using lexical matching.',
        error.message
      );
    }
    const passage = ranked[0]?.passage;
    if (!passage)
      return `Question ${index + 1} could not be matched to evidence in the retrieved PDF passages.`;
    const evidence = bestEvidenceExcerpt(question, passage) || passage.text.slice(0, 360).trim();
    if (!evidence)
      return `Question ${index + 1} could not be matched to an evidence excerpt in the retrieved PDF passages.`;
    question.sourceId = passage.id;
    question.evidence = evidence;
    question.evidenceCheck = await verifyAnswerEvidence(question, passage);
  }
  return '';
}

export function formatPassages(passages) {
  return passages.map((passage) => `[SOURCE ${passage.id}]\n${passage.text}`).join('\n\n');
}

export function bktParameters() {
  return { ...BKT };
}
