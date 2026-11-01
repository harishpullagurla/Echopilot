import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import pdf from 'pdf-parse';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import crypto from 'node:crypto';
import {
  ensureMaterialIndex,
  formatPassages,
  retrievePassages,
  validateQuiz,
  attachEvidence,
  updateBkt,
  selectAdaptiveTopic,
  canonicalTopicId,
} from './ai.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const DATA = path.join(__dirname, 'data');
const FILES = path.join(DATA, 'uploads');
const EMBEDDINGS = path.join(DATA, 'embeddings');
const DB = path.join(DATA, 'db.json');
const app = express();
const port = Number(process.env.PORT || 3001);
app.use(express.json({ limit: '2mb' }));
await fs.mkdir(FILES, { recursive: true });
const quizResponseSchema = {
  type: 'OBJECT',
  properties: {
    title: { type: 'STRING' },
    questions: {
      type: 'ARRAY',
      minItems: 5,
      maxItems: 5,
      items: {
        type: 'OBJECT',
        properties: {
          topic: { type: 'STRING' },
          question: { type: 'STRING' },
          options: { type: 'ARRAY', minItems: 4, maxItems: 4, items: { type: 'STRING' } },
          answer: { type: 'INTEGER', minimum: 0, maximum: 3 },
          explanation: { type: 'STRING' },
          difficulty: { type: 'STRING', enum: ['gentle', 'steady', 'stretch'] },
        },
        required: ['topic', 'question', 'options', 'answer', 'explanation', 'difficulty'],
      },
    },
  },
  required: ['title', 'questions'],
};
const initial = { materials: [], chats: [], attempts: [], mastery: {} };
function normalizeDb(db) {
  // Keep older local JSON files readable as the app's stored shape evolves.
  db.materials ||= [];
  db.chats ||= [];
  db.attempts ||= [];
  db.mastery ||= {};
  db.masteryByMaterial ||= {};
  if (db.materials.length === 1) {
    const only = db.materials[0];
    if (Object.keys(db.mastery).length && !db.masteryByMaterial[only.id])
      db.masteryByMaterial[only.id] = structuredClone(db.mastery);
    for (const attempt of db.attempts)
      if (!attempt.materialId) {
        attempt.materialId = only.id;
        attempt.materialName = only.name;
      }
  }
  for (const attempt of db.attempts)
    if (attempt.materialId && !attempt.materialName)
      attempt.materialName =
        db.materials.find((m) => m.id === attempt.materialId)?.name || 'Removed material';
  return db;
}
async function readDb() {
  try {
    return normalizeDb({ ...initial, ...JSON.parse(await fs.readFile(DB, 'utf8')) });
  } catch {
    return normalizeDb(structuredClone(initial));
  }
}
async function writeDb(data) {
  await fs.writeFile(DB, JSON.stringify(normalizeDb(data), null, 2));
}
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (_req, file, cb) =>
    cb(
      null,
      file.mimetype === 'application/pdf' || file.originalname.toLowerCase().endsWith('.pdf')
    ),
});

async function askGemini(prompt, json = false) {
  // Gemini is called only from the server so the API key never reaches the browser.
  const key = process.env.GEMINI_API_KEY;
  if (!key) return null;
  const model = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(key)}`;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: {
            temperature: 0.35,
            ...(json
              ? { responseMimeType: 'application/json', responseSchema: quizResponseSchema }
              : {}),
          },
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        const error = new Error(
          payload.error?.message || `Gemini request failed (${response.status})`
        );
        if (attempt === 0 && [429, 500, 502, 503, 504].includes(response.status)) {
          await new Promise((resolve) => setTimeout(resolve, 900));
          continue;
        }
        throw error;
      }
      return payload.candidates?.[0]?.content?.parts?.map((p) => p.text || '').join('') || '';
    } catch (error) {
      if (attempt === 0 && (error.name === 'TypeError' || error.cause?.code === 'ECONNRESET')) {
        await new Promise((resolve) => setTimeout(resolve, 900));
        continue;
      }
      throw error;
    }
  }
  return null;
}
function parseModelJson(text) {
  return JSON.parse(text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, ''));
}
const demos = [
  {
    topic: 'Core ideas',
    question: 'In your own words, what is the central idea in this section?',
    options: [
      'A key principle that explains how the topic works',
      'A minor detail with no connection',
      'A historical date only',
      'A definition unrelated to the topic',
    ],
    answer: 0,
    explanation:
      'The central idea connects the section’s details and explains the underlying principle.',
    difficulty: 'gentle',
  },
  {
    topic: 'Applying concepts',
    question: 'Which approach best applies the concept to a new example?',
    options: [
      'Identify the relevant principle, then check the evidence',
      'Memorize the heading without context',
      'Skip the working and guess',
      'Repeat an unrelated definition',
    ],
    answer: 0,
    explanation:
      'Applying a concept means selecting the relevant principle and using it to reason through the example.',
    difficulty: 'steady',
  },
  {
    topic: 'Making connections',
    question: 'Why is it useful to connect this idea with the previous section?',
    options: [
      'It can reveal how the two concepts work together',
      'It makes the material longer',
      'It replaces understanding with memorization',
      'It is never useful',
    ],
    answer: 0,
    explanation:
      'Connections show how concepts build on each other and help you transfer your learning.',
    difficulty: 'stretch',
  },
];

// The UI uses these routes to load state, add materials, and run a quiz lifecycle.
app.get('/api/health', (_req, res) =>
  res.json({ ok: true, geminiConfigured: Boolean(process.env.GEMINI_API_KEY) })
);
app.get('/api/state', async (_req, res) => {
  const db = await readDb();
  await writeDb(db);
  res.json({
    materials: db.materials.map(({ text: _text, ...material }) => material),
    attempts: db.attempts,
    masteryByMaterial: db.masteryByMaterial,
  });
});
app.post('/api/materials', upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'Choose a PDF file to upload.' });
  try {
    const result = await pdf(req.file.buffer);
    const text = result.text.trim();
    if (text.length < 30)
      return res
        .status(422)
        .json({ error: 'No readable text was found. This PDF may be scanned or image-only.' });
    const id = crypto.randomUUID();
    const safeName = path.basename(req.file.originalname).replace(/[^\w. -]/g, '_');
    await fs.writeFile(path.join(FILES, `${id}.pdf`), req.file.buffer);
    const db = await readDb();
    const material = {
      id,
      name: safeName,
      size: req.file.size,
      pages: result.numpages || 0,
      text,
      addedAt: new Date().toISOString(),
    };
    db.materials.unshift(material);
    await writeDb(db);
    // Saving the PDF and its text succeeds even if optional local indexing fails.
    let embeddingsReady = false;
    try {
      await ensureMaterialIndex(text, id, EMBEDDINGS);
      embeddingsReady = true;
    } catch (error) {
      console.warn(
        `Local embedding index will be built when needed for ${safeName}.`,
        error.message
      );
    }
    const { text: _text, ...summary } = material;
    res.status(201).json({ ...summary, embeddingsReady });
  } catch (e) {
    res.status(500).json({ error: `Could not read this PDF: ${e.message}` });
  }
});
app.delete('/api/materials/:id', async (req, res) => {
  const db = await readDb();
  const found = db.materials.find((m) => m.id === req.params.id);
  if (!found) return res.status(404).json({ error: 'Material not found.' });
  db.materials = db.materials.filter((m) => m.id !== req.params.id);
  await writeDb(db);
  await Promise.all([
    fs.rm(path.join(FILES, `${found.id}.pdf`), { force: true }),
    fs.rm(path.join(EMBEDDINGS, `${found.id}.json`), { force: true }),
  ]);
  res.json({ ok: true });
});
app.post('/api/quiz', async (req, res) => {
  const db = await readDb();
  const material = db.materials.find((m) => m.id === req.body.materialId);
  if (!material)
    return res.status(400).json({ error: 'Choose a study material before starting a quiz.' });
  const mastery = db.masteryByMaterial[material.id] || {};
  const materialAttempts = db.attempts.filter((attempt) => attempt.materialId === material.id);
  // Select a practice topic from stored learner state; first-time material has none.
  const selectedTopic = selectAdaptiveTopic(mastery, materialAttempts);
  const targetTopic = req.body.topic || selectedTopic || '';
  try {
    const learnerEstimate = targetTopic ? mastery[targetTopic] : null;
    const learnerScore = Number(learnerEstimate?.score ?? 30);
    const recentTopicResults = materialAttempts
      .flatMap((attempt) => attempt.results || [])
      .filter((result) => !targetTopic || result.topic === targetTopic)
      .slice(0, 10);
    const recentAccuracy = recentTopicResults.length
      ? recentTopicResults.filter((result) => result.correct).length / recentTopicResults.length
      : null;
    const retrieval = await retrievePassages(material.text, material.id, EMBEDDINGS, targetTopic);
    const sourceText = formatPassages(retrieval.passages);
    const knownTopics = Object.keys(mastery);
    const topicRule = targetTopic
      ? `All five questions must assess the selected learning topic for THIS PDF: "${targetTopic}". Use exactly "${targetTopic}" as the topic label on every question. If it represents an existing topic, preserve that exact label.`
      : 'Select a foundational topic or tightly related set of foundational concepts from these passages. Reuse existing topic labels when they clearly describe the same concept; otherwise choose a concise, specific label.';
    // Treat model output as a draft: schema and semantic checks below remain authoritative.
    const instructions = `You are EchoPilot, a careful study assessment writer. Create a quiz for the learner using ONLY the cited source passages below. The selected study material is "${material.name}". ${topicRule}

ADAPTIVE LEARNER STATE: Estimated mastery for this topic is ${learnerScore}/100${recentAccuracy == null ? '' : `; recent accuracy is ${Math.round(recentAccuracy * 100)}% across ${recentTopicResults.length} answers`}. Use this as a continuous signal: start with clear recall when confidence is low, move toward application as confidence and recent accuracy rise, and include a modest stretch question when the learner is doing well. Keep topic focus stable. For a first quiz, start accessible and include varied foundational questions.
EXISTING TOPIC LABELS FOR THIS MATERIAL: ${JSON.stringify(knownTopics)}

ACCURACY RULES:
- Each question must test an important idea explicitly supported by one cited passage. Do not invent facts, add outside knowledge, or infer beyond the evidence.
- Write a clear, self-contained question with exactly one unambiguously correct answer.
- Provide exactly four distinct answer options. Make distractors plausible but definitely incorrect according to the cited passage. Avoid trick wording, double negatives, "all/none of the above", repeated options, and clues that reveal the answer.
- Select the answer index only after checking the question and all four options against the source. The answer is a zero-based integer from 0 to 3.
- Use correct spelling, grammar, and standard mathematical/scientific notation. If the source contains a typo or malformed notation, silently normalize it only when the intended meaning is unambiguous; otherwise choose another fact.
- The explanation must state why the keyed answer is correct and must agree with the source quote.
- Base each question and its explanation on the supplied passages. Do not add source IDs or evidence fields; the server attaches an exact excerpt from the best-matching retrieved passage after generation.
- Give each question a concise topic label and one difficulty value: gentle, steady, or stretch.
- Make the five questions distinct and avoid testing the same fact twice.

RETURN ONLY valid JSON matching this schema exactly:
{"title":"short quiz title","questions":[{"topic":"topic label","question":"question text","options":["A","B","C","D"],"answer":0,"explanation":"supportive explanation grounded in evidence","difficulty":"gentle|steady|stretch"}]}

LEARNER KNOWLEDGE ESTIMATES FOR THIS MATERIAL (0-100): ${JSON.stringify(Object.fromEntries(Object.entries(mastery).map(([name, value]) => [name, value.score])))}
RETRIEVAL MODE: ${retrieval.mode}

SOURCE PASSAGES:
${sourceText}`;

    let quiz;
    let validationError = '';
    let previousDraft = '';
    for (let generation = 0; generation < 3; generation++) {
      const prompt =
        generation === 0
          ? instructions
          : `${instructions}\n\nREPAIR THE DRAFT BELOW. It failed validation: ${validationError}. Return a fully corrected quiz that obeys every rule. Preserve valid questions where possible. Replace any invalid question completely. For every question, verify that its four options are non-empty and meaningfully distinct, and that exactly one option is correct. Check every answer index and explanation against the source before responding.\n\nINVALID DRAFT TO REPAIR:\n${previousDraft}`;
      const output = await askGemini(prompt, true);
      if (!output) throw new Error('No Gemini response');
      previousDraft = output.slice(0, 12000);
      try {
        quiz = parseModelJson(output);
      } catch {
        validationError = 'the response was not valid JSON';
        continue;
      }
      previousDraft = JSON.stringify(quiz);
      validationError = validateQuiz(quiz, targetTopic);
      if (!validationError) break;
    }
    if (validationError)
      throw new Error(`Gemini could not produce a source-verified quiz: ${validationError}`);
    validationError = await attachEvidence(quiz, retrieval.passages);
    if (validationError)
      throw new Error(`Gemini could not produce a source-verified quiz: ${validationError}`);
    quiz.questions = quiz.questions.map((question) => ({
      ...question,
      difficulty: question.difficulty.toLowerCase(),
    }));
    res.json({
      ...quiz,
      id: crypto.randomUUID(),
      materialId: material.id,
      materialName: material.name,
      demo: false,
    });
  } catch (e) {
    if (process.env.GEMINI_API_KEY && e.message !== 'No Gemini response')
      return res.status(502).json({ error: e.message });
    res.json({
      id: crypto.randomUUID(),
      title: targetTopic
        ? `A little more on ${targetTopic}`
        : `A learning check-in for ${material.name}`,
      materialId: material.id,
      materialName: material.name,
      questions: demos,
      demo: true,
    });
  }
});
app.post('/api/quiz/submit', async (req, res) => {
  const { questions, answers, materialId, title } = req.body;
  if (
    !Array.isArray(questions) ||
    !questions.length ||
    !Array.isArray(answers) ||
    questions.length !== answers.length ||
    answers.some((a) => !Number.isInteger(Number(a)))
  )
    return res.status(400).json({ error: 'Quiz answers are incomplete.' });
  const db = await readDb();
  const material = db.materials.find((m) => m.id === materialId);
  if (!material)
    return res.status(400).json({ error: 'This study material is no longer available.' });
  const now = new Date().toISOString();
  // Grade by answer-index comparison; Gemini is not asked to score submissions.
  const results = questions.map((q, i) => ({
    topic: String(q.topic || 'General').slice(0, 100),
    topicId: canonicalTopicId(q.topic || 'General'),
    question: String(q.question || ''),
    options: Array.isArray(q.options) ? q.options.map(String) : [],
    correct: Number(answers[i]) === Number(q.answer),
    selected: Number(answers[i]),
    answer: Number(q.answer),
    explanation: String(q.explanation || ''),
    sourceId: String(q.sourceId || ''),
    evidence: String(q.evidence || ''),
    evidenceCheck: q.evidenceCheck || { status: 'unchecked' },
    difficulty: String(q.difficulty || 'steady'),
  }));
  const mastery = (db.masteryByMaterial[material.id] ||= {});
  for (const r of results)
    mastery[r.topic] = {
      ...updateBkt(mastery[r.topic], r.correct, now),
      topicId: r.topicId,
      label: r.topic,
    };
  const correct = results.filter((r) => r.correct).length;
  const attempt = {
    id: crypto.randomUUID(),
    materialId: material.id,
    materialName: material.name,
    title: String(title || 'Learning check-in').slice(0, 120),
    score: Math.round((correct / results.length) * 100),
    correct,
    total: results.length,
    results,
    createdAt: now,
  };
  db.attempts.unshift(attempt);
  await writeDb(db);
  res.json({ attempt, mastery });
});
app.use('/api', (_req, res) => res.status(404).json({ error: 'API route not found.' }));
// Serve the built React app from the same origin as the API for the production preview.
app.use(express.static(path.join(ROOT, 'dist')));
app.get('*', (_req, res) => res.sendFile(path.join(ROOT, 'dist', 'index.html')));
app.listen(port, () => console.log(`EchoPilot API listening on http://localhost:${port}`));
