import 'dotenv/config';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { bm25Rank, hybridRank } from './ai.js';

const directory = path.dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(
  await fs.readFile(path.join(directory, 'evaluation', 'retrieval-cases.json'), 'utf8')
);
const k = 7;

function metrics(rankedCases) {
  // Recall@k checks coverage; reciprocal rank rewards finding the target near rank one.
  const hits = rankedCases.filter(({ ranking, expected }) =>
    ranking.slice(0, k).some((item) => item.id === expected)
  ).length;
  const reciprocalRanks = rankedCases.map(({ ranking, expected }) => {
    const index = ranking.findIndex((item) => item.id === expected);
    return index < 0 ? 0 : 1 / (index + 1);
  });
  return {
    recallAtK: hits / rankedCases.length,
    mrr: reciprocalRanks.reduce((sum, value) => sum + value, 0) / reciprocalRanks.length,
  };
}

const lexical = fixture.cases.map((test) => ({
  expected: test.expected,
  ranking: bm25Rank(fixture.passages, test.query),
}));
const hybrid = [];
for (const test of fixture.cases) {
  hybrid.push({
    expected: test.expected,
    ranking: await hybridRank(fixture.passages, test.query, fixture.passages.length),
  });
}
const report = {
  queries: fixture.cases.length,
  corpusPassages: fixture.passages.length,
  k,
  bm25: metrics(lexical),
  localHybrid: metrics(hybrid),
};
console.log(JSON.stringify(report, null, 2));
