import { test } from "node:test";
import assert from "node:assert/strict";
import { Filter, bm25, makeTokenizer, rank } from "../finrag.mjs";

const STOP = ["the", "in", "of", "and"];

test("tokenizer keeps runs of two or more word characters, lowercased, minus stopwords", () => {
  const tok = makeTokenizer(STOP);
  assert.deepEqual(tok("The FY2021 capex, in $ millions: 1,234!"), ["fy2021", "capex", "millions", "234"]);
  assert.deepEqual(tok("Société Générale's año"), ["société", "générale", "año"]);
  assert.deepEqual(tok("a I x"), []);
});

const meta = {
  aliases: [["american water works", "American Water Works"], ["american express", "American Express"],
            ["jnj", "Johnson & Johnson"]],
  filings: [
    { doc: "AMERICANEXPRESS_2022_10K", company: "American Express", periods: [2022], quarter: null },
    { doc: "JOHNSON_JOHNSON_2022_10K", company: "Johnson & Johnson", periods: [2022], quarter: null },
    { doc: "JOHNSON_JOHNSON_2023Q2_EARNINGS", company: "Johnson & Johnson", periods: [2023], quarter: [2023, 2] },
  ],
  patterns: {
    year: String.raw`\b(?:fy\s?)?((?:19|20)\d{2})\b`,
    short_fy: String.raw`\bfy\s?'?(\d{2})\b`,
    quarter: [String.raw`\bq([1-4])\s*(?:of\s+)?(?:fy\s*|fiscal\s+(?:year\s+)?)?'?((?:20)?\d{2})\b`,
              String.raw`\b(20\d{2})\s*q([1-4])\b`],
    ordinal: {},
  },
};

test("filter reads company, year and quarter from the question", () => {
  const f = new Filter(meta);
  assert.deepEqual([...f.companiesIn("American Express card spend")], ["American Express"]);
  assert.deepEqual([...f.yearsIn("Compare FY21 with fiscal 2022")].sort(), [2021, 2022]);
  assert.deepEqual(f.allowedDocs("JnJ sales in Q2 of FY2023"), ["JOHNSON_JOHNSON_2023Q2_EARNINGS"]);
  assert.deepEqual(f.allowedDocs("JnJ FY2019 sales"), ["JOHNSON_JOHNSON_2022_10K", "JOHNSON_JOHNSON_2023Q2_EARNINGS"]);
  assert.equal(f.allowedDocs("Which segment grew?"), null);
});

test("bm25 is the Lucene formula, summed in query order", () => {
  const f = { doc: "D", pages: ["x", "y"], lengths: [10, 30], tf: [new Map([["capex", 2]]), new Map([["capex", 1]])],
              idf: { capex: 1.5 } };
  const rows = bm25(["capex", "capex"], [f], { k1: 1.5, b: 0.75, avg_len: 20 });
  const w = (tf, len) => (1.5 * tf) / (tf + 1.5 * (1.0 - 0.75 + (0.75 * len) / 20));
  assert.equal(rows[0].score, w(2, 10) + w(2, 10));
  assert.equal(rows[1].score, w(1, 30) + w(1, 30));
});

test("rank breaks ties by page order and routing keeps relative order", () => {
  const f = { kinds: [new Set(), new Set(["balance"]), new Set()] };
  const rows = [0, 1, 2].map((page) => ({ doc: "D", page, f, score: page === 1 ? 1 : 2 }));
  assert.deepEqual(rank(rows).map((r) => r.page), [0, 2, 1]);
  assert.deepEqual(rank(rows, new Set(["balance"])).map((r) => r.page), [1, 0, 2]);
});
