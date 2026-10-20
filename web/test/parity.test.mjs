// Checks the JS port against what the Python pipeline exported (finrag export-demo).
// FINRAG_DEMO_DIR points at an export; CI exports the synthetic fixture first.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { join } from "node:path";
import { Engine, makeTokenizer } from "../finrag.mjs";

const dir = process.env.FINRAG_DEMO_DIR || "test/data";
const read = (name) => JSON.parse(readFileSync(join(dir, name), "utf8"));
const meta = read("meta.json");
const parity = read("parity.json");
const questions = read("questions.json");
const loadFiling = async (doc) => JSON.parse(gunzipSync(readFileSync(join(dir, "filings", `${doc}.json.gz`))).toString("utf8"));

test("every demo page tokenizes to the token count Python recorded", async () => {
  const tokenize = makeTokenizer(meta.stopwords);
  let pages = 0;
  for (const f of meta.filings.filter((x) => x.demo)) {
    const raw = await loadFiling(f.doc);
    raw.pages.forEach((text, i) => assert.equal(tokenize(text).length, raw.lengths[i], `${f.doc} p${i}`));
    pages += raw.pages.length;
  }
  assert.ok(pages > 0);
});

test("filter picks the same filings as Python for every question", () => {
  const engine = new Engine(meta, loadFiling);
  const seen = new Set();
  for (const c of parity) {
    if (seen.has(c.id)) continue;
    seen.add(c.id);
    const q = questions.find((x) => x.id === c.id);
    assert.deepEqual(engine.filter.allowedDocs(q.q), c.allowed, c.id);
  }
  assert.ok(seen.size > 0);
});

test("top-10 pages and float32 scores match Python", async () => {
  const engine = new Engine(meta, loadFiling);
  for (const c of parity) {
    const q = questions.find((x) => x.id === c.id);
    const res = await engine.search(q.q, { k: 10, route: c.opts.route, drop: c.opts.drop });
    assert.deepEqual(res.hits.map((h) => [h.doc, h.page]), c.top, `${c.id} ${JSON.stringify(c.opts)}`);
    res.hits.forEach((h, i) => assert.equal(h.score, c.scores[i], `${c.id} rank ${i + 1}`));
  }
});
