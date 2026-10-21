// The finrag BM25 pipeline in the browser: the company/year/quarter filter, BM25 with the
// corpus-wide idf, and statement routing. It ports src/finrag/{bm25,filters,statements,
// search}.py. The regexes come from the exported meta.json so both sides read questions the
// same way, and BM25 runs the same float64 arithmetic in the same order as Python, so
// test/parity.test.mjs can require identical rankings and identical scores.

const WORD = /[\p{L}\p{N}_]{2,}/gu; // Python: (?u)\b\w\w+\b

export function makeTokenizer(stopwords) {
  const stop = new Set(stopwords);
  return (text) => {
    const out = [];
    for (const m of text.toLowerCase().matchAll(WORD)) if (!stop.has(m[0])) out.push(m[0]);
    return out;
  };
}

const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().split(/\s+/).filter(Boolean).join(" ");

export class Filter {
  constructor(meta) {
    this.aliases = meta.aliases; // [normalized alias, company], longest first
    this.filings = meta.filings;
    const p = meta.patterns;
    this.year = new RegExp(p.year, "g");
    this.shortFy = new RegExp(p.short_fy, "g");
    this.quarter = p.quarter.map((src) => new RegExp(src, "gi"));
    this.ordinal = p.ordinal;
  }

  companiesIn(text) {
    let padded = ` ${norm(text)} `;
    const found = new Set();
    for (const [alias, company] of this.aliases) {
      const needle = ` ${alias} `;
      if (padded.includes(needle)) {
        found.add(company);
        padded = padded.split(needle).join(" ");
      }
    }
    return found;
  }

  companyTerms(companies) {
    const out = new Set();
    for (const [alias, company] of this.aliases)
      if (companies.has(company)) for (const w of alias.split(" ")) out.add(w);
    return out;
  }

  yearsIn(text) {
    const low = text.toLowerCase();
    const years = new Set();
    for (const m of low.matchAll(this.year)) years.add(Number(m[1]));
    for (const m of low.matchAll(this.shortFy)) years.add(2000 + Number(m[1]));
    return years;
  }

  quartersIn(text) {
    const found = new Set();
    this.quarter.forEach((rx, i) => {
      for (const m of text.matchAll(rx)) {
        let q, y;
        if (i === 0) [q, y] = [Number(m[1]), Number(m[2])];
        else if (i === 1) [y, q] = [Number(m[1]), Number(m[2])];
        else [q, y] = [this.ordinal[m[1].toLowerCase()], Number(m[2])];
        found.add(`${y < 100 ? y + 2000 : y}Q${q}`);
      }
    });
    return found;
  }

  // Same contract as MetadataFilter.allowed_docs: null means no company was found.
  allowedDocs(text, level = "company_year", quarters = true) {
    if (level === "none") return null;
    const companies = this.companiesIn(text);
    if (!companies.size) return null;
    let docs = this.filings.filter((f) => companies.has(f.company));
    if (level === "company_year") {
      const years = this.yearsIn(text);
      const byYear = docs.filter((f) => f.periods.some((p) => years.has(p)));
      if (byYear.length) docs = byYear;
      const qs = quarters ? this.quartersIn(text) : new Set();
      if (qs.size) {
        const byQuarter = docs.filter((f) => f.quarter && qs.has(`${f.quarter[0]}Q${f.quarter[1]}`));
        if (byQuarter.length) docs = byQuarter;
      }
    }
    return docs.map((f) => f.doc).sort();
  }
}

export function makeQuestionKinds(meta) {
  const compile = (obj) => Object.entries(obj).map(([k, src]) => [k, new RegExp(src, "i")]);
  const named = compile(meta.patterns.named);
  const implied = compile(meta.patterns.implied);
  return (question, useImplied = true) => {
    const out = new Set();
    for (const [k, rx] of named) if (rx.test(question)) out.add(k);
    if (useImplied) for (const [k, rx] of implied) if (rx.test(question)) out.add(k);
    return out;
  };
}

// A filing as the browser holds it: page text, token counts, statement kinds, and term
// frequencies counted once on load.
export function indexFiling(raw, tokenize, kindNames) {
  const tf = raw.pages.map((text, i) => {
    const counts = new Map();
    const tokens = tokenize(text);
    for (const t of tokens) counts.set(t, (counts.get(t) || 0) + 1);
    if (tokens.length !== raw.lengths[i]) throw new Error(`${raw.doc} p${i}: ${tokens.length} tokens, expected ${raw.lengths[i]}`);
    return counts;
  });
  const kinds = raw.kinds.map((s) => new Set([...s].map((c) => kindNames[c])));
  return { doc: raw.doc, pages: raw.pages, lengths: raw.lengths, kinds, tf, idf: raw.idf };
}

// BM25 over the pages of `filings`, given in global row order. Per page and query token this
// is the expression bm25.py evaluates, in the same order, so the float64 results are equal.
export function bm25(queryTokens, filings, meta) {
  const { k1, b, avg_len: avg } = meta;
  const rows = [];
  for (const f of filings)
    for (let p = 0; p < f.pages.length; p++) rows.push({ doc: f.doc, page: p, f, score: 0 });
  for (const t of queryTokens) {
    for (const r of rows) {
      const tf = r.f.tf[r.page].get(t);
      if (!tf) continue;
      const normLen = k1 * (1.0 - b + (b * r.f.lengths[r.page]) / avg);
      r.score += (r.f.idf[t] * tf) / (tf + normLen);
    }
  }
  return rows;
}

export function rank(rows, wanted) {
  // Ties go to the earlier page in global order, which is the order rows arrive in.
  const ranked = rows.map((r, i) => ({ ...r, order: i })).sort((x, y) => y.score - x.score || x.order - y.order);
  if (!wanted || !wanted.size) return ranked;
  const hit = (r) => [...r.f.kinds[r.page]].some((k) => wanted.has(k));
  return [...ranked.filter(hit), ...ranked.filter((r) => !hit(r))];
}

export class Engine {
  constructor(meta, loadFiling) {
    this.meta = meta;
    this.tokenize = makeTokenizer(meta.stopwords);
    this.filter = new Filter(meta);
    this.questionKinds = makeQuestionKinds(meta);
    this.loadFiling = loadFiling; // async (doc) => raw filing JSON
    this.cache = new Map();
    this.demo = new Set(meta.filings.filter((f) => f.demo).map((f) => f.doc));
  }

  async filing(doc) {
    if (!this.cache.has(doc)) {
      this.cache.set(doc, this.loadFiling(doc).then((raw) => indexFiling(raw, this.tokenize, this.meta.kinds)));
    }
    return this.cache.get(doc);
  }

  async search(question, { k = 10, route = true, drop = true, quarters = true } = {}) {
    const t0 = performance.now();
    const allowed = this.filter.allowedDocs(question, "company_year", quarters);
    if (!allowed) return { allowed: null, loaded: [], hits: [], wanted: [], tokens: [], pages: 0, loadMs: 0, searchMs: 0 };
    const loaded = allowed.filter((d) => this.demo.has(d));
    const filings = await Promise.all(loaded.map((d) => this.filing(d)));
    const tLoaded = performance.now();
    let tokens = this.tokenize(question);
    if (drop) {
      const terms = this.filter.companyTerms(this.filter.companiesIn(question));
      tokens = tokens.filter((t) => !terms.has(t));
    }
    const wanted = route ? this.questionKinds(question) : new Set();
    const ranked = rank(bm25(tokens, filings, this.meta), wanted);
    return {
      allowed, loaded, wanted: [...wanted], tokens, pages: ranked.length,
      hits: ranked.slice(0, k).map(({ doc, page, score, f }) => ({ doc, page, score, kinds: [...f.kinds[page]], text: f.pages[page] })),
      loadMs: tLoaded - t0, searchMs: performance.now() - tLoaded,
    };
  }
}
