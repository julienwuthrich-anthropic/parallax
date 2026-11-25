// Workspace search: a full-text index of passages, inside the workspace's own SQLite file.
//
// - The index is SQLite FTS5 with BM25 ranking: no extra dependency, no network, no model.
// - A passage is a chunk of one document: a record (with its provenance, fields and a metric
//   summary), an attached file (its caption, and its text when the format is text) or a run
//   (name, notes, config, summary). Traces are not indexed.
// - The index is brought up to date at the start of every search, by comparing a stamp per
//   document. So it also follows writes of another process (the MCP server, an older Parallax)
//   and needs no hook in the write paths.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { z } from "zod";
import { kinds, type Artifact, type ResearchRecord } from "./schema";
import { db, getRecord, listArtifacts, transaction } from "./store";
import { workspaceDataDir } from "./workspaces";

/** Part of every stamp: raise it when the text of a document or the chunking changes. */
const INDEX_VERSION = 1;
const CHUNK_CHARS = 1500;
const MAX_FILE_CHARS = 200_000;
const MAX_CHUNKS_PER_DOCUMENT = 2;

export const passageKinds = [...kinds, "file", "run"] as const;
export type PassageKind = (typeof passageKinds)[number];

export const passageQuerySchema = z.object({
  query: z.string().trim().min(1).max(500),
  kind: z.enum(passageKinds).optional(),
  limit: z.number().int().min(1).max(30).default(8),
});

export type Passage = {
  /** What the passage belongs to. `id` is a record id, an artifact id or a run id. */
  source: "record" | "file" | "run";
  kind: PassageKind;
  id: string;
  /** The records to open for this passage: the record itself, or the ones a file or run is linked to. */
  recordIds: string[];
  title: string;
  /** Records only: how far the claim is checked, and its state. */
  evidence: string | null;
  status: string | null;
  updatedAt: string | null;
  /** The whole chunk, and the part around the matched words («…» marks a match): of the
   *  chunk, or of the title when only the title matches. */
  text: string;
  snippet: string;
  chunk: number;
  /** BM25, higher is better. Only comparable inside one result list. */
  score: number;
};

const ready = new WeakSet<DatabaseSync>();
function index() {
  const database = db();
  if (!ready.has(database)) {
    database.exec(`
      CREATE TABLE IF NOT EXISTS search_docs (doc TEXT PRIMARY KEY, stamp TEXT NOT NULL) WITHOUT ROWID;
      CREATE VIRTUAL TABLE IF NOT EXISTS search_chunks USING fts5(title, body, doc UNINDEXED, kind UNINDEXED, seq UNINDEXED, tokenize='porter unicode61');
    `);
    ready.add(database);
  }
  return database;
}

// ── text of a document ──────────────────────────────────────────────────────────
/** Split on blank lines; a chunk ends before a heading or when the next block does not fit. */
export function chunkText(text: string, size = CHUNK_CHARS): string[] {
  const chunks: string[] = [];
  let current = "";
  const flush = () => {
    if (current.trim()) chunks.push(current.trim());
    current = "";
  };
  for (const block of text.split(/\n{2,}/)) {
    const piece = block.trim();
    if (!piece) continue;
    const heading = /^#{1,6}\s/.test(piece);
    if (
      current &&
      (current.length + piece.length + 2 > size ||
        (heading && current.length > size / 4))
    )
      flush();
    if (piece.length <= size) {
      current = current ? `${current}\n\n${piece}` : piece;
      continue;
    }
    // One block longer than a chunk (a table, a log): cut at the last space that fits.
    let rest = piece;
    while (rest.length > size) {
      const cut = rest.lastIndexOf(" ", size);
      const at = cut > size / 2 ? cut : size;
      chunks.push(rest.slice(0, at).trim());
      rest = rest.slice(at).trim();
    }
    current = rest;
  }
  flush();
  return chunks;
}

const line = (label: string, value: unknown) =>
  value === "" || value === null || value === undefined
    ? ""
    : `${label}: ${typeof value === "string" ? value : JSON.stringify(value)}`;

/** The last value and the range of each metric: what a question about a result needs. */
function metricSummary(metrics: ResearchRecord["metrics"]) {
  const byName = new Map<string, ResearchRecord["metrics"]>();
  for (const m of metrics)
    byName.set(m.name, [...(byName.get(m.name) ?? []), m]);
  return [...byName]
    .slice(0, 40)
    .map(([name, points]) => {
      const last = points.reduce((a, b) => (b.step >= a.step ? b : a));
      const values = points.map((p) => p.value);
      return points.length === 1
        ? `${name} = ${last.value}${last.unit ?? ""} (step ${last.step})`
        : `${name}: last ${last.value}${last.unit ?? ""} at step ${last.step}, min ${Math.min(...values)}, max ${Math.max(...values)}, ${points.length} points`;
    })
    .join("\n");
}

function recordText(record: ResearchRecord) {
  const p = record.provenance;
  const head = [
    `${record.kind}, status ${record.status}, evidence ${record.evidence}`,
    line("tags", record.tags.join(", ")),
    line("repository", p.repository),
    line("commit", p.commit),
    line("seed", p.seed),
    line("model", p.model),
    line("dataset", p.dataset),
    line("command", p.command),
    line("config", p.config.slice(0, 2000)),
    line("source file", p.sourcePath),
    ...Object.entries(record.fields ?? {}).map(([key, value]) =>
      line(key, value),
    ),
    line("gate", record.gate),
    record.kind === "experiment" ? line("decision", record.decision) : "",
    line("cost USD", record.costUsd),
    line("GPU hours", record.gpuHours),
    record.metrics.length ? `metrics:\n${metricSummary(record.metrics)}` : "",
  ].filter(Boolean);
  return `${head.join("\n")}\n\n${record.content}`;
}

const TEXT_MIMES = new Set([
  "text/markdown",
  "text/plain",
  "text/csv",
  "text/tab-separated-values",
  "application/json",
]);

function notebookText(raw: string) {
  const cells = (JSON.parse(raw).cells ?? []) as {
    source?: string | string[];
    outputs?: { text?: string | string[] }[];
  }[];
  const join = (value?: string | string[]) =>
    Array.isArray(value) ? value.join("") : (value ?? "");
  return cells
    .map((cell) =>
      [join(cell.source), ...(cell.outputs ?? []).map((o) => join(o.text))]
        .filter(Boolean)
        .join("\n"),
    )
    .join("\n\n");
}

const htmlText = (raw: string) =>
  raw
    .replace(/<(script|style)\b[\s\S]*?<\/\1>/gi, " ")
    .replace(/<\/(p|div|h[1-6]|li|tr|section)>/gi, "\n\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/[ \t]+/g, " ");

/** The caption of every file; the content too when the format is text. An image or a PDF is
 *  found by its name and caption only. */
function artifactText(artifact: Artifact) {
  let body = "";
  const textual =
    TEXT_MIMES.has(artifact.mime) ||
    artifact.mime === "application/x-ipynb+json" ||
    artifact.mime === "text/html";
  if (textual) {
    const raw = readFileSync(
      path.join(workspaceDataDir(), "artifacts", artifact.hash),
      "utf8",
    );
    try {
      body =
        artifact.mime === "application/x-ipynb+json"
          ? notebookText(raw)
          : artifact.mime === "text/html"
            ? htmlText(raw)
            : raw;
    } catch {
      body = raw; // a notebook that is not valid JSON: index it as plain text
    }
  }
  return [line("caption", artifact.caption), body.slice(0, MAX_FILE_CHARS)]
    .filter(Boolean)
    .join("\n\n");
}

type RunRow = {
  id: string;
  name: string;
  run_group: string;
  tags: string;
  notes: string;
  status: string;
  config: string;
  summary: string;
  experiment_id: string | null;
};
const runText = (run: RunRow) =>
  [
    `training run ${run.id}, status ${run.status}`,
    line("group", run.run_group),
    line("tags", run.tags === "[]" ? "" : run.tags),
    line("notes", run.notes),
    line("config", run.config === "{}" ? "" : run.config.slice(0, 4000)),
    line("summary", run.summary === "{}" ? "" : run.summary.slice(0, 2000)),
  ]
    .filter(Boolean)
    .join("\n");

// ── keeping the index current ───────────────────────────────────────────────────
type Document = {
  stamp: string;
  kind: PassageKind;
  title: string;
  text: () => string;
};

function documents() {
  const database = index();
  const wanted = new Map<string, Document>();
  for (const row of database
    .prepare(
      "SELECT id, json_extract(payload, '$.revision') AS revision, json_extract(payload, '$.kind') AS kind, json_extract(payload, '$.title') AS title FROM records",
    )
    .all() as {
    id: string;
    revision: number;
    kind: PassageKind;
    title: string;
  }[])
    wanted.set(`record:${row.id}`, {
      stamp: `${INDEX_VERSION}:${row.revision}`,
      kind: row.kind,
      title: row.title,
      text: () => recordText(getRecord(row.id)),
    });
  for (const artifact of listArtifacts())
    wanted.set(`file:${artifact.id}`, {
      stamp: `${INDEX_VERSION}:${artifact.hash}`,
      kind: "file",
      title: artifact.name,
      text: () => artifactText(artifact),
    });
  for (const run of database
    .prepare(
      "SELECT id, name, run_group, tags, notes, status, config, summary, experiment_id FROM runs",
    )
    .all() as RunRow[]) {
    const text = runText(run);
    wanted.set(`run:${run.id}`, {
      stamp: `${INDEX_VERSION}:${createHash("sha1").update(`${run.name}\n${text}`).digest("hex").slice(0, 16)}`,
      kind: "run",
      title: run.name || run.id,
      text: () => text,
    });
  }
  return wanted;
}

/** Index what changed since the last search. Returns how many documents were written or removed. */
export function syncSearchIndex() {
  const database = index();
  const wanted = documents();
  const stored = new Map(
    (
      database.prepare("SELECT doc, stamp FROM search_docs").all() as {
        doc: string;
        stamp: string;
      }[]
    ).map((row) => [row.doc, row.stamp]),
  );
  const changed = [...wanted].filter(
    ([doc, document]) => stored.get(doc) !== document.stamp,
  );
  const removed = [...stored.keys()].filter((doc) => !wanted.has(doc));
  if (!changed.length && !removed.length) return 0;
  return transaction(() => {
    const dropChunks = database.prepare(
      "DELETE FROM search_chunks WHERE doc = ?",
    );
    const dropDoc = database.prepare("DELETE FROM search_docs WHERE doc = ?");
    const addChunk = database.prepare(
      "INSERT INTO search_chunks(title, body, doc, kind, seq) VALUES (?, ?, ?, ?, ?)",
    );
    const addDoc = database.prepare(
      "INSERT INTO search_docs(doc, stamp) VALUES (?, ?) ON CONFLICT(doc) DO UPDATE SET stamp = excluded.stamp",
    );
    for (const doc of removed) {
      dropChunks.run(doc);
      dropDoc.run(doc);
    }
    for (const [doc, document] of changed) {
      if (stored.has(doc)) dropChunks.run(doc);
      // A document with no text still gets one chunk, so its title is searchable.
      const chunks = chunkText(document.text());
      (chunks.length ? chunks : [""]).forEach((body, seq) =>
        addChunk.run(document.title, body, doc, document.kind, seq),
      );
      addDoc.run(doc, document.stamp);
    }
    return changed.length + removed.length;
  });
}

// ── search ──────────────────────────────────────────────────────────────────────
/** Free text → an FTS5 query. Every word is optional (OR): BM25 puts the chunks with the most
 *  and the rarest words first. A "quoted phrase" must match as a phrase, and a word with
 *  punctuation inside (`loss/val`, `gpt-6`) also counts as a phrase. Null: nothing to search. */
export function ftsQuery(text: string): string | null {
  const terms = new Set<string>();
  const words = (value: string) => value.match(/[\p{L}\p{N}]+/gu) ?? [];
  const rest = text.replace(/"([^"]+)"/g, (_, phrase: string) => {
    const tokens = words(phrase);
    if (tokens.length) terms.add(`"${tokens.join(" ")}"`);
    return " ";
  });
  for (const word of rest.split(/\s+/)) {
    const tokens = words(word);
    if (tokens.length > 1) terms.add(`"${tokens.join(" ")}"`);
    for (const token of tokens) terms.add(`"${token}"`);
  }
  return terms.size ? [...terms].join(" OR ") : null;
}

/** The best passages of the workspace for a question or a list of keywords. */
export function searchPassages(input: unknown): Passage[] {
  const { query, kind, limit } = passageQuerySchema.parse(input);
  const match = ftsQuery(query);
  if (!match) return [];
  syncSearchIndex();
  const database = index();
  const rows = database
    .prepare(
      `SELECT doc, kind, seq, title, body,
         snippet(search_chunks, -1, '«', '»', ' … ', 28) AS snippet,
         bm25(search_chunks, 4.0, 1.0) AS rank
       FROM search_chunks WHERE search_chunks MATCH ?1 ${kind ? "AND kind = ?3" : ""}
       ORDER BY rank LIMIT ?2`,
    )
    .all(...(kind ? [match, limit * 6, kind] : [match, limit * 6])) as {
    doc: string;
    kind: PassageKind;
    seq: number;
    title: string;
    body: string;
    snippet: string;
    rank: number;
  }[];
  const perDocument = new Map<string, number>();
  const best = rows.filter((row) => {
    const seen = perDocument.get(row.doc) ?? 0;
    perDocument.set(row.doc, seen + 1);
    return seen < MAX_CHUNKS_PER_DOCUMENT;
  });
  const artifacts = new Map(listArtifacts().map((a) => [a.id, a.recordIds]));
  const recordMeta = database.prepare(
    "SELECT json_extract(payload, '$.evidence') AS evidence, json_extract(payload, '$.status') AS status, json_extract(payload, '$.updatedAt') AS updatedAt FROM records WHERE id = ?",
  );
  const runExperiment = database.prepare(
    "SELECT experiment_id FROM runs WHERE id = ?",
  );
  return best.slice(0, limit).map((row) => {
    const at = row.doc.indexOf(":");
    const source = row.doc.slice(0, at) as Passage["source"];
    const id = row.doc.slice(at + 1);
    const meta =
      source === "record"
        ? (recordMeta.get(id) as
            { evidence: string; status: string; updatedAt: string } | undefined)
        : undefined;
    const experiment =
      source === "run"
        ? (
            runExperiment.get(id) as
              { experiment_id: string | null } | undefined
          )?.experiment_id
        : null;
    return {
      source,
      kind: row.kind,
      id,
      recordIds:
        source === "record"
          ? [id]
          : source === "file"
            ? (artifacts.get(id) ?? [])
            : experiment
              ? [experiment]
              : [],
      title: row.title,
      evidence: meta?.evidence ?? null,
      status: meta?.status ?? null,
      updatedAt: meta?.updatedAt ?? null,
      text: row.body,
      snippet: row.snippet,
      chunk: row.seq,
      score: Math.round(-row.rank * 1000) / 1000,
    };
  });
}
