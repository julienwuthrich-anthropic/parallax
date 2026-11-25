import { test, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";

const temp = mkdtempSync(path.join(os.tmpdir(), "parallax-search-test-"));
mkdirSync(path.join(temp, "research"));
process.env.PARALLAX_DATA_DIR = path.join(temp, "private-data");
process.env.PARALLAX_RESEARCH_ROOT = path.join(temp, "research");
const store = await import("../lib/store");
const runs = await import("../lib/runs");
const spaces = await import("../lib/workspaces");
const search = await import("../lib/search");
after(() => {
  store.closeDatabases();
  spaces.closeCatalog();
  rmSync(temp, { recursive: true, force: true });
});

test("chunks end at headings and no chunk is longer than the limit", () => {
  const text = [
    "# One",
    "alpha ".repeat(120).trim(),
    "## Two",
    "beta ".repeat(700).trim(),
  ].join("\n\n");
  const chunks = search.chunkText(text, 1500);
  assert.ok(chunks.length >= 3);
  assert.ok(chunks.every((chunk) => chunk.length <= 1500));
  assert.match(chunks[0], /^# One/);
  assert.match(chunks[1], /^## Two/);
  assert.deepEqual(search.chunkText("  \n\n "), []);
});

test("free text becomes a query of optional words and phrases", () => {
  assert.equal(
    search.ftsQuery('val loss "weak teacher"'),
    '"weak teacher" OR "val" OR "loss"',
  );
  assert.equal(search.ftsQuery("loss/val"), '"loss val" OR "loss" OR "val"');
  // FTS5 operators and quotes in the input are words or nothing, never syntax.
  assert.equal(search.ftsQuery('NEAR( " * ^'), '"NEAR"');
  assert.equal(search.ftsQuery("?!"), null);
  assert.deepEqual(search.searchPassages({ query: "?!" }), []);
});

test("passages come from records, files and runs, best first, and follow edits", () => {
  const finding = store.createRecord({
    kind: "finding",
    title: "Fixture: sycophancy rises under pressure",
    evidence: "observed",
    content:
      "# Result\n\nThe judged models agreed with wrong users more often when the prompt added time pressure.\n\n# Limits\n\nOne seed only.",
    tags: ["pressure"],
  });
  const experiment = store.createRecord({
    kind: "experiment",
    title: "Fixture: baseline sweep",
    content: "Learning rate sweep on the small model.",
    provenance: { seed: 7, commit: "abc1234", model: "qwen-small" },
    metrics: [
      { name: "loss/val", step: 100, value: 1.9 },
      { name: "loss/val", step: 200, value: 1.51 },
    ],
  });
  store.addArtifact(
    Buffer.from("# Notes\n\nThe zebra ablation removed the reward head."),
    "ablation.md",
    { recordId: experiment.id, caption: "Ablation notes" },
  );
  store.addArtifact(Buffer.from([0x89, 0x50, 0x4e, 0x47]), "curve.png", {
    caption: "Validation curve of the kangaroo run",
  });
  runs.applyRunEvents([
    {
      type: "start",
      run: "fixture-run-1",
      name: "wombat-lr3e4",
      config: { optimizer: "adamw", lr: 0.0003 },
      experimentId: experiment.id,
    },
  ]);

  // Stemming: "agree" finds "agreed"; the title counts more than the body.
  const [top] = search.searchPassages({ query: "models agree with users" });
  assert.equal(top.id, finding.id);
  assert.equal(top.source, "record");
  assert.equal(top.kind, "finding");
  assert.equal(top.evidence, "observed");
  assert.deepEqual(top.recordIds, [finding.id]);
  assert.match(top.snippet, /«agreed»/);
  assert.ok(top.score > 0);

  // Provenance and the metric summary are part of the record's text.
  assert.equal(
    search.searchPassages({ query: "abc1234" })[0].id,
    experiment.id,
  );
  const metric = search.searchPassages({ query: "loss/val" })[0];
  assert.equal(metric.id, experiment.id);
  assert.match(
    metric.text,
    /loss\/val: last 1\.51 at step 200, min 1\.51, max 1\.9, 2 points/,
  );

  // A text file by its content, an image by its caption, a run by its config.
  const file = search.searchPassages({ query: "zebra" })[0];
  assert.deepEqual(
    [file.source, file.kind, file.title, file.recordIds],
    ["file", "file", "ablation.md", [experiment.id]],
  );
  assert.equal(
    search.searchPassages({ query: "kangaroo" })[0].title,
    "curve.png",
  );
  const run = search.searchPassages({ query: "adamw" })[0];
  assert.deepEqual(
    [run.source, run.id, run.title, run.recordIds],
    ["run", "fixture-run-1", "wombat-lr3e4", [experiment.id]],
  );

  // The kind filter, and at most two chunks of one document.
  assert.deepEqual(
    search
      .searchPassages({ query: "pressure sweep zebra", kind: "experiment" })
      .map((p) => p.id),
    [experiment.id],
  );
  const long = store.createRecord({
    kind: "note",
    title: "Fixture: long note",
    content: Array.from(
      { length: 6 },
      (_, i) => `# Part ${i}\n\n${"quokka filler words ".repeat(60)}`,
    ).join("\n\n"),
  });
  assert.equal(
    search.searchPassages({ query: "quokka" }).filter((p) => p.id === long.id)
      .length,
    2,
  );

  // An edit replaces the passages; a delete removes them; nothing to do is no write.
  assert.equal(search.syncSearchIndex(), 0);
  store.updateRecord(
    finding.id,
    { ...finding, content: "Retracted: the effect came from a prompt bug." },
    finding.revision,
  );
  assert.equal(
    search.searchPassages({ query: "agreed wrong users" }).length,
    0,
  );
  assert.equal(search.searchPassages({ query: "retracted" })[0].id, finding.id);
  store.deleteRecord(long.id, long.revision);
  assert.equal(search.searchPassages({ query: "quokka" }).length, 0);
});

test("a workspace never returns the passages of another one", () => {
  const other = spaces.createWorkspace({ name: "Fixture second project" });
  spaces.runInWorkspace(other.id, () => {
    assert.equal(search.searchPassages({ query: "zebra retracted" }).length, 0);
    store.createRecord({ kind: "note", title: "Fixture: okapi only here" });
    assert.equal(search.searchPassages({ query: "okapi" }).length, 1);
  });
  assert.equal(search.searchPassages({ query: "okapi" }).length, 0);
});
