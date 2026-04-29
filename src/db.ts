// @ts-ignore sql.js has no type declarations
import initSqlJs from "sql.js";
import * as fs from "fs";
import * as path from "path";
import * as yaml from "js-yaml";
import { DB_PATH } from "./types.js";

export type Database = any;

const SCHEMA = `
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  source_type TEXT NOT NULL DEFAULT 'capture'
    CHECK (source_type IN ('capture', 'transcript', 'bootstrap')),
  epistemic_tag TEXT NOT NULL DEFAULT 'assertion'
    CHECK (epistemic_tag IN ('assertion', 'hypothesis', 'speculation', 'quoting', 'rejected')),
  confidence TEXT NOT NULL DEFAULT 'tentative'
    CHECK (confidence IN ('strong', 'tentative', 'uncertain')),
  superseded_by TEXT REFERENCES chunks(id),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS nodes (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL CHECK (type IN (
    'idea', 'question', 'project',
    'heuristic', 'value', 'mental_model', 'assumption', 'tension', 'preference'
  )),
  summary TEXT NOT NULL,
  confidence TEXT NOT NULL DEFAULT 'tentative'
    CHECK (confidence IN ('strong', 'tentative', 'uncertain')),
  activation REAL NOT NULL DEFAULT 1.0,
  first_seen TEXT NOT NULL DEFAULT (datetime('now')),
  last_accessed TEXT NOT NULL DEFAULT (datetime('now')),
  outcome_score REAL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS node_chunks (
  node_id TEXT NOT NULL REFERENCES nodes(id),
  chunk_id TEXT NOT NULL REFERENCES chunks(id),
  PRIMARY KEY (node_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS edges (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL CHECK (type IN (
    'supports', 'contradicts', 'evolved_into', 'inspired_by',
    'depends_on', 'overrides', 'learned_from', 'scoped_by',
    'rejected', 'belongs_to', 'derived_from'
  )),
  source_node_id TEXT NOT NULL REFERENCES nodes(id),
  target_node_id TEXT NOT NULL REFERENCES nodes(id),
  weight REAL NOT NULL DEFAULT 1.0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS outcomes (
  id TEXT PRIMARY KEY,
  node_id TEXT NOT NULL REFERENCES nodes(id),
  outcome TEXT NOT NULL,
  score REAL NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id TEXT PRIMARY KEY REFERENCES chunks(id),
  vector TEXT NOT NULL,
  model TEXT NOT NULL,
  dims INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS framing (
  id TEXT PRIMARY KEY,
  question TEXT NOT NULL,
  rationale TEXT,
  priority INTEGER NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY,
  tool_name TEXT NOT NULL,
  query_text TEXT,
  node_ids_touched TEXT,
  agent_name TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_activation ON nodes(activation DESC);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type);
CREATE INDEX IF NOT EXISTS idx_outcomes_node ON outcomes(node_id);
CREATE INDEX IF NOT EXISTS idx_framing_priority ON framing(priority);
`;

function ensureDir(filePath: string) {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

export let db: Database;

export async function initDb(): Promise<Database> {
  const SQL = await initSqlJs();
  ensureDir(DB_PATH);
  if (fs.existsSync(DB_PATH)) {
    const buf = fs.readFileSync(DB_PATH);
    db = new SQL.Database(buf);
  } else {
    db = new SQL.Database();
  }
  db.run("PRAGMA foreign_keys = ON;");
  db.run(SCHEMA);
  seedDefaultFraming();
  persistDb();
  return db;
}

/**
 * Seeds framing directives from framing.yaml if the framing table is empty.
 * framing.yaml ships with starter directives — users replace them with their own.
 */
function seedDefaultFraming() {
  const existing = db.exec("SELECT COUNT(*) FROM framing");
  const count = existing.length > 0 ? (existing[0].values[0][0] as number) : 0;
  if (count > 0) return;

  const candidates = [
    path.resolve(__dirname, "..", "framing.yaml"),
    path.resolve(__dirname, "..", "..", "framing.yaml"),
  ];
  const yamlPath = candidates.find((p) => fs.existsSync(p));
  if (!yamlPath) {
    console.warn("framing.yaml not found — use seed_framing to add directives.");
    return;
  }

  const raw = fs.readFileSync(yamlPath, "utf-8");
  const parsed = yaml.load(raw) as { directives?: Array<{ id: string; question: string; rationale?: string; priority: number; version?: number }> };
  if (!parsed?.directives?.length) return;

  for (const d of parsed.directives) {
    db.run(
      "INSERT INTO framing (id, question, rationale, priority, version) VALUES (?, ?, ?, ?, ?)",
      [d.id, d.question, d.rationale ?? null, d.priority, d.version ?? 1]
    );
  }
}

export function persistDb() {
  const data = db.export();
  const buffer = Buffer.from(data);
  const tmp = DB_PATH + ".tmp";
  ensureDir(DB_PATH);
  fs.writeFileSync(tmp, buffer);
  fs.renameSync(tmp, DB_PATH);
}

export function uuid(): string {
  return crypto.randomUUID();
}

export function now(): string {
  return new Date().toISOString().replace("T", " ").slice(0, 19);
}

/**
 * Parameterized single-row read. Uses try/finally to prevent stmt memory
 * leak if bind() throws. sql.js's exec() does not accept parameters —
 * use this helper for any read that takes user-supplied or variable input.
 */
export function queryOne(sql: string, params: any[]): Record<string, any> | null {
  const stmt = db.prepare(sql);
  try {
    stmt.bind(params);
    return stmt.step() ? stmt.getAsObject() : null;
  } finally {
    stmt.free();
  }
}

export function queryScalar(sql: string, params: any[]): any {
  const row = queryOne(sql, params);
  return row ? Object.values(row)[0] : null;
}

export function getNode(nodeId: string): Record<string, unknown> | null {
  return queryOne("SELECT * FROM nodes WHERE id = ?", [nodeId]);
}

export function queryNodes(sql: string): Record<string, unknown>[] {
  const rows = db.exec(sql);
  if (rows.length === 0) return [];
  return rows[0].values.map((vals: any[]) => {
    const obj: Record<string, unknown> = {};
    rows[0].columns.forEach((c: string, i: number) => { obj[c] = vals[i]; });
    return obj;
  });
}

export function bumpActivation(nodeId: string) {
  db.run("UPDATE nodes SET activation = MIN(activation + 0.1, 1.0), last_accessed = ? WHERE id = ?", [now(), nodeId]);
}
