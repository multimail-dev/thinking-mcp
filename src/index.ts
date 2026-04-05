#!/usr/bin/env node
// thinking-mcp: MCP server that models how you think
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
// @ts-ignore sql.js has no type declarations
import initSqlJs from "sql.js";
type Database = any;
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type NodeType = "idea" | "question" | "project" | "heuristic" | "value" | "mental_model" | "assumption" | "tension" | "preference";
type EpistemicTag = "assertion" | "hypothesis" | "speculation" | "quoting" | "rejected";
type Confidence = "strong" | "tentative" | "uncertain";
type EdgeType = "supports" | "contradicts" | "evolved_into" | "inspired_by" | "depends_on" | "overrides" | "learned_from" | "scoped_by" | "rejected" | "belongs_to" | "derived_from";

const VALID_NODE_TYPES = new Set<string>(["idea", "question", "project", "heuristic", "value", "mental_model", "assumption", "tension", "preference"]);
const VALID_EPISTEMIC_TAGS = new Set<string>(["assertion", "hypothesis", "speculation", "quoting", "rejected"]);
const VALID_CONFIDENCE = new Set<string>(["strong", "tentative", "uncertain"]);
const VALID_EDGE_TYPES = new Set<string>(["supports", "contradicts", "evolved_into", "inspired_by", "depends_on", "overrides", "learned_from", "scoped_by", "rejected", "belongs_to", "derived_from"]);

const DECAY_RATES: Record<string, number> = {
  value: 0.98, assumption: 0.98,
  heuristic: 0.96, mental_model: 0.96, preference: 0.96, tension: 0.96,
  idea: 0.93, question: 0.93, project: 0.93,
};

interface ExtractedPattern {
  text: string;
  type: string;
  confidence: string;
  epistemic: string;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const DB_PATH = process.env.THINKING_MCP_DB_PATH
  || path.join(os.homedir(), ".thinking-mcp", "mind.db");
const EMBEDDING_PROVIDER = process.env.THINKING_MCP_EMBEDDING_PROVIDER || "voyage";
const VOYAGE_API_KEY = process.env.VOYAGE_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

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

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_activation ON nodes(activation DESC);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type);
CREATE INDEX IF NOT EXISTS idx_outcomes_node ON outcomes(node_id);
`;

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

let db: Database;

function ensureDir(filePath: string) {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

async function initDb(): Promise<Database> {
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
  persistDb();
  return db;
}

function persistDb() {
  const data = db.export();
  const buffer = Buffer.from(data);
  const tmp = DB_PATH + ".tmp";
  ensureDir(DB_PATH);
  fs.writeFileSync(tmp, buffer);
  fs.renameSync(tmp, DB_PATH);
}

function uuid(): string {
  return crypto.randomUUID();
}

function now(): string {
  return new Date().toISOString().replace("T", " ").slice(0, 19);
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

let embeddingModel = "";
let embeddingDims = 0;

async function embed(texts: string[]): Promise<number[][]> {
  if (EMBEDDING_PROVIDER === "voyage") {
    if (!VOYAGE_API_KEY) throw new Error("VOYAGE_API_KEY required for voyage embeddings");
    const resp = await fetch("https://api.voyageai.com/v1/embeddings", {
      method: "POST",
      headers: { "Authorization": `Bearer ${VOYAGE_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: "voyage-3-lite", input: texts, input_type: "document" }),
    });
    if (!resp.ok) throw new Error(`Voyage API error: ${resp.status} ${await resp.text()}`);
    const json = await resp.json() as { data: { embedding: number[] }[] };
    const vectors = json.data.map(d => d.embedding);
    if (vectors.length > 0) {
      embeddingModel = "voyage-3-lite";
      embeddingDims = vectors[0].length;
    }
    return vectors;
  }

  if (EMBEDDING_PROVIDER === "openai") {
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY required for openai embeddings");
    const resp = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: { "Authorization": `Bearer ${OPENAI_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: "text-embedding-3-small", input: texts }),
    });
    if (!resp.ok) throw new Error(`OpenAI API error: ${resp.status} ${await resp.text()}`);
    const json = await resp.json() as { data: { embedding: number[] }[] };
    const vectors = json.data.map(d => d.embedding);
    if (vectors.length > 0) {
      embeddingModel = "text-embedding-3-small";
      embeddingDims = vectors[0].length;
    }
    return vectors;
  }

  if (EMBEDDING_PROVIDER === "ollama") {
    const vectors: number[][] = [];
    for (const text of texts) {
      const resp = await fetch(`${OLLAMA_URL}/api/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "nomic-embed-text", prompt: text }),
      });
      if (!resp.ok) throw new Error(`Ollama error: ${resp.status}`);
      const json = await resp.json() as { embedding: number[] };
      vectors.push(json.embedding);
    }
    if (vectors.length > 0) {
      embeddingModel = "nomic-embed-text";
      embeddingDims = vectors[0].length;
    }
    return vectors;
  }

  throw new Error(`Unknown embedding provider: ${EMBEDDING_PROVIDER}`);
}

async function embedOne(text: string): Promise<number[] | null> {
  try {
    const result = await embed([text]);
    return result[0] ?? null;
  } catch (e) {
    console.error("Embedding error:", e);
    return null;
  }
}

function validateDims(vector: number[]): boolean {
  if (embeddingDims === 0) return true;
  return vector.length === embeddingDims;
}

// ---------------------------------------------------------------------------
// Vector search (brute-force cosine over stored embeddings)
// ---------------------------------------------------------------------------

function cosine(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom === 0 ? 0 : dot / denom;
}

function vectorSearch(queryVec: number[], topK = 20): { chunkId: string; score: number }[] {
  const rows = db.exec("SELECT chunk_id, vector FROM embeddings");
  if (rows.length === 0 || rows[0].values.length === 0) return [];

  const scored: { chunkId: string; score: number }[] = [];
  for (const row of rows[0].values) {
    const chunkId = row[0] as string;
    const vec = JSON.parse(row[1] as string) as number[];
    if (vec.length !== queryVec.length) continue;
    scored.push({ chunkId, score: cosine(queryVec, vec) });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}

// ---------------------------------------------------------------------------
// RRF scoring fusion
// ---------------------------------------------------------------------------

function rrfFuse(rankings: Map<string, number>[]): Map<string, number> {
  const k = 60;
  const fused = new Map<string, number>();
  for (const ranking of rankings) {
    const sorted = [...ranking.entries()].sort((a, b) => b[1] - a[1]);
    sorted.forEach(([id], rank) => {
      fused.set(id, (fused.get(id) || 0) + 1 / (k + rank + 1));
    });
  }
  return fused;
}

// ---------------------------------------------------------------------------
// Activation decay
// ---------------------------------------------------------------------------

function decayedActivation(activation: number, lastAccessed: string, nodeType: string): number {
  const days = (Date.now() - new Date(lastAccessed).getTime()) / 86400000;
  const rate = DECAY_RATES[nodeType] || 0.95;
  return Math.max(activation * Math.pow(rate, days), 0.01);
}

// ---------------------------------------------------------------------------
// Hub dampening
// ---------------------------------------------------------------------------

function hubDampen(nodeId: string, score: number): number {
  const result = db.exec(`SELECT COUNT(*) FROM edges WHERE source_node_id = '${nodeId}' OR target_node_id = '${nodeId}'`);
  const edgeCount = (result[0]?.values[0]?.[0] as number) || 0;
  if (edgeCount > 10) return score / Math.log2(edgeCount);
  return score;
}

// ---------------------------------------------------------------------------
// Extraction (inline via Anthropic Haiku)
// ---------------------------------------------------------------------------

const EXTRACTION_PROMPT = `You are extracting cognitive patterns from text. Extract ONLY patterns that reveal how the user thinks, decides, or evaluates.

For each pattern, output a JSON object:
- "text": The pattern in the user's voice
- "type": One of: heuristic, mental_model, preference, value, assumption, tension, question, project, idea
- "confidence": strong | tentative | uncertain
- "epistemic": assertion | hypothesis | speculation

Type selection, work through IN ORDER (do NOT default to "idea"):
1. Decision rule or "when X, do Y"? -> heuristic
2. Framework for thinking about a domain? -> mental_model
3. Conflict or tradeoff between held beliefs? -> tension
4. What the user prefers, likes, or chooses? -> preference
5. Core principle or what they care about? -> value
6. Something taken as given that could be wrong? -> assumption
7. Active question being investigated? -> question
8. Ongoing effort with a goal? -> project
9. ONLY if none above -> idea

Rules: 0-8 patterns max. Quality over quantity. Corrections are strongest signal. "idea" should be RARE.
Output a JSON array. [] if none found.

Text to analyze:
`;

async function extractPatterns(text: string): Promise<ExtractedPattern[]> {
  if (!ANTHROPIC_API_KEY) return [];
  try {
    const resp = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": ANTHROPIC_API_KEY,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-haiku-4-5-20251001",
        max_tokens: 1024,
        messages: [{ role: "user", content: EXTRACTION_PROMPT + text }],
      }),
    });
    if (!resp.ok) return [];
    const json = await resp.json() as { content: { type: string; text: string }[] };
    const content = json.content?.[0]?.text || "";
    const match = content.match(/\[[\s\S]*\]/);
    if (!match) return [];
    const parsed = JSON.parse(match[0]) as ExtractedPattern[];
    return parsed.filter(p =>
      p.text && p.type && VALID_NODE_TYPES.has(p.type) &&
      VALID_CONFIDENCE.has(p.confidence || "tentative") &&
      VALID_EPISTEMIC_TAGS.has(p.epistemic || "assertion")
    );
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getNode(nodeId: string): Record<string, unknown> | null {
  const rows = db.exec(`SELECT * FROM nodes WHERE id = '${nodeId}'`);
  if (rows.length === 0 || rows[0].values.length === 0) return null;
  const cols = rows[0].columns;
  const vals = rows[0].values[0];
  const obj: Record<string, unknown> = {};
  cols.forEach((c: string, i: number) => { obj[c] = vals[i]; });
  return obj;
}

function queryNodes(sql: string): Record<string, unknown>[] {
  const rows = db.exec(sql);
  if (rows.length === 0) return [];
  return rows[0].values.map((vals: any[]) => {
    const obj: Record<string, unknown> = {};
    rows[0].columns.forEach((c: string, i: number) => { obj[c] = vals[i]; });
    return obj;
  });
}

function bumpActivation(nodeId: string) {
  db.run(`UPDATE nodes SET activation = MIN(activation + 0.1, 1.0), last_accessed = '${now()}' WHERE id = '${nodeId}'`);
}

// ---------------------------------------------------------------------------
// Tool: capture
// ---------------------------------------------------------------------------

async function toolCapture(text: string, nodeType?: string, epistemicTag = "assertion", confidence = "tentative"): Promise<string> {
  if (nodeType && VALID_NODE_TYPES.has(nodeType)) {
    const chunkId = uuid(), nodeId = uuid();
    const vec = await embedOne(text);
    if (!vec) return "Error: failed to generate embedding";
    if (!validateDims(vec)) return "Error: embedding dimension mismatch";

    db.run("INSERT INTO chunks (id, text, source_type, epistemic_tag, confidence) VALUES (?, ?, 'capture', ?, ?)", [chunkId, text, epistemicTag, confidence]);
    db.run("INSERT INTO nodes (id, type, summary, confidence, activation) VALUES (?, ?, ?, ?, 1.0)", [nodeId, nodeType, text, confidence]);
    db.run("INSERT INTO node_chunks (node_id, chunk_id) VALUES (?, ?)", [nodeId, chunkId]);
    db.run("INSERT INTO embeddings (chunk_id, vector, model, dims) VALUES (?, ?, ?, ?)", [chunkId, JSON.stringify(vec), embeddingModel, embeddingDims]);
    persistDb();
    return `Captured as ${nodeType} (${epistemicTag}, ${confidence}). Node: ${nodeId}`;
  }

  // Extraction path
  const patterns = await extractPatterns(text);
  if (patterns.length === 0) {
    const chunkId = uuid(), nodeId = uuid();
    const vec = await embedOne(text);
    if (!vec) return "Error: failed to generate embedding";
    db.run("INSERT INTO chunks (id, text, source_type, epistemic_tag, confidence) VALUES (?, ?, 'capture', ?, ?)", [chunkId, text, epistemicTag, confidence]);
    db.run("INSERT INTO nodes (id, type, summary, confidence, activation) VALUES (?, ?, ?, ?, 1.0)", [nodeId, "idea", text, confidence]);
    db.run("INSERT INTO node_chunks (node_id, chunk_id) VALUES (?, ?)", [nodeId, chunkId]);
    db.run("INSERT INTO embeddings (chunk_id, vector, model, dims) VALUES (?, ?, ?, ?)", [chunkId, JSON.stringify(vec), embeddingModel, embeddingDims]);
    persistDb();
    return `No patterns extracted. Stored as idea. Node: ${nodeId}`;
  }

  const nodeIds: string[] = [];
  for (const p of patterns) {
    const chunkId = uuid(), nodeId = uuid();
    const vec = await embedOne(p.text);
    if (!vec) continue;
    db.run("INSERT INTO chunks (id, text, source_type, epistemic_tag, confidence) VALUES (?, ?, 'capture', ?, ?)", [chunkId, p.text, p.epistemic || "assertion", p.confidence || "tentative"]);
    db.run("INSERT INTO nodes (id, type, summary, confidence, activation) VALUES (?, ?, ?, ?, 1.0)", [nodeId, p.type, p.text, p.confidence || "tentative"]);
    db.run("INSERT INTO node_chunks (node_id, chunk_id) VALUES (?, ?)", [nodeId, chunkId]);
    db.run("INSERT INTO embeddings (chunk_id, vector, model, dims) VALUES (?, ?, ?, ?)", [chunkId, JSON.stringify(vec), embeddingModel, embeddingDims]);
    nodeIds.push(nodeId);
  }
  persistDb();
  const summary = patterns.map(p => `${p.type}: "${p.text.slice(0, 50)}"`).join("; ");
  return `Extracted ${patterns.length} pattern(s): ${summary}`;
}

// ---------------------------------------------------------------------------
// Tool: what_do_i_think
// ---------------------------------------------------------------------------

async function toolWhatDoIThink(topic: string): Promise<string> {
  const queryVec = await embedOne(topic);
  if (!queryVec) return "Error: failed to embed query";

  // Vector ranking
  const vectorHits = vectorSearch(queryVec, 40);
  const vectorScores = new Map<string, number>();
  for (const h of vectorHits) {
    const nc = db.exec(`SELECT node_id FROM node_chunks WHERE chunk_id = '${h.chunkId}'`);
    if (nc.length > 0) {
      for (const row of nc[0].values) {
        const nid = row[0] as string;
        vectorScores.set(nid, Math.max(vectorScores.get(nid) || 0, h.score));
      }
    }
  }

  // Keyword fallback
  const kw = topic.split(/\s+/).filter(w => w.length > 3).map(w => `summary LIKE '%${w}%'`).join(" OR ");
  const keywordScores = new Map<string, number>();
  if (kw) {
    const kwNodes = queryNodes(`SELECT id, activation FROM nodes WHERE ${kw} LIMIT 20`);
    kwNodes.forEach((n, i) => { keywordScores.set(n.id as string, 1 / (i + 1)); });
  }

  // Activation ranking
  const allIds = new Set([...vectorScores.keys(), ...keywordScores.keys()]);
  const activationScores = new Map<string, number>();
  for (const id of allIds) {
    const node = getNode(id);
    if (!node) continue;
    const act = decayedActivation(node.activation as number, node.last_accessed as string, node.type as string);
    activationScores.set(id, act);
  }

  // RRF fusion
  const fused = rrfFuse([vectorScores, keywordScores, activationScores]);
  const ranked = [...fused.entries()].sort((a, b) => b[1] - a[1]).slice(0, 20);

  // Bump activation on accessed nodes
  for (const [id] of ranked) bumpActivation(id);

  // Build response
  const positions = ranked.map(([id]) => {
    const node = getNode(id);
    if (!node) return null;
    return {
      type: node.type,
      summary: node.summary,
      confidence: node.confidence,
      activation: decayedActivation(node.activation as number, node.last_accessed as string, node.type as string).toFixed(3),
      first_seen: node.first_seen,
    };
  }).filter(Boolean);

  persistDb();
  return JSON.stringify({ topic, positions, node_count: positions.length }, null, 2);
}

// ---------------------------------------------------------------------------
// Tool: what_connects
// ---------------------------------------------------------------------------

async function toolWhatConnects(domainA: string, domainB: string): Promise<string> {
  const vecA = await embedOne(domainA);
  const vecB = await embedOne(domainB);
  if (!vecA || !vecB) return "Error: failed to embed domains";

  const processTypes = new Set(["heuristic", "value", "mental_model", "tension", "assumption", "preference"]);
  const allNodes = queryNodes("SELECT * FROM nodes");

  const bridges: { id: string; type: string; summary: string; scoreA: number; scoreB: number; min: number }[] = [];

  for (const node of allNodes) {
    if (!processTypes.has(node.type as string)) continue;
    const chunks = db.exec(`SELECT chunk_id FROM node_chunks WHERE node_id = '${node.id}'`);
    if (chunks.length === 0) continue;

    let bestScoreA = 0, bestScoreB = 0;
    for (const row of chunks[0].values) {
      const embRow = db.exec(`SELECT vector FROM embeddings WHERE chunk_id = '${row[0]}'`);
      if (embRow.length === 0) continue;
      const vec = JSON.parse(embRow[0].values[0][0] as string) as number[];
      bestScoreA = Math.max(bestScoreA, cosine(vecA, vec));
      bestScoreB = Math.max(bestScoreB, cosine(vecB, vec));
    }

    const minScore = Math.min(bestScoreA, bestScoreB);
    if (minScore > 0.18) {
      const dampened = hubDampen(node.id as string, minScore);
      bridges.push({
        id: node.id as string,
        type: node.type as string,
        summary: node.summary as string,
        scoreA: bestScoreA,
        scoreB: bestScoreB,
        min: dampened,
      });
    }
  }

  bridges.sort((a, b) => b.min - a.min);
  return JSON.stringify({ domain_a: domainA, domain_b: domainB, bridges: bridges.slice(0, 10) }, null, 2);
}

// ---------------------------------------------------------------------------
// Tool: what_tensions_exist
// ---------------------------------------------------------------------------

async function toolWhatTensionsExist(topic?: string): Promise<string> {
  let tensionNodes = queryNodes("SELECT * FROM nodes WHERE type = 'tension'");
  const contradictEdges = queryNodes("SELECT * FROM edges WHERE type = 'contradicts'");
  const weakAssumptions = queryNodes("SELECT * FROM nodes WHERE type = 'assumption' AND (confidence = 'uncertain' OR confidence = 'tentative')");

  if (topic) {
    const vec = await embedOne(topic);
    if (vec) {
      const hits = vectorSearch(vec, 40);
      const relevantChunks = new Set(hits.filter(h => h.score > 0.2).map(h => h.chunkId));
      const relevantNodes = new Set<string>();
      for (const cid of relevantChunks) {
        const nc = db.exec(`SELECT node_id FROM node_chunks WHERE chunk_id = '${cid}'`);
        if (nc.length > 0) nc[0].values.forEach((r: any[]) => relevantNodes.add(r[0] as string));
      }
      tensionNodes = tensionNodes.filter(n => relevantNodes.has(n.id as string));
    }
  }

  return JSON.stringify({
    explicit_tensions: tensionNodes.map(n => ({ type: n.type, summary: n.summary, confidence: n.confidence })),
    contradictions: contradictEdges.map(e => ({ source: e.source_node_id, target: e.target_node_id, weight: e.weight })),
    weak_assumptions: weakAssumptions.map(n => ({ summary: n.summary, confidence: n.confidence })),
  }, null, 2);
}

// ---------------------------------------------------------------------------
// Tool: where_am_i_uncertain
// ---------------------------------------------------------------------------

async function toolWhereAmIUncertain(domain?: string): Promise<string> {
  let nodes = queryNodes("SELECT * FROM nodes WHERE confidence IN ('uncertain', 'tentative') OR (type IN ('heuristic', 'mental_model', 'assumption') AND outcome_score IS NULL)");

  if (domain) {
    const vec = await embedOne(domain);
    if (vec) {
      const hits = vectorSearch(vec, 40);
      const relevantNodes = new Set<string>();
      for (const h of hits) {
        const nc = db.exec(`SELECT node_id FROM node_chunks WHERE chunk_id = '${h.chunkId}'`);
        if (nc.length > 0) nc[0].values.forEach((r: any[]) => relevantNodes.add(r[0] as string));
      }
      nodes = nodes.filter(n => relevantNodes.has(n.id as string));
    }
  }

  return JSON.stringify({
    uncertain: nodes.filter(n => n.confidence === "uncertain").map(n => ({ type: n.type, summary: n.summary })),
    tentative: nodes.filter(n => n.confidence === "tentative").map(n => ({ type: n.type, summary: n.summary })),
    untested: nodes.filter(n => n.outcome_score === null && ["heuristic", "mental_model", "assumption"].includes(n.type as string)).map(n => ({ type: n.type, summary: n.summary })),
  }, null, 2);
}

// ---------------------------------------------------------------------------
// Tool: suggest_exploration
// ---------------------------------------------------------------------------

async function toolSuggestExploration(currentContext: string): Promise<string> {
  const vec = await embedOne(currentContext);
  if (!vec) return "Error: failed to embed context";

  const hits = vectorSearch(vec, 40);
  const nodeScores = new Map<string, number>();
  for (const h of hits) {
    const nc = db.exec(`SELECT node_id FROM node_chunks WHERE chunk_id = '${h.chunkId}'`);
    if (nc.length > 0) {
      for (const row of nc[0].values) nodeScores.set(row[0] as string, Math.max(nodeScores.get(row[0] as string) || 0, h.score));
    }
  }

  const candidates: { id: string; type: string; summary: string; activation: number; similarity: number }[] = [];
  for (const [id, similarity] of nodeScores) {
    const node = getNode(id);
    if (!node) continue;
    const act = decayedActivation(node.activation as number, node.last_accessed as string, node.type as string);
    if (act < 0.5 && similarity > 0.15) {
      candidates.push({ id, type: node.type as string, summary: node.summary as string, activation: act, similarity });
    }
  }

  // Sort by similarity descending, prefer type diversity
  candidates.sort((a, b) => b.similarity - a.similarity);
  const typeSeen = new Set<string>();
  const diverse: typeof candidates = [];
  for (const c of candidates) {
    if (!typeSeen.has(c.type) || diverse.length < 5) {
      diverse.push(c);
      typeSeen.add(c.type);
    }
    if (diverse.length >= 10) break;
  }

  return JSON.stringify({ current_context: currentContext, forgotten_but_relevant: diverse }, null, 2);
}

// ---------------------------------------------------------------------------
// Tool: how_would_user_decide
// ---------------------------------------------------------------------------

async function toolHowWouldUserDecide(context: string, options?: string): Promise<string> {
  const vec = await embedOne(context + (options ? " " + options : ""));
  if (!vec) return "Error: failed to embed context";

  const hits = vectorSearch(vec, 40);
  const nodeIds = new Set<string>();
  for (const h of hits) {
    const nc = db.exec(`SELECT node_id FROM node_chunks WHERE chunk_id = '${h.chunkId}'`);
    if (nc.length > 0) nc[0].values.forEach((r: any[]) => nodeIds.add(r[0] as string));
  }

  const grouped: Record<string, { summary: string; confidence: string; activation: number }[]> = {};
  for (const id of nodeIds) {
    const node = getNode(id);
    if (!node) continue;
    const t = node.type as string;
    if (!["heuristic", "value", "mental_model", "preference", "assumption"].includes(t)) continue;
    if (!grouped[t]) grouped[t] = [];
    grouped[t].push({
      summary: node.summary as string,
      confidence: node.confidence as string,
      activation: decayedActivation(node.activation as number, node.last_accessed as string, t),
    });
  }

  for (const t in grouped) grouped[t].sort((a, b) => b.activation - a.activation);

  return JSON.stringify({ context, options: options || null, reasoning_inputs: grouped }, null, 2);
}

// ---------------------------------------------------------------------------
// Tool: what_has_changed
// ---------------------------------------------------------------------------

async function toolWhatHasChanged(domain: string): Promise<string> {
  const vec = await embedOne(domain);
  if (!vec) return "Error: failed to embed domain";

  const hits = vectorSearch(vec, 40);
  const nodeIds = new Set<string>();
  for (const h of hits) {
    const nc = db.exec(`SELECT node_id FROM node_chunks WHERE chunk_id = '${h.chunkId}'`);
    if (nc.length > 0) nc[0].values.forEach((r: any[]) => nodeIds.add(r[0] as string));
  }

  const nodes = [...nodeIds].map(id => getNode(id)).filter(Boolean).sort((a, b) =>
    new Date(a!.first_seen as string).getTime() - new Date(b!.first_seen as string).getTime()
  );

  const evolutions = queryNodes(`SELECT * FROM edges WHERE type = 'evolved_into' AND (source_node_id IN (${[...nodeIds].map(id => `'${id}'`).join(",") || "''"}) OR target_node_id IN (${[...nodeIds].map(id => `'${id}'`).join(",") || "''"}))`);

  const outcomes = queryNodes(`SELECT * FROM outcomes WHERE node_id IN (${[...nodeIds].map(id => `'${id}'`).join(",") || "''"}) ORDER BY created_at DESC`);

  return JSON.stringify({
    domain,
    timeline: nodes.map(n => ({ type: n!.type, summary: n!.summary, first_seen: n!.first_seen, confidence: n!.confidence })),
    evolution_chains: evolutions,
    outcomes: outcomes.slice(0, 10),
  }, null, 2);
}

// ---------------------------------------------------------------------------
// Tool: correct
// ---------------------------------------------------------------------------

async function toolCorrect(nodeId: string, newSummary: string, newConfidence?: string): Promise<string> {
  const existing = getNode(nodeId);
  if (!existing) return `Error: node ${nodeId} not found`;

  db.run(`UPDATE nodes SET summary = ?, activation = MIN(activation + 2.0, 10.0), last_accessed = ? ${newConfidence ? ", confidence = ?" : ""} WHERE id = ?`,
    newConfidence ? [newSummary, now(), newConfidence, nodeId] : [newSummary, now(), nodeId]);

  // Supersede old chunks
  const oldChunks = db.exec(`SELECT chunk_id FROM node_chunks WHERE node_id = '${nodeId}'`);
  const newChunkId = uuid();
  if (oldChunks.length > 0) {
    for (const row of oldChunks[0].values) {
      db.run(`UPDATE chunks SET superseded_by = ? WHERE id = ? AND superseded_by IS NULL`, [newChunkId, row[0]]);
    }
  }

  const vec = await embedOne(newSummary);
  if (vec) {
    db.run("INSERT INTO chunks (id, text, source_type, epistemic_tag, confidence) VALUES (?, ?, 'capture', 'assertion', ?)", [newChunkId, newSummary, newConfidence || "strong"]);
    db.run("INSERT INTO node_chunks (node_id, chunk_id) VALUES (?, ?)", [nodeId, newChunkId]);
    db.run("INSERT INTO embeddings (chunk_id, vector, model, dims) VALUES (?, ?, ?, ?)", [newChunkId, JSON.stringify(vec), embeddingModel, embeddingDims]);
  }

  persistDb();
  const oldSummary = (existing.summary as string).slice(0, 60);
  return `Corrected node ${nodeId}. Old: "${oldSummary}". New: "${newSummary.slice(0, 60)}".`;
}

// ---------------------------------------------------------------------------
// Tool: record_outcome
// ---------------------------------------------------------------------------

function toolRecordOutcome(nodeId: string, outcome: string, score: number): string {
  const existing = getNode(nodeId);
  if (!existing) return `Error: node ${nodeId} not found`;

  db.run("INSERT INTO outcomes (id, node_id, outcome, score) VALUES (?, ?, ?, ?)", [uuid(), nodeId, outcome, score]);

  const currentScore = (existing.outcome_score as number | null) ?? 0;
  const newScore = currentScore === 0 ? score : (currentScore * 0.7 + score * 0.3);
  db.run("UPDATE nodes SET outcome_score = ?, last_accessed = ? WHERE id = ?", [newScore, now(), nodeId]);

  persistDb();
  return `Recorded outcome for "${(existing.summary as string).slice(0, 50)}". Score: ${currentScore.toFixed(2)} -> ${newScore.toFixed(2)}`;
}

// ---------------------------------------------------------------------------
// Tool: get_node
// ---------------------------------------------------------------------------

function toolGetNode(nodeId: string): string {
  const node = getNode(nodeId);
  if (!node) return `Error: node ${nodeId} not found`;

  const chunks = queryNodes(`SELECT c.* FROM chunks c JOIN node_chunks nc ON c.id = nc.chunk_id WHERE nc.node_id = '${nodeId}'`);
  const edges = queryNodes(`SELECT * FROM edges WHERE source_node_id = '${nodeId}' OR target_node_id = '${nodeId}'`);
  const outs = queryNodes(`SELECT * FROM outcomes WHERE node_id = '${nodeId}' ORDER BY created_at DESC`);

  return JSON.stringify({ node, chunks, edges, outcomes: outs }, null, 2);
}

// ---------------------------------------------------------------------------
// Tool: get_neighbors
// ---------------------------------------------------------------------------

function toolGetNeighbors(nodeId: string): string {
  const node = getNode(nodeId);
  if (!node) return `Error: node ${nodeId} not found`;

  const edges = queryNodes(`SELECT * FROM edges WHERE source_node_id = '${nodeId}' OR target_node_id = '${nodeId}'`);
  const neighborIds = new Set<string>();
  for (const e of edges) {
    if (e.source_node_id !== nodeId) neighborIds.add(e.source_node_id as string);
    if (e.target_node_id !== nodeId) neighborIds.add(e.target_node_id as string);
  }

  const neighbors = [...neighborIds].map(id => getNode(id)).filter(Boolean).map(n => ({
    id: n!.id, type: n!.type, summary: n!.summary, activation: n!.activation,
  }));

  return JSON.stringify({ node: { id: node.id, type: node.type, summary: node.summary }, neighbors, edges }, null, 2);
}

// ---------------------------------------------------------------------------
// Tool: search_nodes
// ---------------------------------------------------------------------------

function toolSearchNodes(query: string, nodeType?: string): string {
  const words = query.split(/\s+/).filter(w => w.length > 2).map(w => `summary LIKE '%${w}%'`);
  if (words.length === 0) return JSON.stringify({ query, nodes: [] });

  let sql = `SELECT * FROM nodes WHERE (${words.join(" OR ")})`;
  if (nodeType && VALID_NODE_TYPES.has(nodeType)) sql += ` AND type = '${nodeType}'`;
  sql += " ORDER BY activation DESC LIMIT 20";

  const nodes = queryNodes(sql);
  return JSON.stringify({ query, type_filter: nodeType || null, count: nodes.length, nodes }, null, 2);
}

// ---------------------------------------------------------------------------
// Tool: merge_nodes
// ---------------------------------------------------------------------------

function toolMergeNodes(keepId: string, mergeId: string): string {
  const keep = getNode(keepId);
  const merge = getNode(mergeId);
  if (!keep) return `Error: node ${keepId} not found`;
  if (!merge) return `Error: node ${mergeId} not found`;

  // Transfer chunks
  db.run(`UPDATE node_chunks SET node_id = ? WHERE node_id = ? AND chunk_id NOT IN (SELECT chunk_id FROM node_chunks WHERE node_id = ?)`, [keepId, mergeId, keepId]);
  db.run(`DELETE FROM node_chunks WHERE node_id = ?`, [mergeId]);

  // Retarget edges
  db.run(`UPDATE edges SET source_node_id = ? WHERE source_node_id = ?`, [keepId, mergeId]);
  db.run(`UPDATE edges SET target_node_id = ? WHERE target_node_id = ?`, [keepId, mergeId]);
  db.run(`DELETE FROM edges WHERE source_node_id = target_node_id`);

  // Transfer outcomes
  db.run(`UPDATE outcomes SET node_id = ? WHERE node_id = ?`, [keepId, mergeId]);

  // Take higher activation
  const newAct = Math.max(keep.activation as number, merge.activation as number);
  db.run(`UPDATE nodes SET activation = ? WHERE id = ?`, [newAct, keepId]);

  // Delete merged node
  db.run(`DELETE FROM nodes WHERE id = ?`, [mergeId]);

  persistDb();
  return `Merged "${(merge.summary as string).slice(0, 50)}" into "${(keep.summary as string).slice(0, 50)}".`;
}

// ---------------------------------------------------------------------------
// Tool: archive_node
// ---------------------------------------------------------------------------

function toolArchiveNode(nodeId: string): string {
  const existing = getNode(nodeId);
  if (!existing) return `Error: node ${nodeId} not found`;

  db.run(`UPDATE nodes SET activation = 0.0, last_accessed = ? WHERE id = ?`, [now(), nodeId]);
  persistDb();
  return `Archived node ${nodeId} ("${(existing.summary as string).slice(0, 50)}"). Will not appear in results.`;
}

// ---------------------------------------------------------------------------
// Tool: bootstrap
// ---------------------------------------------------------------------------

const BOOTSTRAP_PHASES = [
  {
    name: "decisions",
    questions: [
      "Tell me about a decision you made in the last two weeks.",
      "What options did you reject?",
      "What tradeoff mattered most?",
      "What rule did you apply, even if you didn't name it at the time?",
    ],
  },
  {
    name: "heuristics",
    questions: [
      "When you're stuck on a problem, what's your first move?",
      "What decision rule do you apply that others might not?",
      "How do you decide what NOT to work on?",
    ],
  },
  {
    name: "mental_models",
    questions: [
      "What framework do you use repeatedly across different domains?",
      "How do you evaluate whether an idea is worth pursuing?",
    ],
  },
  {
    name: "tensions",
    questions: [
      "Where do two things you believe pull in opposite directions?",
      "What tradeoff do you keep revisiting without a clear answer?",
    ],
  },
  {
    name: "assumptions",
    questions: [
      "What do you assume is true that you haven't tested?",
      "What would change your mind about something you hold strongly?",
    ],
  },
];

function toolBootstrap(action?: string): string {
  const phaseRow = db.exec("SELECT value FROM meta WHERE key = 'bootstrap_phase'");
  const currentPhase = phaseRow.length > 0 && phaseRow[0].values.length > 0 ? parseInt(phaseRow[0].values[0][0] as string) : 0;

  const completeRow = db.exec("SELECT value FROM meta WHERE key = 'bootstrap_complete'");
  const isComplete = completeRow.length > 0 && completeRow[0].values[0]?.[0] === "true";

  if (action === "status" || isComplete) {
    const nodeCount = db.exec("SELECT COUNT(*) FROM nodes");
    const count = nodeCount[0]?.values[0]?.[0] || 0;
    return JSON.stringify({
      complete: isComplete,
      phase: currentPhase,
      total_phases: BOOTSTRAP_PHASES.length,
      node_count: count,
      message: isComplete
        ? `Bootstrap complete. ${count} nodes in the graph. Use capture to keep adding patterns from conversations.`
        : `Bootstrap in progress. Phase ${currentPhase + 1}/${BOOTSTRAP_PHASES.length}: ${BOOTSTRAP_PHASES[currentPhase]?.name || "done"}.`,
    }, null, 2);
  }

  if (currentPhase >= BOOTSTRAP_PHASES.length) {
    db.run("INSERT OR REPLACE INTO meta (key, value) VALUES ('bootstrap_complete', 'true')");
    persistDb();
    return JSON.stringify({ complete: true, message: "All phases done. Bootstrap complete." });
  }

  const phase = BOOTSTRAP_PHASES[currentPhase];
  return JSON.stringify({
    complete: false,
    phase: currentPhase + 1,
    phase_name: phase.name,
    total_phases: BOOTSTRAP_PHASES.length,
    questions: phase.questions,
    instructions: "Ask the user these questions one at a time. Feed each answer back through the capture tool (without nodeType, so extraction runs automatically). When done with all questions, call bootstrap again to advance to the next phase.",
    advance: `After capturing answers, call: bootstrap with no arguments to move to phase ${currentPhase + 2}.`,
  }, null, 2);
}

function advanceBootstrap(): string {
  const phaseRow = db.exec("SELECT value FROM meta WHERE key = 'bootstrap_phase'");
  const currentPhase = phaseRow.length > 0 && phaseRow[0].values.length > 0 ? parseInt(phaseRow[0].values[0][0] as string) : 0;
  const nextPhase = currentPhase + 1;

  if (nextPhase >= BOOTSTRAP_PHASES.length) {
    db.run("INSERT OR REPLACE INTO meta (key, value) VALUES ('bootstrap_phase', ?)", [String(nextPhase)]);
    db.run("INSERT OR REPLACE INTO meta (key, value) VALUES ('bootstrap_complete', 'true')");
    persistDb();
    return JSON.stringify({ complete: true, message: "Bootstrap complete. All phases done." });
  }

  db.run("INSERT OR REPLACE INTO meta (key, value) VALUES ('bootstrap_phase', ?)", [String(nextPhase)]);
  persistDb();
  return toolBootstrap();
}

// ---------------------------------------------------------------------------
// Tool: ping
// ---------------------------------------------------------------------------

function toolPing(): string {
  const nodeCount = db.exec("SELECT COUNT(*) FROM nodes");
  const chunkCount = db.exec("SELECT COUNT(*) FROM chunks");
  const n = nodeCount[0]?.values[0]?.[0] || 0;
  const c = chunkCount[0]?.values[0]?.[0] || 0;
  return `thinking-mcp online. ${n} nodes, ${c} chunks. DB: ${DB_PATH}`;
}

// ---------------------------------------------------------------------------
// Server setup
// ---------------------------------------------------------------------------

async function main() {
  await initDb();

  const server = new McpServer({ name: "thinking-mcp", version: "0.1.0" });

  // --- ping ---
  server.tool("ping", "Health check. Returns node count, chunk count, and DB path. READ-ONLY.", {}, () =>
    ({ content: [{ type: "text" as const, text: toolPing() }] }));

  // --- capture ---
  server.tool("capture",
    "Capture a thought into the cognitive graph. If nodeType is provided, stores as that type directly. If omitted, runs inline extraction to classify the text into properly typed nodes. WRITES to SQLite. Does NOT create edges between nodes.",
    {
      text: z.string().describe("The thought, observation, or raw text to capture"),
      nodeType: z.string().optional().describe("Optional. Skips extraction if provided. Valid: idea, question, heuristic, value, mental_model, assumption, tension, preference, project"),
      epistemicTag: z.string().default("assertion").describe("assertion, hypothesis, speculation, quoting, rejected"),
      confidence: z.string().default("tentative").describe("strong, tentative, uncertain"),
    },
    async ({ text, nodeType, epistemicTag, confidence }) =>
      ({ content: [{ type: "text" as const, text: await toolCapture(text, nodeType, epistemicTag, confidence) }] }));

  // --- what_do_i_think ---
  server.tool("what_do_i_think",
    "Query your thinking on a topic. Returns nodes ranked by relevance and activation. READ-ONLY (but bumps activation on accessed nodes). Use this before capture to check if a pattern already exists.",
    { topic: z.string().describe("The topic to explore your thinking on") },
    async ({ topic }) =>
      ({ content: [{ type: "text" as const, text: await toolWhatDoIThink(topic) }] }));

  // --- what_connects ---
  server.tool("what_connects",
    "Find unexpected bridges between two domains. Scores every process node against both domains and returns those relevant to both. READ-ONLY. Hub dampening applied to prevent common connectors from dominating.",
    { domain_a: z.string().describe("First domain"), domain_b: z.string().describe("Second domain") },
    async ({ domain_a, domain_b }) =>
      ({ content: [{ type: "text" as const, text: await toolWhatConnects(domain_a, domain_b) }] }));

  // --- what_tensions_exist ---
  server.tool("what_tensions_exist",
    "Surface contradictions and weak spots. Finds tension nodes, contradicts edges, and low-confidence assumptions. READ-ONLY. Optional topic filter.",
    { topic: z.string().optional().describe("Optional topic to focus on") },
    async ({ topic }) =>
      ({ content: [{ type: "text" as const, text: await toolWhatTensionsExist(topic) }] }));

  // --- where_am_i_uncertain ---
  server.tool("where_am_i_uncertain",
    "Find areas of low confidence or untested patterns. READ-ONLY. Optional domain filter.",
    { domain: z.string().optional().describe("Optional domain to focus on") },
    async ({ domain }) =>
      ({ content: [{ type: "text" as const, text: await toolWhereAmIUncertain(domain) }] }));

  // --- suggest_exploration ---
  server.tool("suggest_exploration",
    "Surface forgotten patterns relevant to your current context. Finds low-activation nodes semantically close to what you're working on. READ-ONLY. Use for creative cross-pollination.",
    { current_context: z.string().describe("What you're currently working on or thinking about") },
    async ({ current_context }) =>
      ({ content: [{ type: "text" as const, text: await toolSuggestExploration(current_context) }] }));

  // --- how_would_user_decide ---
  server.tool("how_would_user_decide",
    "Reason through a decision using captured heuristics, values, and mental models. Returns relevant patterns grouped by type. READ-ONLY. Does NOT make the decision.",
    {
      context: z.string().describe("The decision context"),
      options: z.string().optional().describe("Comma-separated options being considered"),
    },
    async ({ context, options }) =>
      ({ content: [{ type: "text" as const, text: await toolHowWouldUserDecide(context, options) }] }));

  // --- what_has_changed ---
  server.tool("what_has_changed",
    "Show how your thinking on a topic has evolved over time. Returns a timeline of nodes, evolution chains, and outcome history. READ-ONLY.",
    { domain: z.string().describe("The domain to check evolution on") },
    async ({ domain }) =>
      ({ content: [{ type: "text" as const, text: await toolWhatHasChanged(domain) }] }));

  // --- correct ---
  server.tool("correct",
    "Correct a node in the graph. Updates summary, supersedes old chunks, boosts activation +2.0. This is the strongest learning signal. WRITES to SQLite.",
    {
      node_id: z.string().describe("The node ID to correct"),
      new_summary: z.string().describe("The corrected summary"),
      new_confidence: z.string().optional().describe("Updated confidence: strong, tentative, uncertain"),
    },
    async ({ node_id, new_summary, new_confidence }) =>
      ({ content: [{ type: "text" as const, text: await toolCorrect(node_id, new_summary, new_confidence) }] }));

  // --- record_outcome ---
  server.tool("record_outcome",
    "Record what happened after a decision influenced by a node. Writes to outcomes table and updates the node's track record. WRITES to SQLite.",
    {
      node_id: z.string().describe("The node that influenced the decision"),
      outcome: z.string().describe("What happened"),
      score: z.number().describe("Outcome quality from -1.0 (terrible) to 1.0 (excellent)"),
    },
    ({ node_id, outcome, score }) =>
      ({ content: [{ type: "text" as const, text: toolRecordOutcome(node_id, outcome, score) }] }));

  // --- get_node ---
  server.tool("get_node",
    "Get a node by ID with all chunks, edges, and outcome history. READ-ONLY. Use before correcting, merging, or archiving.",
    { node_id: z.string().describe("The node ID") },
    ({ node_id }) =>
      ({ content: [{ type: "text" as const, text: toolGetNode(node_id) }] }));

  // --- get_neighbors ---
  server.tool("get_neighbors",
    "Get all nodes connected to a node via edges (1-hop). READ-ONLY. Returns the node, its neighbors, and connecting edges.",
    { node_id: z.string().describe("The node ID") },
    ({ node_id }) =>
      ({ content: [{ type: "text" as const, text: toolGetNeighbors(node_id) }] }));

  // --- search_nodes ---
  server.tool("search_nodes",
    "Keyword search across node summaries with optional type filter. Up to 20 results ranked by activation. READ-ONLY. For semantic search, use what_do_i_think instead.",
    {
      query: z.string().describe("Keyword to search for"),
      nodeType: z.string().optional().describe("Optional type filter"),
    },
    ({ query, nodeType }) =>
      ({ content: [{ type: "text" as const, text: toolSearchNodes(query, nodeType) }] }));

  // --- merge_nodes ---
  server.tool("merge_nodes",
    "Merge two duplicate nodes. Keeps the first, transfers chunks/edges/outcomes from the second, deletes the second. DESTRUCTIVE. Use get_node on both first.",
    {
      keep_id: z.string().describe("The node ID to keep"),
      merge_id: z.string().describe("The node ID to merge and delete"),
    },
    ({ keep_id, merge_id }) =>
      ({ content: [{ type: "text" as const, text: toolMergeNodes(keep_id, merge_id) }] }));

  // --- archive_node ---
  server.tool("archive_node",
    "Set a node's activation to 0. It stays in the graph but drops out of results. WRITES. Prefer correct if the node needs updating rather than removal.",
    { node_id: z.string().describe("The node ID to archive") },
    ({ node_id }) =>
      ({ content: [{ type: "text" as const, text: toolArchiveNode(node_id) }] }));

  // --- bootstrap ---
  server.tool("bootstrap",
    "Guided first-run experience. Returns questions for the agent to ask the user across 5 phases. Feed answers back through capture (without nodeType). Call again with no args to advance phases. READ-ONLY (but tracks phase state).",
    { action: z.string().optional().describe("'start' to begin or resume, 'status' to check progress, omit to advance to next phase") },
    ({ action }) => {
      const text = action === "start" || action === "status" ? toolBootstrap(action) : advanceBootstrap();
      return { content: [{ type: "text" as const, text }] };
    });

  // Connect
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(e => { console.error("Fatal:", e); process.exit(1); });
