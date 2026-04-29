import { VALID_NODE_TYPES, VALID_EDGE_TYPES } from "./types.js";
import type { ExtractedRelation } from "./types.js";
import { db, persistDb, uuid, now, getNode, queryNodes, bumpActivation, queryOne, queryScalar } from "./db.js";
import { embedOne, validateDims, embeddingModel, embeddingDims, vectorSearch, cosine, rrfFuse, decayedActivation, hubDampen } from "./embed.js";
import { extractPatterns } from "./extract.js";
import { DB_PATH } from "./types.js";
import { pagerank } from "./pagerank.js";
import type { PREdge } from "./pagerank.js";

// ---------------------------------------------------------------------------
// Edge type inference heuristic
// ---------------------------------------------------------------------------

const EDGE_TYPE_MAP: Record<string, Record<string, string>> = {
  heuristic:  { heuristic: "supports", value: "derived_from", mental_model: "scoped_by" },
  tension:    { value: "contradicts", heuristic: "contradicts" },
  assumption: { mental_model: "depends_on" },
  preference: { value: "derived_from" },
  idea:       { project: "belongs_to" },
  question:   { assumption: "depends_on" },
};

function inferEdgeType(sourceType: string, targetType: string): string {
  return EDGE_TYPE_MAP[sourceType]?.[targetType]
    ?? (sourceType === targetType ? "supports" : "inspired_by");
}

// ---------------------------------------------------------------------------
// Auto-edge creation (Stage 1: LLM-suggested, Stage 2: similarity)
// ---------------------------------------------------------------------------

async function createEdgesFromExtraction(
  nodeId: string, _nodeType: string,
  relatesTo: ExtractedRelation[]
): Promise<number> {
  let created = 0;
  for (const rel of relatesTo) {
    if (!rel.concept?.trim() || rel.concept.trim().length < 2) continue;
    if (!VALID_EDGE_TYPES.has(rel.edge_type)) continue;
    const vec = await embedOne(rel.concept.trim());
    if (!vec) continue;
    const hits = vectorSearch(vec, 3);
    for (const hit of hits) {
      if (hit.score < 0.65) break;
      const targetNodeId = queryScalar(
        "SELECT node_id FROM node_chunks WHERE chunk_id = ?", [hit.chunkId]
      ) as string | null;
      if (!targetNodeId || targetNodeId === nodeId) continue;
      const existing = queryScalar(
        "SELECT COUNT(*) FROM edges WHERE type = ? AND source_node_id = ? AND target_node_id = ?",
        [rel.edge_type, nodeId, targetNodeId]
      );
      if ((existing as number) > 0) continue;
      db.run(
        "INSERT INTO edges (id, type, source_node_id, target_node_id, weight) VALUES (?, ?, ?, ?, ?)",
        ["ae1-" + uuid(), rel.edge_type, nodeId, targetNodeId, hit.score]
      );
      created++;
      break; // one edge per relates_to entry
    }
  }
  return created;
}

async function createEdgesBySimilarity(
  nodeId: string, nodeType: string, embedding: number[]
): Promise<number> {
  const MAX_AUTO_EDGES = 3;
  const hits = vectorSearch(embedding, 10);
  let created = 0;
  for (const hit of hits) {
    if (created >= MAX_AUTO_EDGES) break;
    const targetNodeId = queryScalar(
      "SELECT node_id FROM node_chunks WHERE chunk_id = ?", [hit.chunkId]
    ) as string | null;
    if (!targetNodeId || targetNodeId === nodeId) continue;
    const targetNode = getNode(targetNodeId);
    if (!targetNode) continue;
    const threshold = targetNode.type === nodeType ? 0.75 : 0.80;
    if (hit.score < threshold) continue;
    const edgeType = inferEdgeType(nodeType, targetNode.type as string);
    const existing = queryScalar(
      "SELECT COUNT(*) FROM edges WHERE source_node_id = ? AND target_node_id = ?",
      [nodeId, targetNodeId]
    );
    if ((existing as number) > 0) continue;
    db.run(
      "INSERT INTO edges (id, type, source_node_id, target_node_id, weight) VALUES (?, ?, ?, ?, ?)",
      ["ae2-" + uuid(), edgeType, nodeId, targetNodeId, hit.score]
    );
    created++;
  }
  return created;
}

/**
 * Create edges for a newly captured node. Wraps both stages in try/catch
 * so edge creation failures never block capture.
 */
async function autoCreateEdges(
  nodeId: string, nodeType: string, embedding: number[],
  relatesTo?: ExtractedRelation[]
): Promise<number> {
  let totalEdges = 0;
  try {
    // Stage 1: LLM-suggested edges from extraction
    if (relatesTo && relatesTo.length > 0) {
      totalEdges += await createEdgesFromExtraction(nodeId, nodeType, relatesTo);
    }
    // Stage 2: vector-similarity fallback
    totalEdges += await createEdgesBySimilarity(nodeId, nodeType, embedding);
    if (totalEdges > 0) invalidatePageRankCache();
  } catch (e) {
    console.error("Auto-edge creation error (non-fatal):", e);
  }
  return totalEdges;
}

// ---------------------------------------------------------------------------
// PageRank computation with caching
// ---------------------------------------------------------------------------

const PAGERANK_CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes
let pageRankCache: { ranks: Map<string, number>; timestamp: number } | null = null;

/**
 * Compute PageRank over the node graph. Caches results for 5 minutes.
 * Cache is invalidated when edges are created (auto-edge creation calls this
 * implicitly on next retrieval).
 */
export function computePageRank(): Map<string, number> {
  if (pageRankCache && Date.now() - pageRankCache.timestamp < PAGERANK_CACHE_TTL_MS) {
    return pageRankCache.ranks;
  }

  const ranks = new Map<string, number>();
  const edgeRows = db.exec("SELECT source_node_id, target_node_id, weight FROM edges");
  if (edgeRows.length === 0 || edgeRows[0].values.length === 0) {
    pageRankCache = { ranks, timestamp: Date.now() };
    return ranks;
  }

  // Build node index: string ID → integer
  const nodeIndex = new Map<string, number>();
  const nodeIds: string[] = [];
  for (const row of edgeRows[0].values) {
    const src = row[0] as string;
    const tgt = row[1] as string;
    if (!nodeIndex.has(src)) { nodeIndex.set(src, nodeIds.length); nodeIds.push(src); }
    if (!nodeIndex.has(tgt)) { nodeIndex.set(tgt, nodeIds.length); nodeIds.push(tgt); }
  }

  // Also include nodes with no edges (they get the uniform baseline)
  const allNodeRows = db.exec("SELECT id FROM nodes");
  if (allNodeRows.length > 0) {
    for (const row of allNodeRows[0].values) {
      const nid = row[0] as string;
      if (!nodeIndex.has(nid)) { nodeIndex.set(nid, nodeIds.length); nodeIds.push(nid); }
    }
  }

  const prEdges: PREdge[] = edgeRows[0].values.map((row: any[]) => ({
    source: nodeIndex.get(row[0] as string)!,
    target: nodeIndex.get(row[1] as string)!,
    weight: row[2] as number,
  }));

  const prResult = pagerank(nodeIds.length, prEdges);
  for (let i = 0; i < nodeIds.length; i++) {
    ranks.set(nodeIds[i], prResult[i]);
  }

  pageRankCache = { ranks, timestamp: Date.now() };
  return ranks;
}

/** Invalidate cached PageRank (call after edge mutations). */
export function invalidatePageRankCache(): void {
  pageRankCache = null;
}

// ---------------------------------------------------------------------------
// ping
// ---------------------------------------------------------------------------

export function toolPing(): string {
  const nodeCount = db.exec("SELECT COUNT(*) FROM nodes");
  const chunkCount = db.exec("SELECT COUNT(*) FROM chunks");
  const edgeCount = db.exec("SELECT COUNT(*) FROM edges");
  const n = nodeCount[0]?.values[0]?.[0] || 0;
  const c = chunkCount[0]?.values[0]?.[0] || 0;
  const e = edgeCount[0]?.values[0]?.[0] || 0;
  return `thinking-mcp online. ${n} nodes, ${c} chunks, ${e} edges. DB: ${DB_PATH}`;
}

// ---------------------------------------------------------------------------
// capture
// ---------------------------------------------------------------------------

export async function toolCapture(text: string, nodeType?: string, epistemicTag = "assertion", confidence = "tentative"): Promise<string> {
  if (nodeType && VALID_NODE_TYPES.has(nodeType)) {
    const chunkId = uuid(), nodeId = uuid();
    const vec = await embedOne(text);
    if (!vec) return "Error: failed to generate embedding";
    if (!validateDims(vec)) return "Error: embedding dimension mismatch";

    db.run("INSERT INTO chunks (id, text, source_type, epistemic_tag, confidence) VALUES (?, ?, 'capture', ?, ?)", [chunkId, text, epistemicTag, confidence]);
    db.run("INSERT INTO nodes (id, type, summary, confidence, activation) VALUES (?, ?, ?, ?, 1.0)", [nodeId, nodeType, text, confidence]);
    db.run("INSERT INTO node_chunks (node_id, chunk_id) VALUES (?, ?)", [nodeId, chunkId]);
    db.run("INSERT INTO embeddings (chunk_id, vector, model, dims) VALUES (?, ?, ?, ?)", [chunkId, JSON.stringify(vec), embeddingModel, embeddingDims]);

    const edgeCount = await autoCreateEdges(nodeId, nodeType, vec);
    persistDb();
    const edgeNote = edgeCount > 0 ? ` ${edgeCount} edge(s) created.` : "";
    return `Captured as ${nodeType} (${epistemicTag}, ${confidence}). Node: ${nodeId}.${edgeNote}`;
  }

  const patterns = await extractPatterns(text);
  if (patterns.length === 0) {
    const chunkId = uuid(), nodeId = uuid();
    const vec = await embedOne(text);
    if (!vec) return "Error: failed to generate embedding";
    db.run("INSERT INTO chunks (id, text, source_type, epistemic_tag, confidence) VALUES (?, ?, 'capture', ?, ?)", [chunkId, text, epistemicTag, confidence]);
    db.run("INSERT INTO nodes (id, type, summary, confidence, activation) VALUES (?, ?, ?, ?, 1.0)", [nodeId, "idea", text, confidence]);
    db.run("INSERT INTO node_chunks (node_id, chunk_id) VALUES (?, ?)", [nodeId, chunkId]);
    db.run("INSERT INTO embeddings (chunk_id, vector, model, dims) VALUES (?, ?, ?, ?)", [chunkId, JSON.stringify(vec), embeddingModel, embeddingDims]);

    const edgeCount = await autoCreateEdges(nodeId, "idea", vec);
    persistDb();
    const edgeNote = edgeCount > 0 ? ` ${edgeCount} edge(s) created.` : "";
    return `No patterns extracted. Stored as idea. Node: ${nodeId}.${edgeNote}`;
  }

  // Two-pass flow: insert all nodes first, then create edges.
  // This ensures intra-capture relates_to references can resolve
  // (pattern A's relates_to might reference pattern B from the same capture).
  const inserted: { nodeId: string; type: string; vec: number[]; relatesTo?: typeof patterns[0]["relates_to"] }[] = [];
  for (const p of patterns) {
    const chunkId = uuid(), nodeId = uuid();
    const vec = await embedOne(p.text);
    if (!vec) continue;
    db.run("INSERT INTO chunks (id, text, source_type, epistemic_tag, confidence) VALUES (?, ?, 'capture', ?, ?)", [chunkId, p.text, p.epistemic || "assertion", p.confidence || "tentative"]);
    db.run("INSERT INTO nodes (id, type, summary, confidence, activation) VALUES (?, ?, ?, ?, 1.0)", [nodeId, p.type, p.text, p.confidence || "tentative"]);
    db.run("INSERT INTO node_chunks (node_id, chunk_id) VALUES (?, ?)", [nodeId, chunkId]);
    db.run("INSERT INTO embeddings (chunk_id, vector, model, dims) VALUES (?, ?, ?, ?)", [chunkId, JSON.stringify(vec), embeddingModel, embeddingDims]);
    inserted.push({ nodeId, type: p.type, vec, relatesTo: p.relates_to });
  }

  // Pass 2: create edges now that all nodes + embeddings exist
  let totalEdges = 0;
  for (const { nodeId, type, vec, relatesTo } of inserted) {
    totalEdges += await autoCreateEdges(nodeId, type, vec, relatesTo);
  }
  persistDb();
  const summary = patterns.map(p => `${p.type}: "${p.text.slice(0, 50)}"`).join("; ");
  const edgeNote = totalEdges > 0 ? ` ${totalEdges} edge(s) created.` : "";
  return `Extracted ${patterns.length} pattern(s): ${summary}.${edgeNote}`;
}

// ---------------------------------------------------------------------------
// what_do_i_think
// ---------------------------------------------------------------------------

export async function toolWhatDoIThink(topic: string): Promise<string> {
  const queryVec = await embedOne(topic);
  if (!queryVec) return "Error: failed to embed query";

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

  const words = topic.split(/\s+/).filter(w => w.length > 3).map(w => w.replace(/[%_'\\]/g, ""));
  const keywordScores = new Map<string, number>();
  if (words.length > 0) {
    const conditions = words.map(() => "summary LIKE ?").join(" OR ");
    const params = words.map(w => `%${w}%`);
    const stmt = db.prepare(`SELECT id, activation FROM nodes WHERE (${conditions}) LIMIT 20`);
    try {
      stmt.bind(params);
      let idx = 0;
      while (stmt.step()) {
        const row = stmt.getAsObject();
        keywordScores.set(row.id as string, 1 / (idx + 1));
        idx++;
      }
    } finally {
      stmt.free();
    }
  }

  const allIds = new Set([...vectorScores.keys(), ...keywordScores.keys()]);
  const activationScores = new Map<string, number>();
  const outcomeScores = new Map<string, number>();
  for (const id of allIds) {
    const node = getNode(id);
    if (!node) continue;
    activationScores.set(id, decayedActivation(node.activation as number, node.last_accessed as string, node.type as string));
    outcomeScores.set(id, (node.outcome_score as number | null) ?? 0);
  }

  // PageRank: structural importance signal (5th channel)
  // Filter to allIds only — prevents globally important but topic-irrelevant nodes
  // from entering the fused ranking
  const fullPageRank = computePageRank();
  const pageRankScores = new Map<string, number>();
  for (const id of allIds) {
    const pr = fullPageRank.get(id);
    if (pr !== undefined) pageRankScores.set(id, pr);
  }

  // Weights: vector 3x, keyword 3x, activation 1x, outcome 0.5x, pagerank 2x
  const fused = rrfFuse([vectorScores, keywordScores, activationScores, outcomeScores, pageRankScores], [3, 3, 1, 0.5, 2]);
  const ranked = [...fused.entries()].sort((a, b) => b[1] - a[1]).slice(0, 20);

  for (const [id] of ranked) bumpActivation(id);

  const positions = ranked.map(([id]) => {
    const node = getNode(id);
    if (!node) return null;
    const pr = pageRankScores.get(id);
    return {
      type: node.type, summary: node.summary, confidence: node.confidence,
      activation: decayedActivation(node.activation as number, node.last_accessed as string, node.type as string).toFixed(3),
      outcome_score: node.outcome_score,
      pagerank: pr !== undefined ? pr.toFixed(6) : null,
      first_seen: node.first_seen,
    };
  }).filter(Boolean);

  persistDb();
  return JSON.stringify({ topic, positions, node_count: positions.length }, null, 2);
}

// ---------------------------------------------------------------------------
// what_connects
// ---------------------------------------------------------------------------

export async function toolWhatConnects(domainA: string, domainB: string): Promise<string> {
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
      bridges.push({ id: node.id as string, type: node.type as string, summary: node.summary as string, scoreA: bestScoreA, scoreB: bestScoreB, min: hubDampen(node.id as string, minScore) });
    }
  }

  bridges.sort((a, b) => b.min - a.min);
  return JSON.stringify({ domain_a: domainA, domain_b: domainB, bridges: bridges.slice(0, 10) }, null, 2);
}

// ---------------------------------------------------------------------------
// what_tensions_exist
// ---------------------------------------------------------------------------

export async function toolWhatTensionsExist(topic?: string): Promise<string> {
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
// where_am_i_uncertain
// ---------------------------------------------------------------------------

export async function toolWhereAmIUncertain(domain?: string): Promise<string> {
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
// suggest_exploration
// ---------------------------------------------------------------------------

export async function toolSuggestExploration(currentContext: string): Promise<string> {
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

  const prScores = computePageRank();
  const candidates: { id: string; type: string; summary: string; activation: number; similarity: number; pagerank: number }[] = [];
  for (const [id, similarity] of nodeScores) {
    const node = getNode(id);
    if (!node) continue;
    const act = decayedActivation(node.activation as number, node.last_accessed as string, node.type as string);
    if (act < 0.5 && similarity > 0.15) {
      const pr = prScores.get(id) ?? 0;
      candidates.push({ id, type: node.type as string, summary: node.summary as string, activation: act, similarity, pagerank: pr });
    }
  }

  // Sort by similarity boosted by PageRank (structurally important forgotten nodes rank higher)
  candidates.sort((a, b) => (b.similarity + b.pagerank * 10) - (a.similarity + a.pagerank * 10));
  const typeSeen = new Set<string>();
  const diverse: typeof candidates = [];
  for (const c of candidates) {
    if (!typeSeen.has(c.type) || diverse.length < 5) { diverse.push(c); typeSeen.add(c.type); }
    if (diverse.length >= 10) break;
  }

  return JSON.stringify({ current_context: currentContext, forgotten_but_relevant: diverse }, null, 2);
}

// ---------------------------------------------------------------------------
// how_would_user_decide
// ---------------------------------------------------------------------------

export async function toolHowWouldUserDecide(context: string, options?: string): Promise<string> {
  const vec = await embedOne(context + (options ? " " + options : ""));
  if (!vec) return "Error: failed to embed context";

  const hits = vectorSearch(vec, 40);
  const nodeIds = new Set<string>();
  for (const h of hits) {
    const nc = db.exec(`SELECT node_id FROM node_chunks WHERE chunk_id = '${h.chunkId}'`);
    if (nc.length > 0) nc[0].values.forEach((r: any[]) => nodeIds.add(r[0] as string));
  }

  const prScores = computePageRank();
  const grouped: Record<string, { summary: string; confidence: string; activation: number; pagerank: number }[]> = {};
  for (const id of nodeIds) {
    const node = getNode(id);
    if (!node) continue;
    const t = node.type as string;
    if (!["heuristic", "value", "mental_model", "preference", "assumption"].includes(t)) continue;
    if (!grouped[t]) grouped[t] = [];
    const pr = prScores.get(id) ?? 0;
    grouped[t].push({ summary: node.summary as string, confidence: node.confidence as string, activation: decayedActivation(node.activation as number, node.last_accessed as string, t), pagerank: pr });
  }
  // Sort by activation boosted by PageRank
  for (const t in grouped) grouped[t].sort((a, b) => (b.activation + b.pagerank * 5) - (a.activation + a.pagerank * 5));

  return JSON.stringify({ context, options: options || null, reasoning_inputs: grouped }, null, 2);
}

// ---------------------------------------------------------------------------
// what_has_changed
// ---------------------------------------------------------------------------

export async function toolWhatHasChanged(domain: string): Promise<string> {
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

  const idList = [...nodeIds].map(id => `'${id}'`).join(",") || "''";
  const evolutions = queryNodes(`SELECT * FROM edges WHERE type = 'evolved_into' AND (source_node_id IN (${idList}) OR target_node_id IN (${idList}))`);
  const outcomes = queryNodes(`SELECT * FROM outcomes WHERE node_id IN (${idList}) ORDER BY created_at DESC`);

  return JSON.stringify({
    domain,
    timeline: nodes.map(n => ({ type: n!.type, summary: n!.summary, first_seen: n!.first_seen, confidence: n!.confidence })),
    evolution_chains: evolutions,
    outcomes: outcomes.slice(0, 10),
  }, null, 2);
}

// ---------------------------------------------------------------------------
// correct
// ---------------------------------------------------------------------------

export async function toolCorrect(nodeId: string, newSummary: string, newConfidence?: string): Promise<string> {
  const existing = getNode(nodeId);
  if (!existing) return `Error: node ${nodeId} not found`;

  db.run(`UPDATE nodes SET summary = ?, activation = MIN(activation + 2.0, 10.0), last_accessed = ? ${newConfidence ? ", confidence = ?" : ""} WHERE id = ?`,
    newConfidence ? [newSummary, now(), newConfidence, nodeId] : [newSummary, now(), nodeId]);

  const oldChunks = db.exec(`SELECT chunk_id FROM node_chunks WHERE node_id = '${nodeId}'`);
  const newChunkId = uuid();

  // Insert new chunk FIRST so FK on superseded_by can reference it
  const vec = await embedOne(newSummary);
  if (vec) {
    db.run("INSERT INTO chunks (id, text, source_type, epistemic_tag, confidence) VALUES (?, ?, 'capture', 'assertion', ?)", [newChunkId, newSummary, newConfidence || "strong"]);
    db.run("INSERT INTO node_chunks (node_id, chunk_id) VALUES (?, ?)", [nodeId, newChunkId]);
    db.run("INSERT INTO embeddings (chunk_id, vector, model, dims) VALUES (?, ?, ?, ?)", [newChunkId, JSON.stringify(vec), embeddingModel, embeddingDims]);

    // Now supersede old chunks — new chunk exists so FK is satisfied
    if (oldChunks.length > 0) {
      for (const row of oldChunks[0].values) {
        db.run(`UPDATE chunks SET superseded_by = ? WHERE id = ? AND superseded_by IS NULL`, [newChunkId, row[0]]);
      }
    }
  }

  persistDb();
  return `Corrected node ${nodeId}. Old: "${(existing.summary as string).slice(0, 60)}". New: "${newSummary.slice(0, 60)}".`;
}

// ---------------------------------------------------------------------------
// record_outcome
// ---------------------------------------------------------------------------

export function toolRecordOutcome(nodeId: string, outcome: string, score: number): string {
  const existing = getNode(nodeId);
  if (!existing) return `Error: node ${nodeId} not found`;

  db.run("INSERT INTO outcomes (id, node_id, outcome, score) VALUES (?, ?, ?, ?)", [uuid(), nodeId, outcome, score]);
  const currentScore = (existing.outcome_score as number | null) ?? 0;
  const newScore = currentScore === 0 ? score : (currentScore * 0.7 + score * 0.3);
  db.run("UPDATE nodes SET outcome_score = ?, last_accessed = ? WHERE id = ?", [newScore, now(), nodeId]);

  // Propagate to neighbors via edge-type-aware dampening (1-hop only)
  const PROP_WEIGHTS: Record<string, number> = { supports: 0.3, contradicts: -0.2, depends_on: 0.15 };
  const edgeRows = queryNodes(
    `SELECT id, type, source_node_id, target_node_id FROM edges WHERE source_node_id = '${nodeId}' OR target_node_id = '${nodeId}'`
  );
  let propagated = 0;
  for (const edge of edgeRows) {
    const w = PROP_WEIGHTS[edge.type as string];
    if (w === undefined) continue;
    const neighborId = (edge.source_node_id as string) === nodeId ? edge.target_node_id : edge.source_node_id;
    db.run(
      `UPDATE nodes SET outcome_score = COALESCE(outcome_score, 0) * 0.7 + ? * 0.3, last_accessed = ? WHERE id = ?`,
      [score * w, now(), neighborId]
    );
    propagated++;
  }

  // Auto-update confidence after 3+ outcomes
  const countRows = db.exec(`SELECT COUNT(*) FROM outcomes WHERE node_id = '${nodeId}'`);
  const outcomeCount = countRows.length > 0 ? (countRows[0].values[0][0] as number) : 0;
  if (outcomeCount >= 3) {
    if (newScore > 0.5) db.run(`UPDATE nodes SET confidence = 'strong' WHERE id = ? AND confidence != 'strong'`, [nodeId]);
    else if (newScore < -0.3) db.run(`UPDATE nodes SET confidence = 'uncertain' WHERE id = ? AND confidence != 'uncertain'`, [nodeId]);
  }

  persistDb();
  const propNote = propagated > 0 ? ` Propagated to ${propagated} neighbor(s).` : "";
  return `Recorded outcome for "${(existing.summary as string).slice(0, 50)}". Score: ${currentScore.toFixed(2)} -> ${newScore.toFixed(2)}.${propNote}`;
}

// ---------------------------------------------------------------------------
// get_node
// ---------------------------------------------------------------------------

export function toolGetNode(nodeId: string): string {
  const node = getNode(nodeId);
  if (!node) return `Error: node ${nodeId} not found`;
  const chunks = queryNodes(`SELECT c.* FROM chunks c JOIN node_chunks nc ON c.id = nc.chunk_id WHERE nc.node_id = '${nodeId}'`);
  const edges = queryNodes(`SELECT * FROM edges WHERE source_node_id = '${nodeId}' OR target_node_id = '${nodeId}'`);
  const outs = queryNodes(`SELECT * FROM outcomes WHERE node_id = '${nodeId}' ORDER BY created_at DESC`);
  return JSON.stringify({ node, chunks, edges, outcomes: outs }, null, 2);
}

// ---------------------------------------------------------------------------
// get_neighbors
// ---------------------------------------------------------------------------

export function toolGetNeighbors(nodeId: string): string {
  const node = getNode(nodeId);
  if (!node) return `Error: node ${nodeId} not found`;
  const edges = queryNodes(`SELECT * FROM edges WHERE source_node_id = '${nodeId}' OR target_node_id = '${nodeId}'`);
  const neighborIds = new Set<string>();
  for (const e of edges) {
    if (e.source_node_id !== nodeId) neighborIds.add(e.source_node_id as string);
    if (e.target_node_id !== nodeId) neighborIds.add(e.target_node_id as string);
  }
  const neighbors = [...neighborIds].map(id => getNode(id)).filter(Boolean).map(n => ({ id: n!.id, type: n!.type, summary: n!.summary, activation: n!.activation }));
  return JSON.stringify({ node: { id: node.id, type: node.type, summary: node.summary }, neighbors, edges }, null, 2);
}

// ---------------------------------------------------------------------------
// search_nodes
// ---------------------------------------------------------------------------

export function toolSearchNodes(query: string, nodeType?: string): string {
  const words = query.split(/\s+/).filter(w => w.length > 2).map(w => w.replace(/[%_'\\]/g, ""));
  if (words.length === 0) return JSON.stringify({ query, nodes: [] });
  const conditions = words.map(() => "summary LIKE ?").join(" OR ");
  const params: any[] = words.map(w => `%${w}%`);
  let sql = `SELECT * FROM nodes WHERE (${conditions})`;
  if (nodeType && VALID_NODE_TYPES.has(nodeType)) {
    sql += " AND type = ?";
    params.push(nodeType);
  }
  sql += " ORDER BY activation DESC LIMIT 20";
  const stmt = db.prepare(sql);
  const nodes: Record<string, unknown>[] = [];
  try {
    stmt.bind(params);
    while (stmt.step()) {
      nodes.push(stmt.getAsObject() as Record<string, unknown>);
    }
  } finally {
    stmt.free();
  }
  return JSON.stringify({ query, type_filter: nodeType || null, count: nodes.length, nodes }, null, 2);
}

// ---------------------------------------------------------------------------
// merge_nodes
// ---------------------------------------------------------------------------

export function toolMergeNodes(keepId: string, mergeId: string): string {
  const keep = getNode(keepId);
  const merge = getNode(mergeId);
  if (!keep) return `Error: node ${keepId} not found`;
  if (!merge) return `Error: node ${mergeId} not found`;

  db.run(`UPDATE node_chunks SET node_id = ? WHERE node_id = ? AND chunk_id NOT IN (SELECT chunk_id FROM node_chunks WHERE node_id = ?)`, [keepId, mergeId, keepId]);
  db.run(`DELETE FROM node_chunks WHERE node_id = ?`, [mergeId]);
  db.run(`UPDATE edges SET source_node_id = ? WHERE source_node_id = ?`, [keepId, mergeId]);
  db.run(`UPDATE edges SET target_node_id = ? WHERE target_node_id = ?`, [keepId, mergeId]);
  db.run(`DELETE FROM edges WHERE source_node_id = target_node_id`);
  db.run(`UPDATE outcomes SET node_id = ? WHERE node_id = ?`, [keepId, mergeId]);
  db.run(`UPDATE nodes SET activation = ? WHERE id = ?`, [Math.max(keep.activation as number, merge.activation as number), keepId]);
  db.run(`DELETE FROM nodes WHERE id = ?`, [mergeId]);

  invalidatePageRankCache(); // edges were rewired/deleted
  persistDb();
  return `Merged "${(merge.summary as string).slice(0, 50)}" into "${(keep.summary as string).slice(0, 50)}".`;
}

// ---------------------------------------------------------------------------
// archive_node
// ---------------------------------------------------------------------------

export function toolArchiveNode(nodeId: string): string {
  const existing = getNode(nodeId);
  if (!existing) return `Error: node ${nodeId} not found`;
  db.run(`UPDATE nodes SET activation = 0.0, last_accessed = ? WHERE id = ?`, [now(), nodeId]);
  persistDb();
  return `Archived node ${nodeId} ("${(existing.summary as string).slice(0, 50)}"). Will not appear in results.`;
}
