import { EMBEDDING_PROVIDER, VOYAGE_API_KEY, OPENAI_API_KEY, OLLAMA_URL } from "./types.js";
import { db } from "./db.js";
import { DECAY_RATES } from "./types.js";
import { cosine } from "./cosine.js";

export let embeddingModel = "";
export let embeddingDims = 0;

export async function embed(texts: string[]): Promise<number[][]> {
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
    if (vectors.length > 0) { embeddingModel = "voyage-3-lite"; embeddingDims = vectors[0].length; }
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
    if (vectors.length > 0) { embeddingModel = "text-embedding-3-small"; embeddingDims = vectors[0].length; }
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
    if (vectors.length > 0) { embeddingModel = "nomic-embed-text"; embeddingDims = vectors[0].length; }
    return vectors;
  }

  throw new Error(`Unknown embedding provider: ${EMBEDDING_PROVIDER}`);
}

export async function embedOne(text: string): Promise<number[] | null> {
  try {
    const result = await embed([text]);
    return result[0] ?? null;
  } catch (e) {
    console.error("Embedding error:", e);
    return null;
  }
}

export function validateDims(vector: number[]): boolean {
  if (embeddingDims === 0) return true;
  return vector.length === embeddingDims;
}

// Re-export vendor-copied cosine for backward compatibility with existing imports
export { cosine } from "./cosine.js";

export function vectorSearch(queryVec: number[], topK = 20): { chunkId: string; score: number }[] {
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

export function rrfFuse(rankings: Map<string, number>[], weights?: number[]): Map<string, number> {
  const k = 60;
  const fused = new Map<string, number>();
  rankings.forEach((ranking, i) => {
    const w = weights?.[i] ?? 1;
    const sorted = [...ranking.entries()].sort((a, b) => b[1] - a[1]);
    sorted.forEach(([id], rank) => {
      fused.set(id, (fused.get(id) || 0) + w / (k + rank + 1));
    });
  });
  return fused;
}

export function decayedActivation(activation: number, lastAccessed: string, nodeType: string): number {
  const days = (Date.now() - new Date(lastAccessed).getTime()) / 86400000;
  const rate = DECAY_RATES[nodeType] || 0.95;
  return Math.max(activation * Math.pow(rate, days), 0.01);
}

export function hubDampen(nodeId: string, score: number): number {
  const result = db.exec(`SELECT COUNT(*) FROM edges WHERE source_node_id = '${nodeId}' OR target_node_id = '${nodeId}'`);
  const edgeCount = (result[0]?.values[0]?.[0] as number) || 0;
  if (edgeCount > 10) return score / Math.log2(edgeCount);
  return score;
}
