import * as path from "path";
import * as os from "os";

export type NodeType = "idea" | "question" | "project" | "heuristic" | "value" | "mental_model" | "assumption" | "tension" | "preference";
export type EpistemicTag = "assertion" | "hypothesis" | "speculation" | "quoting" | "rejected";
export type Confidence = "strong" | "tentative" | "uncertain";
export type EdgeType = "supports" | "contradicts" | "evolved_into" | "inspired_by" | "depends_on" | "overrides" | "learned_from" | "scoped_by" | "rejected" | "belongs_to" | "derived_from";

export const VALID_NODE_TYPES = new Set<string>(["idea", "question", "project", "heuristic", "value", "mental_model", "assumption", "tension", "preference"]);
export const VALID_EPISTEMIC_TAGS = new Set<string>(["assertion", "hypothesis", "speculation", "quoting", "rejected"]);
export const VALID_CONFIDENCE = new Set<string>(["strong", "tentative", "uncertain"]);
export const VALID_EDGE_TYPES = new Set<string>(["supports", "contradicts", "evolved_into", "inspired_by", "depends_on", "overrides", "learned_from", "scoped_by", "rejected", "belongs_to", "derived_from"]);

export const DECAY_RATES: Record<string, number> = {
  value: 0.98, assumption: 0.98,
  heuristic: 0.96, mental_model: 0.96, preference: 0.96, tension: 0.96,
  idea: 0.93, question: 0.93, project: 0.93,
};

export interface ExtractedRelation {
  concept: string;
  edge_type: string;
}

export interface ExtractedPattern {
  text: string;
  type: string;
  confidence: string;
  epistemic: string;
  relates_to?: ExtractedRelation[];
}

export const DB_PATH = process.env.THINKING_MCP_DB_PATH
  || path.join(os.homedir(), ".thinking-mcp", "mind.db");
export const EMBEDDING_PROVIDER = process.env.THINKING_MCP_EMBEDDING_PROVIDER || "voyage";
export const VOYAGE_API_KEY = process.env.VOYAGE_API_KEY;
export const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
export const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
export const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
