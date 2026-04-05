#!/usr/bin/env node
// thinking-mcp: MCP server that models how you think
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { initDb } from "./db.js";
import { toolPing, toolCapture, toolWhatDoIThink, toolWhatConnects, toolWhatTensionsExist, toolWhereAmIUncertain, toolSuggestExploration, toolHowWouldUserDecide, toolWhatHasChanged, toolCorrect, toolRecordOutcome, toolGetNode, toolGetNeighbors, toolSearchNodes, toolMergeNodes, toolArchiveNode } from "./tools.js";
import { toolBootstrap, advanceBootstrap } from "./bootstrap.js";

async function main() {
  await initDb();

  const server = new McpServer({ name: "thinking-mcp", version: "0.1.0" });

  server.tool("ping",
    "Health check. Returns node count, chunk count, and DB path. READ-ONLY.",
    {}, () => ({ content: [{ type: "text" as const, text: toolPing() }] }));

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

  server.tool("what_do_i_think",
    "Query your thinking on a topic. Returns nodes ranked by relevance and activation. READ-ONLY (but bumps activation on accessed nodes). Use this before capture to check if a pattern already exists.",
    { topic: z.string().describe("The topic to explore your thinking on") },
    async ({ topic }) => ({ content: [{ type: "text" as const, text: await toolWhatDoIThink(topic) }] }));

  server.tool("what_connects",
    "Find unexpected bridges between two domains. Scores every process node against both domains and returns those relevant to both. READ-ONLY. Hub dampening applied.",
    { domain_a: z.string().describe("First domain"), domain_b: z.string().describe("Second domain") },
    async ({ domain_a, domain_b }) => ({ content: [{ type: "text" as const, text: await toolWhatConnects(domain_a, domain_b) }] }));

  server.tool("what_tensions_exist",
    "Surface contradictions and weak spots. Finds tension nodes, contradicts edges, and low-confidence assumptions. READ-ONLY. Optional topic filter.",
    { topic: z.string().optional().describe("Optional topic to focus on") },
    async ({ topic }) => ({ content: [{ type: "text" as const, text: await toolWhatTensionsExist(topic) }] }));

  server.tool("where_am_i_uncertain",
    "Find areas of low confidence or untested patterns. READ-ONLY. Optional domain filter.",
    { domain: z.string().optional().describe("Optional domain to focus on") },
    async ({ domain }) => ({ content: [{ type: "text" as const, text: await toolWhereAmIUncertain(domain) }] }));

  server.tool("suggest_exploration",
    "Surface forgotten patterns relevant to your current context. Finds low-activation nodes semantically close to what you're working on. READ-ONLY. Use for creative cross-pollination.",
    { current_context: z.string().describe("What you're currently working on or thinking about") },
    async ({ current_context }) => ({ content: [{ type: "text" as const, text: await toolSuggestExploration(current_context) }] }));

  server.tool("how_would_user_decide",
    "Reason through a decision using captured heuristics, values, and mental models. Returns relevant patterns grouped by type. READ-ONLY. Does NOT make the decision.",
    { context: z.string().describe("The decision context"), options: z.string().optional().describe("Comma-separated options being considered") },
    async ({ context, options }) => ({ content: [{ type: "text" as const, text: await toolHowWouldUserDecide(context, options) }] }));

  server.tool("what_has_changed",
    "Show how your thinking on a topic has evolved over time. Returns a timeline of nodes, evolution chains, and outcome history. READ-ONLY.",
    { domain: z.string().describe("The domain to check evolution on") },
    async ({ domain }) => ({ content: [{ type: "text" as const, text: await toolWhatHasChanged(domain) }] }));

  server.tool("correct",
    "Correct a node in the graph. Updates summary, supersedes old chunks, boosts activation +2.0. Strongest learning signal. WRITES to SQLite.",
    { node_id: z.string().describe("The node ID to correct"), new_summary: z.string().describe("The corrected summary"), new_confidence: z.string().optional().describe("Updated confidence: strong, tentative, uncertain") },
    async ({ node_id, new_summary, new_confidence }) => ({ content: [{ type: "text" as const, text: await toolCorrect(node_id, new_summary, new_confidence) }] }));

  server.tool("record_outcome",
    "Record what happened after a decision influenced by a node. Writes to outcomes table and updates the node's track record. WRITES to SQLite.",
    { node_id: z.string().describe("The node that influenced the decision"), outcome: z.string().describe("What happened"), score: z.number().describe("Outcome quality from -1.0 (terrible) to 1.0 (excellent)") },
    ({ node_id, outcome, score }) => ({ content: [{ type: "text" as const, text: toolRecordOutcome(node_id, outcome, score) }] }));

  server.tool("get_node",
    "Get a node by ID with all chunks, edges, and outcome history. READ-ONLY. Use before correcting, merging, or archiving.",
    { node_id: z.string().describe("The node ID") },
    ({ node_id }) => ({ content: [{ type: "text" as const, text: toolGetNode(node_id) }] }));

  server.tool("get_neighbors",
    "Get all nodes connected to a node via edges (1-hop). READ-ONLY. Returns the node, its neighbors, and connecting edges.",
    { node_id: z.string().describe("The node ID") },
    ({ node_id }) => ({ content: [{ type: "text" as const, text: toolGetNeighbors(node_id) }] }));

  server.tool("search_nodes",
    "Keyword search across node summaries with optional type filter. Up to 20 results ranked by activation. READ-ONLY. For semantic search, use what_do_i_think instead.",
    { query: z.string().describe("Keyword to search for"), nodeType: z.string().optional().describe("Optional type filter") },
    ({ query, nodeType }) => ({ content: [{ type: "text" as const, text: toolSearchNodes(query, nodeType) }] }));

  server.tool("merge_nodes",
    "Merge two duplicate nodes. Keeps the first, transfers chunks/edges/outcomes from the second, deletes the second. DESTRUCTIVE. Use get_node on both first.",
    { keep_id: z.string().describe("The node ID to keep"), merge_id: z.string().describe("The node ID to merge and delete") },
    ({ keep_id, merge_id }) => ({ content: [{ type: "text" as const, text: toolMergeNodes(keep_id, merge_id) }] }));

  server.tool("archive_node",
    "Set a node's activation to 0. It stays in the graph but drops out of results. WRITES. Prefer correct if the node needs updating rather than removal.",
    { node_id: z.string().describe("The node ID to archive") },
    ({ node_id }) => ({ content: [{ type: "text" as const, text: toolArchiveNode(node_id) }] }));

  server.tool("bootstrap",
    "Guided first-run experience. Returns questions for the agent to ask the user across 5 phases. Feed answers back through capture (without nodeType). Call again with no args to advance phases. Tracks phase state in meta table.",
    { action: z.string().optional().describe("'start' to begin or resume, 'status' to check progress, omit to advance to next phase") },
    ({ action }) => {
      const text = action === "start" || action === "status" ? toolBootstrap(action) : advanceBootstrap();
      return { content: [{ type: "text" as const, text }] };
    });

  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(e => { console.error("Fatal:", e); process.exit(1); });
