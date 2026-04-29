# thinking-mcp

MCP server that models how you think. Not what you know.

## Install

```json
{
  "mcpServers": {
    "thinking": {
      "command": "npx",
      "args": ["-y", "@multimail/thinking-mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "VOYAGE_API_KEY": "pa-..."
      }
    }
  }
}
```

Works with Claude Code, Claude Desktop, Cursor, Windsurf, and any MCP client.

## What this is

Every conversation contains cognitive patterns you don't notice. Decision rules you apply without naming them. Tensions between things you believe. Assumptions you've never tested.

Kahneman calls it System 1 — fast, automatic, invisible. The heuristics that fire before you know you're deciding. Most AI memory systems store what you said. This models how you decide.

thinking-mcp captures your heuristics, values, tensions, assumptions, preferences, and mental models as a typed graph. Nodes connect through real edges — supports, contradicts, depends on, evolved into. When an agent queries the graph, structurally important patterns surface first.

It is not a memory vault. It is a mirror for your decision architecture.

## First run

On first use, the server walks you through a bootstrap conversation. It starts from a real decision you made recently. Then it works backward into the heuristics and assumptions behind it.

The bootstrap tool returns questions for your agent to ask you. You answer naturally. Each answer gets captured, typed, and added to the graph. The first pass takes about 20 minutes. After that, capture keeps extending the model.

## How it works

Capture starts with typed extraction. An LLM classifies each statement against an ordered checklist — is this a decision rule? A framework for thinking? A tension between beliefs? The checklist forces specific types before falling through to "idea." Nine node types: heuristic, mental_model, tension, value, assumption, preference, question, project, idea.

Then it builds the graph automatically. Two mechanisms create edges:

1. **Extraction-suggested relationships.** The LLM identifies connections between concepts during classification. "This heuristic contradicts that assumption." These become typed edges.
2. **Vector similarity.** New nodes connect to semantically similar existing nodes. Edge types are inferred from the node type pair — a heuristic near a value becomes `derived_from`, a tension near a value becomes `contradicts`.

The graph gets structural weighting through PageRank. Patterns that connect many other patterns rank higher. Patterns that bridge disconnected domains rank higher. Structural importance propagates through the topology.

Retrieval fuses five signals via Reciprocal Rank Fusion:

- **Vector similarity** — semantic closeness to the query
- **Keyword overlap** — exact term matches
- **Activation** — recency and reinforcement (decays by type: values persist, ideas fade)
- **Outcome history** — track record from recorded decisions
- **PageRank** — structural importance in the graph

System 1 produces the pattern. System 2 gets the ranked evidence.

## Tools

| Tool | What it does | Side effects |
|------|-------------|-------------|
| `ping` | Health check with node, chunk, and edge counts plus DB path | READ-ONLY |
| `what_do_i_think` | Query your thinking on a topic with 5-signal ranking: vector, keyword, activation, outcome, and PageRank | READ-ONLY (bumps activation) |
| `what_connects` | Find bridges between two domains | READ-ONLY |
| `what_tensions_exist` | Surface contradictions and weak spots | READ-ONLY |
| `where_am_i_uncertain` | Find low-confidence or untested patterns | READ-ONLY |
| `suggest_exploration` | Surface forgotten but structurally important nodes using similarity plus PageRank | READ-ONLY |
| `how_would_user_decide` | Reconstruct likely reasoning for a new decision, ranked with structural importance | READ-ONLY |
| `what_has_changed` | Timeline of how your thinking evolved | READ-ONLY |
| `capture` | Add a thought with typed extraction and automatic edge creation through similarity and extracted relationships | WRITES |
| `correct` | Fix a node. Strongest learning signal. | WRITES |
| `record_outcome` | Track what happened after a decision | WRITES |
| `get_node` | Inspect a node with all its data | READ-ONLY |
| `get_neighbors` | Return real 1-hop graph topology: connected nodes and the edges between them | READ-ONLY |
| `search_nodes` | Keyword search with optional type filter | READ-ONLY |
| `merge_nodes` | Combine duplicate nodes | DESTRUCTIVE |
| `archive_node` | Drop a node from results without deleting | WRITES |
| `get_framing` | Return lens-shaping questions to answer before any substantive response | READ-ONLY |
| `seed_framing` | Set or replace framing directives (max 5) | DESTRUCTIVE |
| `bootstrap` | Guided first-run Q&A (5 phases) | WRITES |

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | | Claude Haiku for inline extraction |
| `VOYAGE_API_KEY` | Yes | | Voyage AI for embeddings |
| `THINKING_MCP_DB_PATH` | No | `~/.thinking-mcp/mind.db` | SQLite database location |
| `THINKING_MCP_EMBEDDING_PROVIDER` | No | `voyage` | `voyage`, `openai`, or `ollama` |

## What leaves your machine

Embedding text goes to Voyage AI (or your configured provider). Extraction text goes to Anthropic when capture is called without a nodeType. Everything else stays local in SQLite. No telemetry. No analytics.

## Limitations

Vector search is brute-force cosine similarity over all stored embeddings. This keeps the system zero-infrastructure — no vector database, no external index. Fine for personal use up to roughly 100K nodes. Past that you need a real vector index.

Extraction is still an LLM interpreting your language. It can overread weak evidence. The prompt caps at 8 patterns per input and requires explicit evidence for "strong" confidence, but it is making judgment calls about what you meant.

The graph is only as good as what enters the conversation stream. If you only talk about code, it will only model how you think about code.

---

Built by [multimail.dev](https://multimail.dev)
