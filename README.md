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

Every conversation you have contains cognitive patterns you don't notice. Decision rules you apply without naming them. Tensions between things you believe. Assumptions you've never tested.

This server extracts those patterns, stores them in a typed graph, and lets any AI agent query them over MCP. The agent can find contradictions in your thinking, bridge connections between domains you never linked, or predict how you'd approach a decision you haven't faced yet.

It is not a memory system. Memory stores what you said. This models how you decide.

## First run

On first use, the server walks you through a bootstrap conversation. It starts with a real decision you made recently and works backward to understand your heuristics, mental models, and assumptions. Takes about 20 minutes. After that, the extraction pipeline runs against your conversations to build the graph continuously.

The bootstrap tool returns questions for your agent to ask you. You answer naturally. Each answer gets classified and stored as typed nodes in the graph.

## How it works

When you capture text (or an agent does on your behalf), an LLM classifies each statement against an ordered checklist. Is this a decision rule? A framework for thinking? A tension between beliefs? A preference? The checklist forces specific types before falling through to the generic "idea" bucket.

Each node gets:
- A type (heuristic, mental_model, tension, value, assumption, preference, question, project, idea)
- An epistemic tag (assertion, hypothesis, speculation)
- A confidence score (strong, tentative, uncertain)
- An activation level that decays over time

Typed edges connect nodes: supports, contradicts, evolved_into, depends_on. This makes it a real graph, not a flat embedding store. Agents can traverse relationships, not just retrieve by similarity.

Activation decay means the graph stays current. Values decay slowly. Ideas decay fast. If you haven't revisited or reinforced a pattern in weeks, it fades. Validated heuristics stay hot.

Scoring uses Reciprocal Rank Fusion across vector similarity, keyword matching, and activation. Hub dampening prevents well-connected nodes from dominating every query.

## The extraction problem

You can tell an agent to "capture that thought" mid-conversation. It won't. Not reliably. Agents follow the task at hand and forget side quests.

The real path is a backend pipeline that processes your conversation transcripts on a schedule. Your conversations already contain the patterns. You just need something reading them after the fact.

The capture tool supports both modes. Pass a nodeType and it stores directly. Omit nodeType and it runs inline extraction to classify the text into properly typed nodes.

## Tools

| Tool | What it does | Side effects |
|------|-------------|-------------|
| `what_do_i_think` | Query your thinking on a topic | READ-ONLY (bumps activation) |
| `what_connects` | Find bridges between two domains | READ-ONLY |
| `what_tensions_exist` | Surface contradictions and weak spots | READ-ONLY |
| `where_am_i_uncertain` | Find low-confidence or untested patterns | READ-ONLY |
| `suggest_exploration` | Forgotten patterns near your current context | READ-ONLY |
| `how_would_user_decide` | Reconstruct reasoning for a new decision | READ-ONLY |
| `what_has_changed` | Timeline of how your thinking evolved | READ-ONLY |
| `capture` | Add a thought (with or without inline extraction) | WRITES |
| `correct` | Fix a node. Strongest learning signal. | WRITES |
| `record_outcome` | Track what happened after a decision | WRITES |
| `get_node` | Inspect a node with all its data | READ-ONLY |
| `get_neighbors` | 1-hop graph traversal from a node | READ-ONLY |
| `search_nodes` | Keyword search with optional type filter | READ-ONLY |
| `merge_nodes` | Combine duplicate nodes | DESTRUCTIVE |
| `archive_node` | Drop a node from results without deleting | WRITES |
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

The vector search is brute-force cosine over all stored embeddings. Fine for personal use up to maybe 100K nodes. Past that you would need a real vector index.

Extraction can hallucinate patterns from weak evidence. The prompt caps at 8 patterns per input and requires explicit evidence for "strong" confidence, but it is still an LLM reading your words and guessing what you meant.

The graph is only as good as the conversations you have. If you only talk about code, it will only model how you think about code.

---

Built by [multimail.dev](https://multimail.dev)
