# thinking-mcp

Portable MCP server for cognitive pattern storage and graph queries.

## Rules

- Single file: everything lives in src/index.ts until there's a concrete reason to split
- No cloud dependencies. SQLite via sql.js (WASM). Embeddings and extraction via direct fetch.
- No emdashes in any prose (README, comments, tool descriptions). Use periods and restructured sentences.
- Tool descriptions must state: what it does, what it does NOT do, side effects (READ-ONLY or writes)
- Never use console.log (corrupts JSON-RPC on stdout). Use console.error for all logging.
- Use string literal unions for type validation, not enums
- Validate embedding dimensions on every embed call
- Use bun, not npm

## Node Types

Valid: idea, question, heuristic, value, mental_model, assumption, tension, preference, project

## Epistemic Tags

Valid: assertion, hypothesis, speculation, quoting, rejected

## Confidence Levels

Valid: strong, tentative, uncertain. Default: tentative.

## Testing

No formal test suite. Verify by:
- `bun run build` for compilation
- `bun run dev` for stdio handshake test
- Manual MCP tool calls via Claude Code or Claude Desktop
