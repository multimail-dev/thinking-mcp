---
date: 2026-04-28
session: warm-start handoff
prior-summary: docs/session-summaries/2026-04-28_23-00-47.md
---

## Where we left off (2026-04-28, end-of-day)

- Shipped auto-edge creation + PageRank in thinking-mcp@0.2.0 (public npm) and ported to thinking-mcp-pro.
- Extracted a standalone, database-agnostic auto-edge module into Software Incubator (`catalog/ts/graph/auto-edge/`) with 25 tests and LLM sidecar doc. PR #20 open.
- The next target is davidbase-mining — CF Workers + D1, different storage layer than sql.js. User also noted get_framing directives aren't firing there.
- README rewritten in both repos (public: Kahneman System 1/2 framing; private: frank internal docs for agents).

## Open state

- **SI PR #20** — `feat/auto-edge-catalog-entry` → `main` at https://github.com/IiInfra/software-incubator/pull/20. Needs merge.
- **thinking-mcp-pro branch `feat/auto-edge-pagerank`** — merged into `feat/portable-vault-cognitive-mcp`, safe to delete.
- **davidbase-mining get_framing** — user reported directives aren't firing. Not yet investigated.

## Delegated work awaiting supervisor review

None — all delegated work reviewed.

## Recommended first move next session

**Option A (default) — Fix get_framing in davidbase-mining, then port auto-edge:**

```bash
# 1. Merge SI PR
cd ~/Documents/GitHub/Software\ Incubator && gh pr merge 20 --squash --delete-branch

# 2. Investigate davidbase-mining framing
cd ~/Documents/GitHub/davidbase-mining/davidbase_mining/worker
cat src/index.ts | head -60
cat schema.sql

# 3. Plan the D1 port
# Read the SI auto-edge module for the callback interface:
cat ~/Documents/GitHub/Software\ Incubator/catalog/ts/graph/auto-edge/auto-edge.llm.md
```

**Option B (fallback) — Start with davidbase-mining port directly:**

```bash
cd ~/Documents/GitHub/davidbase-mining/davidbase_mining/worker
cat schema.sql
cat src/index.ts
# Then /ce:plan for the D1 auto-edge integration
```

## Decisions locked in this session

- **Two-stage auto-edge creation**: Stage 1 = LLM-suggested (`ae1-` prefix, threshold 0.65), Stage 2 = vector similarity (`ae2-` prefix, same-type 0.75, cross-type 0.80, max 3)
- **PageRank weights**: 2x in RRF (alongside vector 3x, keyword 3x, activation 1x, outcome 0.5x)
- **Edge type inference**: `EDGE_TYPE_MAP` with fallback supports (same type) / inspired_by (cross type)
- **Two-pass capture**: Insert all nodes first, create edges second (so intra-capture relates_to resolves)
- **Error isolation**: autoCreateEdges wraps in try/catch, never blocks capture
- **SI extraction is callback-based**: No sql.js dependency. Consumers wire up their own storage via `AutoEdgeCallbacks`

## Decisions still open

- **davidbase-mining: vendor-copy or import the SI module?** D1/Workers may need a different approach than sql.js repos.
- **Unify vault_edges + cognitive edges in PageRank?** Currently separate graphs. Could compound if merged.
- **davidbase-mining get_framing bug**: Not yet diagnosed. Could be missing framing.yaml, schema mismatch, or code path issue.

## Ship order (updated)

- ✅ thinking-mcp@0.2.0 — auto-edge + PageRank + README (npm published)
- ✅ thinking-mcp-pro — auto-edge + PageRank port + README
- ✅ Software Incubator — auto-edge catalog extraction (PR #20 open)
- davidbase-mining — fix get_framing + port auto-edge + PageRank ← next
- Migrate inline copies to SI catalog imports (low priority)

## Worth capturing as feedback memories (next session, not done yet)

- Two-pass capture pattern is critical for multi-pattern extraction — codex review caught the node-ordering bug
- addCrossLink preservation is the #1 risk when porting from thinking-mcp reference to pro variants
- Codex review Sharp Directive gates are expensive (8 rounds) but effective — every round caught a real issue

## Repo state at handoff

| Repo | Visibility | Branch | HEAD | Safe to delete? |
|------|-----------|--------|------|-----------------|
| thinking-mcp | PUBLIC | `main` | `1f378ee` docs: rewrite README with Kahneman framing | N/A (default) |
| thinking-mcp-pro | PRIVATE | `feat/portable-vault-cognitive-mcp` | `5a0b0b8` docs: rewrite README | N/A (active) |
| thinking-mcp-pro | PRIVATE | `feat/auto-edge-pagerank` | `712d6e1` feat: add auto-edge | YES — merged |
| Software Incubator | PRIVATE | `main` | `2be7f12` chore: NEXT-SESSION | N/A (default) |
| Software Incubator | PRIVATE | `feat/auto-edge-catalog-entry` | `d53d2ec` feat(catalog): add auto-edge | After PR #20 merges |

### Implementation method for davidbase-mining port

**Where the code lives (read in this order):**

1. **Standalone module** (database-agnostic, the cleanest reference):
   `~/Documents/GitHub/Software Incubator/catalog/ts/graph/auto-edge/auto-edge.ts`
   - `AutoEdgeCallbacks` interface — the 6 callbacks you wire to D1
   - `AutoEdgeOptions` — all configurable thresholds
   - `autoCreateEdges()` — the entry point

2. **Integration pattern** (how to wire callbacks to a real DB):
   `~/Documents/GitHub/Software Incubator/catalog/ts/graph/auto-edge/auto-edge.llm.md`
   - Section "Integration pattern" has a complete sql.js example
   - Section "Two-pass capture pattern" shows the multi-pattern flow

3. **Inline implementation** (how it actually runs in production):
   `~/Documents/GitHub/thinking-mcp/src/tools.ts` lines 47-295
   - `EDGE_TYPE_MAP` + `inferEdgeType` (lines 51-63)
   - `createEdgesFromExtraction` (lines 69-100)
   - `createEdgesBySimilarity` (lines 102-131)
   - `autoCreateEdges` (lines 137-154)
   - `computePageRank` (lines 168-212)
   - `toolCapture` with two-pass flow (lines 237-295)

4. **Vendor utilities** (already in SI catalog):
   - `catalog/ts/graph/pagerank/pagerank.ts` — power-iteration PageRank
   - `catalog/ts/vectorize/cosine/cosine.ts` — cosine similarity

**D1-specific considerations for davidbase-mining:**
- D1 uses `env.DB.prepare(sql).bind(...params).first()` instead of sql.js `db.prepare().bind().step()`
- D1 queries are async (return Promises), sql.js is sync
- The `findSimilarByEmbedding` callback may need to use Vectorize instead of brute-force cosine
- Edge IDs with `ae1-`/`ae2-` prefixes work the same in D1
- Schema already has an `edges` table (check `schema.sql` to confirm structure matches)
