import { ANTHROPIC_API_KEY, VALID_NODE_TYPES, VALID_CONFIDENCE, VALID_EPISTEMIC_TAGS } from "./types.js";
import type { ExtractedPattern } from "./types.js";

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

export async function extractPatterns(text: string): Promise<ExtractedPattern[]> {
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
