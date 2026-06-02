// vendor-origin: Software Incubator catalog/ts/vectorize/cosine/cosine.ts
// vendor-version: 2026-04-17
// vendor-sha: 3392af58

/**
 * Cosine similarity in [-1, 1]. Returns 0 when either vector has zero
 * magnitude — avoids NaN from a divide-by-zero and keeps the function
 * total. Callers comparing to a learned threshold do not need to
 * special-case empty embeddings.
 *
 * Mismatched lengths throw — callers must align dimensionality before
 * comparing. (Alternative: return 0; rejected because a length mismatch
 * indicates a caller bug, not a degenerate-but-valid input.)
 */
export function cosine(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`cosine: vector length mismatch (${a.length} vs ${b.length})`);
  }
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i]!;
    const bi = b[i]!;
    dot += ai * bi;
    magA += ai * ai;
    magB += bi * bi;
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}
