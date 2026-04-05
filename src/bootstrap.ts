import { db, persistDb } from "./db.js";

const PHASES = [
  {
    name: "decisions",
    questions: [
      "Tell me about a decision you made in the last two weeks.",
      "What options did you reject?",
      "What tradeoff mattered most?",
      "What rule did you apply, even if you didn't name it at the time?",
    ],
  },
  {
    name: "heuristics",
    questions: [
      "When you're stuck on a problem, what's your first move?",
      "What decision rule do you apply that others might not?",
      "How do you decide what NOT to work on?",
    ],
  },
  {
    name: "mental_models",
    questions: [
      "What framework do you use repeatedly across different domains?",
      "How do you evaluate whether an idea is worth pursuing?",
    ],
  },
  {
    name: "tensions",
    questions: [
      "Where do two things you believe pull in opposite directions?",
      "What tradeoff do you keep revisiting without a clear answer?",
    ],
  },
  {
    name: "assumptions",
    questions: [
      "What do you assume is true that you haven't tested?",
      "What would change your mind about something you hold strongly?",
    ],
  },
];

export function toolBootstrap(action?: string): string {
  const phaseRow = db.exec("SELECT value FROM meta WHERE key = 'bootstrap_phase'");
  const currentPhase = phaseRow.length > 0 && phaseRow[0].values.length > 0 ? parseInt(phaseRow[0].values[0][0] as string) : 0;

  const completeRow = db.exec("SELECT value FROM meta WHERE key = 'bootstrap_complete'");
  const isComplete = completeRow.length > 0 && completeRow[0].values[0]?.[0] === "true";

  if (action === "status" || isComplete) {
    const nodeCount = db.exec("SELECT COUNT(*) FROM nodes");
    const count = nodeCount[0]?.values[0]?.[0] || 0;
    return JSON.stringify({
      complete: isComplete,
      phase: currentPhase,
      total_phases: PHASES.length,
      node_count: count,
      message: isComplete
        ? `Bootstrap complete. ${count} nodes in the graph. Use capture to keep adding patterns from conversations.`
        : `Bootstrap in progress. Phase ${currentPhase + 1}/${PHASES.length}: ${PHASES[currentPhase]?.name || "done"}.`,
    }, null, 2);
  }

  if (currentPhase >= PHASES.length) {
    db.run("INSERT OR REPLACE INTO meta (key, value) VALUES ('bootstrap_complete', 'true')");
    persistDb();
    return JSON.stringify({ complete: true, message: "All phases done. Bootstrap complete." });
  }

  const phase = PHASES[currentPhase];
  return JSON.stringify({
    complete: false,
    phase: currentPhase + 1,
    phase_name: phase.name,
    total_phases: PHASES.length,
    questions: phase.questions,
    instructions: "Ask the user these questions one at a time. Feed each answer back through the capture tool (without nodeType, so extraction runs automatically). When done with all questions, call bootstrap again to advance to the next phase.",
    advance: `After capturing answers, call bootstrap with no arguments to move to phase ${currentPhase + 2}.`,
  }, null, 2);
}

export function advanceBootstrap(): string {
  const phaseRow = db.exec("SELECT value FROM meta WHERE key = 'bootstrap_phase'");
  const currentPhase = phaseRow.length > 0 && phaseRow[0].values.length > 0 ? parseInt(phaseRow[0].values[0][0] as string) : 0;
  const nextPhase = currentPhase + 1;

  if (nextPhase >= PHASES.length) {
    db.run("INSERT OR REPLACE INTO meta (key, value) VALUES ('bootstrap_phase', ?)", [String(nextPhase)]);
    db.run("INSERT OR REPLACE INTO meta (key, value) VALUES ('bootstrap_complete', 'true')");
    persistDb();
    return JSON.stringify({ complete: true, message: "Bootstrap complete. All phases done." });
  }

  db.run("INSERT OR REPLACE INTO meta (key, value) VALUES ('bootstrap_phase', ?)", [String(nextPhase)]);
  persistDb();
  return toolBootstrap();
}
