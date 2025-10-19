export type MoveActor = "human" | "ai";

export interface MoveRecord {
  san: string;
  uci: string;
  actor: MoveActor;
  timestamp: string;
  evaluation?: number | null;
  rationale?: string[];
}

export type SessionStatus =
  | "in_progress"
  | "white_won"
  | "black_won"
  | "drawn"
  | "aborted";

export interface SessionState {
  id: string;
  status: SessionStatus;
  playerColor: "white" | "black";
  difficulty: "deterministic" | "stochastic";
  currentFen: string;
  moves: MoveRecord[];
  activeModelVersion?: string | null;
  undoCount: number;
  evaluation?: number | null;
  startedAt: string;
  updatedAt: string;
  endedAt?: string | null;
  traceId?: string | null;
}

export interface CreateSessionRequest {
  playerColor: "white" | "black";
  difficulty: "deterministic" | "stochastic";
}
