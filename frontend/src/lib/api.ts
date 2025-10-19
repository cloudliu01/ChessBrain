import type {
  CreateSessionRequest,
  SessionState,
} from "../types/session";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

export class ApiError extends Error {
  constructor(public readonly code: string, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    let code = "unknown_error";
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      code = payload.code ?? code;
      message = payload.message ?? message;
    } catch {
      // ignore parse errors
    }
    throw new ApiError(code, message);
  }

  return (await response.json()) as T;
}

export async function createSession(
  body: CreateSessionRequest,
): Promise<SessionState> {
  return request<SessionState>("/sessions", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function fetchSession(sessionId: string): Promise<SessionState> {
  return request<SessionState>(`/sessions/${sessionId}`);
}

export async function submitMove(
  sessionId: string,
  uci: string,
): Promise<SessionState> {
  return request<SessionState>(`/sessions/${sessionId}/moves`, {
    method: "POST",
    body: JSON.stringify({ uci }),
  });
}

export async function undoMove(sessionId: string): Promise<SessionState> {
  return request<SessionState>(`/sessions/${sessionId}/undo`, {
    method: "POST",
  });
}

export async function resignSession(sessionId: string): Promise<SessionState> {
  return request<SessionState>(`/sessions/${sessionId}/resign`, {
    method: "POST",
  });
}
