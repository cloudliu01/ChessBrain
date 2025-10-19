import { useCallback, useEffect, useMemo, useState } from "react";
import BoardView from "../components/Match/BoardView";
import {
  ApiError,
  createSession,
  resignSession,
  submitMove,
  undoMove,
} from "../lib/api";
import type { SessionState } from "../types/session";

const DEFAULT_REQUEST = {
  playerColor: "white" as const,
  difficulty: "deterministic" as const,
};

function statusLabel(status: SessionState["status"]) {
  switch (status) {
    case "white_won":
      return "White wins";
    case "black_won":
      return "Black wins";
    case "drawn":
      return "Draw";
    case "aborted":
      return "Aborted";
    default:
      return "In progress";
  }
}

export default function MatchPage() {
  const [session, setSession] = useState<SessionState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [isSubmitting, setSubmitting] = useState(false);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const next = await createSession(DEFAULT_REQUEST);
        if (mounted) {
          setSession(next);
        }
      } catch (err) {
        if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("Failed to start session.");
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  const isHumanTurn = useMemo(() => {
    if (!session) {
      return false;
    }
    const parts = session.currentFen.split(" ");
    if (parts.length < 2) {
      return false;
    }
    const active = parts[1];
    if (session.playerColor === "white") {
      return active === "w";
    }
    return active === "b";
  }, [session]);

  const isGameOver = session ? session.status !== "in_progress" : false;

  const handleMove = useCallback(
    async (uci: string) => {
      if (!session) {
        return false;
      }
      setSubmitting(true);
      try {
        const updated = await submitMove(session.id, uci);
        setSession(updated);
        setInfo(null);
        return true;
      } catch (err) {
        const message =
          err instanceof ApiError ? err.message : "Move rejected. Try again.";
        setInfo(message);
        return false;
      } finally {
        setSubmitting(false);
      }
    },
    [session],
  );

  const handleUndo = useCallback(async () => {
    if (!session) {
      return;
    }
    setSubmitting(true);
    try {
      const updated = await undoMove(session.id);
      setSession(updated);
      setInfo("Move undone.");
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Unable to undo the last move.";
      setInfo(message);
    } finally {
      setSubmitting(false);
    }
  }, [session]);

  const handleResign = useCallback(async () => {
    if (!session) {
      return;
    }
    try {
      const updated = await resignSession(session.id);
      setSession(updated);
      setInfo("You resigned the match.");
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Unable to resign right now.";
      setInfo(message);
    }
  }, [session]);

  if (loading) {
    return (
      <main className="layout">
        <section className="panel" style={{ margin: "auto" }}>
          <p>Loading match...</p>
        </section>
      </main>
    );
  }

  if (error || !session) {
    return (
      <main className="layout">
        <section className="panel" style={{ margin: "auto" }}>
          <h1>Match unavailable</h1>
          <p>{error ?? "Unable to establish a session."}</p>
        </section>
      </main>
    );
  }

  const disabled = isSubmitting || !isHumanTurn || isGameOver;

  return (
    <main className="layout">
      <section className="panel" style={{ flex: "0 0 560px" }}>
        <header style={{ marginBottom: "1rem" }}>
          <h1 style={{ margin: 0 }}>ChessBrain Match Play</h1>
          <p style={{ margin: "0.5rem 0 0" }}>
            Model: <strong>{session.activeModelVersion ?? "unknown"}</strong>
          </p>
          <p style={{ margin: "0.25rem 0 0" }}>
            Status: <strong>{statusLabel(session.status)}</strong>
          </p>
        </header>
        <BoardView
          fen={session.currentFen}
          perspective={session.playerColor}
          disabled={disabled}
          onMove={handleMove}
        />
        <div style={{ marginTop: "1rem", display: "flex", gap: "0.5rem" }}>
          <button
            type="button"
            onClick={handleUndo}
            disabled={isSubmitting || session.moves.length === 0}
          >
            Undo
          </button>
          <button
            type="button"
            onClick={handleResign}
            disabled={isSubmitting || isGameOver}
          >
            Resign
          </button>
        </div>
        {info && (
          <div style={{ marginTop: "0.75rem", color: "#f5b971" }}>{info}</div>
        )}
      </section>
      <aside
        className="panel"
        style={{
          flex: "1 1 auto",
          maxWidth: "520px",
          backgroundColor: "rgba(18, 24, 32, 0.85)",
          borderLeft: "1px solid rgba(255,255,255,0.05)",
        }}
      >
        <h2 style={{ marginTop: 0 }}>Moves</h2>
        <ol style={{ paddingLeft: "1.25rem" }}>
          {session.moves.map((move, index) => (
            <li key={`${move.timestamp}-${index}`} style={{ marginBottom: 8 }}>
              <strong>{move.actor === "human" ? "You" : "AI"}:</strong>{" "}
              <span>{move.san}</span>
              {move.rationale?.length ? (
                <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>
                  {move.rationale.join(", ")}
                </div>
              ) : null}
            </li>
          ))}
        </ol>
      </aside>
    </main>
  );
}
