from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, List, Protocol
from uuid import UUID, uuid4

import chess

from src.chessbrain.domain.models.inference_service import InferenceService, MoveSuggestion


class SessionStatus(str, Enum):
    in_progress = "in_progress"
    white_won = "white_won"
    black_won = "black_won"
    drawn = "drawn"
    aborted = "aborted"


class PlayerColor(str, Enum):
    white = "white"
    black = "black"


class SessionDifficulty(str, Enum):
    deterministic = "deterministic"
    stochastic = "stochastic"


class MoveActor(str, Enum):
    human = "human"
    ai = "ai"


@dataclass
class MoveRecord:
    san: str
    uci: str
    actor: MoveActor
    timestamp: datetime
    evaluation: float | None = None
    rationale: List[str] = field(default_factory=list)


@dataclass
class GameSession:
    id: UUID
    status: SessionStatus
    player_color: PlayerColor
    difficulty: SessionDifficulty
    initial_fen: str
    current_fen: str
    moves: List[MoveRecord] = field(default_factory=list)
    active_model_version: str | None = None
    evaluation: float | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    undo_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


class GameSessionRepository(Protocol):
    """Persistence contract for session entities."""

    def create(self, session: GameSession) -> GameSession:
        ...

    def get(self, session_id: UUID) -> GameSession | None:
        ...

    def save(self, session: GameSession) -> GameSession:
        ...


class SessionError(RuntimeError):
    """Base class for session-related domain errors."""

    code: str = "session_error"


class SessionNotFoundError(SessionError):
    code = "session_not_found"


class IllegalMoveError(SessionError):
    code = "illegal_move"


class SessionCompletedError(SessionError):
    code = "session_completed"


class UndoNotAvailableError(SessionError):
    code = "nothing_to_undo"


class SessionManager:
    """Coordinate chess sessions, applying human moves and model responses."""

    def __init__(
        self,
        repository: GameSessionRepository,
        inference: InferenceService,
    ) -> None:
        self._repository = repository
        self._inference = inference

    def create_session(
        self,
        *,
        player_color: PlayerColor,
        difficulty: SessionDifficulty,
        initial_fen: str | None = None,
    ) -> GameSession:
        board = chess.Board(initial_fen) if initial_fen else chess.Board()
        now = datetime.now(timezone.utc)
        session = GameSession(
            id=uuid4(),
            status=SessionStatus.in_progress,
            player_color=player_color,
            difficulty=difficulty,
            initial_fen=board.fen(),
            current_fen=board.fen(),
            moves=[],
            active_model_version=self._inference.model_version or "unversioned",
            evaluation=None,
            started_at=now,
            updated_at=now,
        )

        if player_color == PlayerColor.black and not board.is_game_over(claim_draw=True):
            self._perform_ai_move(session, board)

        self._sync_from_board(session, board)
        return self._repository.create(session)

    def get_session(self, session_id: UUID) -> GameSession:
        session = self._repository.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id} not found.")
        return session

    def submit_move(self, session_id: UUID, uci: str) -> GameSession:
        session = self.get_session(session_id)
        if session.status is not SessionStatus.in_progress:
            raise SessionCompletedError(f"Session {session_id} already completed.")

        board = self._build_board(session)
        expected_color = self._human_turn(session)
        if board.turn != expected_color:
            raise IllegalMoveError("It is not the human player's turn.")

        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise IllegalMoveError(f"Invalid UCI string: {uci}") from exc

        if move not in board.legal_moves:
            raise IllegalMoveError(f"Move {uci} is not legal in the current position.")

        now = datetime.now(timezone.utc)
        san = board.san(move)
        board.push(move)
        session.moves.append(
            MoveRecord(
                san=san,
                uci=uci,
                actor=MoveActor.human,
                timestamp=now,
            )
        )
        session.updated_at = now
        self._sync_from_board(session, board)

        if session.status is SessionStatus.in_progress and board.turn != expected_color:
            self._perform_ai_move(session, board)

        return self._repository.save(session)

    def undo_last(self, session_id: UUID) -> GameSession:
        session = self.get_session(session_id)
        if not session.moves:
            raise UndoNotAvailableError("No moves to undo.")

        expected_turn = self._human_turn(session)

        # Remove moves until it is once again the human player's turn.
        while session.moves:
            session.moves.pop()
            board = self._build_board(session)
            if board.turn == expected_turn or not session.moves:
                break

        session.undo_count += 1
        session.ended_at = None
        session.status = SessionStatus.in_progress
        session.updated_at = datetime.now(timezone.utc)

        board = self._build_board(session)
        self._sync_from_board(session, board)
        return self._repository.save(session)

    def resign(self, session_id: UUID) -> GameSession:
        session = self.get_session(session_id)
        if session.status is not SessionStatus.in_progress:
            return session

        now = datetime.now(timezone.utc)
        session.status = (
            SessionStatus.black_won if session.player_color is PlayerColor.white else SessionStatus.white_won
        )
        session.ended_at = now
        session.updated_at = now
        return self._repository.save(session)

    def _perform_ai_move(self, session: GameSession, board: chess.Board) -> None:
        if board.is_game_over(claim_draw=True):
            self._sync_from_board(session, board)
            return

        suggestion = self._safe_select_move(board, session.difficulty)
        move = suggestion.move
        if move not in board.legal_moves:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                self._sync_from_board(session, board)
                return
            move = legal_moves[0]

        now = datetime.now(timezone.utc)
        san = board.san(move)
        board.push(move)
        session.moves.append(
            MoveRecord(
                san=san,
                uci=move.uci(),
                actor=MoveActor.ai,
                timestamp=now,
                evaluation=suggestion.evaluation,
                rationale=list(suggestion.rationale or []),
            )
        )
        session.evaluation = suggestion.evaluation
        session.updated_at = now
        self._sync_from_board(session, board)

    def _safe_select_move(
        self,
        board: chess.Board,
        difficulty: SessionDifficulty,
    ) -> MoveSuggestion:
        try:
            suggestion = self._inference.select_move(
                board.copy(stack=False),
                difficulty=difficulty.value,
            )
        except Exception as exc:  # pragma: no cover - defensive programming
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                raise
            fallback_move = legal_moves[0]
            return MoveSuggestion(move=fallback_move, probability=1.0, evaluation=None, rationale=["fallback"])
        return suggestion

    def _sync_from_board(self, session: GameSession, board: chess.Board) -> None:
        session.current_fen = board.fen()
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            session.status = SessionStatus.in_progress
            session.ended_at = None
            return

        if outcome.winner is None:
            session.status = SessionStatus.drawn
        elif outcome.winner == chess.WHITE:
            session.status = SessionStatus.white_won
        else:
            session.status = SessionStatus.black_won

        session.ended_at = session.ended_at or datetime.now(timezone.utc)

    def _build_board(self, session: GameSession) -> chess.Board:
        board = chess.Board(session.initial_fen)
        for move in session.moves:
            board.push(chess.Move.from_uci(move.uci))
        return board

    def _human_turn(self, session: GameSession) -> chess.Color:
        return chess.WHITE if session.player_color is PlayerColor.white else chess.BLACK


__all__ = [
    "GameSession",
    "GameSessionRepository",
    "IllegalMoveError",
    "MoveActor",
    "MoveRecord",
    "PlayerColor",
    "SessionCompletedError",
    "SessionDifficulty",
    "SessionManager",
    "SessionNotFoundError",
    "SessionStatus",
    "SessionError",
    "UndoNotAvailableError",
]
