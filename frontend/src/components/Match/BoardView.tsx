import { useCallback } from "react";
import { Chessboard } from "react-chessboard";

interface BoardViewProps {
  fen: string;
  perspective: "white" | "black";
  disabled?: boolean;
  onMove: (uci: string) => Promise<boolean> | boolean;
}

function needsPromotion(
  piece: string,
  targetSquare: string,
  perspective: "white" | "black",
): boolean {
  if (!piece.toLowerCase().startsWith("p")) {
    return false;
  }
  if (perspective === "white") {
    return targetSquare.endsWith("8");
  }
  return targetSquare.endsWith("1");
}

export default function BoardView({
  fen,
  perspective,
  disabled = false,
  onMove,
}: BoardViewProps) {
  const handleDrop = useCallback(
    async (source: string, target: string, piece: string) => {
      if (disabled) {
        return false;
      }
      let uci = `${source}${target}`;
      if (needsPromotion(piece, target, perspective)) {
        uci += "q";
      }
      try {
        const result = await onMove(uci);
        return result;
      } catch {
        return false;
      }
    },
    [disabled, onMove, perspective],
  );

  return (
    <Chessboard
      position={fen}
      boardOrientation={perspective}
      arePiecesDraggable={!disabled}
      onPieceDrop={handleDrop}
      customDarkSquareStyle={{ backgroundColor: "#2d3648" }}
      customLightSquareStyle={{ backgroundColor: "#f0d9b5" }}
      animationDuration={250}
      boardWidth={520}
    />
  );
}
