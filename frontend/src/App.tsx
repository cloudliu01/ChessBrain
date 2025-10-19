import { Navigate, Route, Routes } from "react-router-dom";
import MatchPage from "./pages/MatchPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/match" replace />} />
      <Route path="/match" element={<MatchPage />} />
    </Routes>
  );
}

export default App;
