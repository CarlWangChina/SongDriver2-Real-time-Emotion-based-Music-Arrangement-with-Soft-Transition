import "./App.css";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import Adaptation from "./pages/Adaptation";
import Index from "./pages/Index";
import Recommendation from "./pages/Recommendation";
import Information from "./pages/Information";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Index />}></Route>
        <Route path="/home" element={<Home />}></Route>
        <Route path="/adaptation" element={<Adaptation />}></Route>
        <Route path="/recommendation" element={<Recommendation />}></Route>
        <Route path="/information" element={<Information />}></Route>
      </Routes>
    </Router>
  );
}

export default App;
