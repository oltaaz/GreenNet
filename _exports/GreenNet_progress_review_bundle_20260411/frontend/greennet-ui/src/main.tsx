import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { BackendStatusProvider } from "./hooks/useBackendStatus";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BackendStatusProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </BackendStatusProvider>
  </StrictMode>,
);
