import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { getBackendHealth, isBackendUnavailableError } from "../lib/api";

type BackendStatusValue = {
  status: "checking" | "online" | "offline";
  expectedBackendUrl: string;
  message: string;
  refresh: () => Promise<void>;
};

const BackendStatusContext = createContext<BackendStatusValue | null>(null);

const DEFAULT_EXPECTED_URL = "http://127.0.0.1:8000";

export function BackendStatusProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<"checking" | "online" | "offline">("checking");
  const [expectedBackendUrl, setExpectedBackendUrl] = useState(DEFAULT_EXPECTED_URL);
  const [message, setMessage] = useState("Checking GreenNet backend connectivity.");

  const refresh = useCallback(async (): Promise<void> => {
    try {
      const health = await getBackendHealth();
      setStatus("online");
      setExpectedBackendUrl(health.expectedBackendUrl);
      setMessage(`Connected to GreenNet backend at ${health.expectedBackendUrl}.`);
    } catch (error) {
      setStatus("offline");
      if (isBackendUnavailableError(error)) {
        setExpectedBackendUrl(error.expectedUrl);
        setMessage(`Backend API unavailable. Expected GreenNet FastAPI at ${error.expectedUrl}. Start the backend first.`);
        return;
      }

      setMessage("Backend API unavailable. Start the GreenNet FastAPI backend and refresh.");
    }
  }, []);

  useEffect(() => {
    void refresh();

    const intervalId = window.setInterval(() => {
      void refresh();
    }, 15000);

    const handleFocus = () => {
      void refresh();
    };

    window.addEventListener("focus", handleFocus);
    return () => {
      window.clearInterval(intervalId);
      window.removeEventListener("focus", handleFocus);
    };
  }, [refresh]);

  const value = useMemo(
    () => ({
      status,
      expectedBackendUrl,
      message,
      refresh,
    }),
    [expectedBackendUrl, message, refresh, status],
  );

  return <BackendStatusContext.Provider value={value}>{children}</BackendStatusContext.Provider>;
}

export function useBackendStatus(): BackendStatusValue {
  const context = useContext(BackendStatusContext);
  if (!context) {
    throw new Error("useBackendStatus must be used inside BackendStatusProvider");
  }
  return context;
}
