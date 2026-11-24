// Backend access. The base URL is injected at build time by Vite.

const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(
  /\/+$/,
  ""
);

/**
 * Turn whatever the backend (or the network) produced into one readable line.
 * FastAPI validation errors arrive as a list of objects, so they are
 * flattened into a sentence instead of being shown as raw JSON.
 */
function describeError(detail, status) {
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        const field = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "";
        const message = item.msg || "invalid value";
        return field ? `${field}: ${message}` : message;
      })
      .filter(Boolean);
    if (parts.length) {
      return parts.join("; ");
    }
  }
  if (detail && typeof detail === "object") {
    return JSON.stringify(detail);
  }
  return `Request failed with status ${status}`;
}

/**
 * Send one chat message.
 * Resolves with the parsed body, or throws an Error carrying a readable
 * message plus the HTTP status when there is one.
 */
export async function sendChatMessage({ sessionId, message, systemPrompt }) {
  const body = { session_id: sessionId, message };
  if (systemPrompt && systemPrompt.trim()) {
    body.system_prompt = systemPrompt.trim();
  }

  let response;
  try {
    response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (networkError) {
    // The backend is unreachable, CORS blocked the call, and so on.
    const error = new Error(
      `Could not reach the backend at ${API_URL}. ${networkError.message}`
    );
    error.status = 0;
    throw error;
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const error = new Error(
      describeError(payload ? payload.detail : null, response.status)
    );
    error.status = response.status;
    throw error;
  }

  return payload;
}

export { API_URL };
