"""Tests for the key pool failover and the failure simulation header."""

import litellm
import pytest

from app.main import app
from tests.conftest import TEST_MODEL, make_response


def rate_limit_error() -> litellm.RateLimitError:
    return litellm.RateLimitError(
        message="429 quota exceeded", llm_provider="gemini", model=TEST_MODEL
    )


def server_error() -> litellm.InternalServerError:
    return litellm.InternalServerError(
        message="500 upstream failure", llm_provider="gemini", model=TEST_MODEL
    )


# ------------------------- simulated failures --------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_simulated_failure_of_key_1_falls_over_to_key_2(
    client, fake_completion, status
):
    fake = fake_completion(make_response("Answered by the second key."))

    response = client.post(
        "/chat",
        json={"session_id": f"s-sim-{status}", "message": f"Simulate a {status}"},
        headers={"X-Simulate-Failure": f"key_1={status}"},
    )

    assert response.status_code == 200
    body = response.json()
    metadata = body["metadata"]
    assert metadata["key_id"] == "key_2"
    assert metadata["fallback_attempts"] == 1
    assert metadata["resolution_source"] == "LIVE_INFERENCE"

    # A simulated request returns a genuine model answer, not an error text.
    assert body["reply"] == "Answered by the second key."

    # key_1 never reached the provider, key_2 did.
    assert fake.call_count == 1
    assert fake.api_key_of_call(0) == "test-key-two"


def test_both_keys_simulated_down_returns_503(client, fake_completion):
    fake = fake_completion(make_response("never used"))

    response = client.post(
        "/chat",
        json={"session_id": "s-sim-both", "message": "Both keys are down"},
        headers={"X-Simulate-Failure": "key_1=429,key_2=500"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "all providers unavailable"}
    assert fake.call_count == 0


def test_wildcard_failure_returns_503_and_leaves_history_untouched(
    client, fake_completion
):
    fake = fake_completion(make_response("never used"))

    # Seed one successful exchange so there is history that must not change.
    client.post("/chat", json={"session_id": "s-wild", "message": "First question"})
    history_before = client.get("/sessions/s-wild/messages").json()["messages"]
    assert len(history_before) == 2

    response = client.post(
        "/chat",
        json={"session_id": "s-wild", "message": "This one cannot be answered"},
        headers={"X-Simulate-Failure": "*=429"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "all providers unavailable"}

    history_after = client.get("/sessions/s-wild/messages").json()["messages"]
    assert history_after == history_before
    assert app.state.db.count_messages("s-wild") == 2


def test_wildcard_failure_on_a_new_session_saves_nothing(client, fake_completion):
    fake_completion(make_response("never used"))

    response = client.post(
        "/chat",
        json={"session_id": "s-wild-new", "message": "Nothing should be stored"},
        headers={"X-Simulate-Failure": "*=429"},
    )

    assert response.status_code == 503
    assert app.state.db.count_messages("s-wild-new") == 0
    # The session was never created either.
    assert client.get("/sessions/s-wild-new/messages").status_code == 404


def test_an_explicit_key_status_overrides_the_wildcard(client, fake_completion):
    """key_1 is excluded from the wildcard by naming it with a healthy path."""
    fake = fake_completion(make_response("Answered by the second key."))

    # The wildcard takes every key down, so nothing can answer.
    down = client.post(
        "/chat",
        json={"session_id": "s-mix-a", "message": "Mixed header A"},
        headers={"X-Simulate-Failure": "*=503"},
    )
    assert down.status_code == 503

    # Naming only key_1 leaves key_2 healthy.
    up = client.post(
        "/chat",
        json={"session_id": "s-mix-b", "message": "Mixed header B"},
        headers={"X-Simulate-Failure": "key_1=503"},
    )
    assert up.status_code == 200
    assert up.json()["metadata"]["key_id"] == "key_2"


def test_invalid_header_parts_are_ignored(client, fake_completion):
    fake = fake_completion(make_response("Still answered by key_1."))

    response = client.post(
        "/chat",
        json={"session_id": "s-bad-header", "message": "Bad header parts"},
        headers={"X-Simulate-Failure": "garbage,key_1=999,key_2,=429,key_1=abc"},
    )

    # None of those parts is a valid <key_id>=<status> pair with an allowed
    # status, so the request runs normally on key_1.
    assert response.status_code == 200
    assert response.json()["metadata"]["key_id"] == "key_1"
    assert response.json()["metadata"]["fallback_attempts"] == 0
    assert fake.call_count == 1


def test_the_header_is_ignored_when_simulation_is_disabled(
    client, fake_completion, settings, monkeypatch
):
    fake = fake_completion(make_response("Simulation is switched off."))
    monkeypatch.setattr(settings, "enable_failure_simulation", False)

    response = client.post(
        "/chat",
        json={"session_id": "s-sim-off", "message": "Header should be ignored"},
        headers={"X-Simulate-Failure": "key_1=429"},
    )

    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["key_id"] == "key_1"
    assert metadata["fallback_attempts"] == 0
    assert fake.api_key_of_call(0) == "test-key-one"


def test_a_simulated_request_skips_the_cache_lookup(client, fake_completion):
    """Simulation must exercise the failover loop, not return a cached hit."""
    fake = fake_completion(
        make_response("First, from key_1."),
        make_response("Second, from key_2."),
    )

    first = client.post(
        "/chat", json={"session_id": "s-skip-a", "message": "Identical question"}
    ).json()
    assert first["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert first["metadata"]["key_id"] == "key_1"

    # The same question would normally hit the cache, but the header makes
    # the request go through the failover loop instead.
    second = client.post(
        "/chat",
        json={"session_id": "s-skip-b", "message": "Identical question"},
        headers={"X-Simulate-Failure": "key_1=429"},
    ).json()

    assert second["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert second["metadata"]["key_id"] == "key_2"
    assert second["metadata"]["fallback_attempts"] == 1
    assert fake.call_count == 2


# --------------------------- real provider errors ---------------------------


def test_a_real_rate_limit_on_key_1_falls_over_to_key_2(client, fake_completion):
    fake = fake_completion(
        rate_limit_error(),
        make_response("Recovered on the second key."),
    )

    response = client.post(
        "/chat", json={"session_id": "s-real-429", "message": "Real rate limit"}
    )

    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["key_id"] == "key_2"
    assert metadata["fallback_attempts"] == 1
    assert response.json()["reply"] == "Recovered on the second key."

    # Both keys were tried, in order.
    assert fake.call_count == 2
    assert fake.api_key_of_call(0) == "test-key-one"
    assert fake.api_key_of_call(1) == "test-key-two"


def test_a_real_server_error_on_key_1_falls_over_to_key_2(client, fake_completion):
    fake_completion(server_error(), make_response("Recovered after a 500."))

    response = client.post(
        "/chat", json={"session_id": "s-real-500", "message": "Real server error"}
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["key_id"] == "key_2"
    assert response.json()["metadata"]["fallback_attempts"] == 1


def test_real_failures_on_every_key_return_503(client, fake_completion):
    fake = fake_completion(rate_limit_error(), rate_limit_error())

    response = client.post(
        "/chat", json={"session_id": "s-real-both", "message": "Everything is down"}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "all providers unavailable"}
    assert fake.call_count == 2
    assert app.state.db.count_messages("s-real-both") == 0


def test_a_timeout_is_treated_as_a_transient_failure(client, fake_completion):
    timeout = litellm.Timeout(
        message="request timed out", model=TEST_MODEL, llm_provider="gemini"
    )
    fake_completion(timeout, make_response("Recovered after a timeout."))

    response = client.post(
        "/chat", json={"session_id": "s-timeout", "message": "Timeout please"}
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["key_id"] == "key_2"


def test_an_empty_reply_is_treated_as_a_failure_so_the_reply_is_never_blank(
    client, fake_completion
):
    fake_completion(
        make_response(""),
        make_response("A real answer from the second key."),
    )

    response = client.post(
        "/chat", json={"session_id": "s-empty", "message": "Empty first reply"}
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "A real answer from the second key."
    assert response.json()["metadata"]["key_id"] == "key_2"


def test_failures_never_leak_a_stack_trace_or_a_key(client, fake_completion):
    fake_completion(rate_limit_error(), rate_limit_error())

    response = client.post(
        "/chat", json={"session_id": "s-leak", "message": "Do not leak anything"}
    )

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    body = response.text
    assert "Traceback" not in body
    assert "test-key-one" not in body
    assert "test-key-two" not in body
    assert response.json() == {"detail": "all providers unavailable"}
