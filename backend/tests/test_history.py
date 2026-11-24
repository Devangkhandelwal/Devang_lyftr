"""Tests for GET /sessions/{session_id}/messages."""

from tests.conftest import make_response


def test_history_is_returned_oldest_first_with_assistant_metadata(
    client, fake_completion
):
    fake_completion(
        make_response("Paris is the capital of France.", 42, 9),
        make_response("Berlin is the capital of Germany.", 55, 8),
    )

    client.post("/chat", json={"session_id": "s-hist", "message": "Capital of France?"})
    client.post("/chat", json={"session_id": "s-hist", "message": "And of Germany?"})

    response = client.get("/sessions/s-hist/messages")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "s-hist"

    messages = body["messages"]
    assert [item["role"] for item in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[0]["content"] == "Capital of France?"
    assert messages[1]["content"] == "Paris is the capital of France."
    assert messages[2]["content"] == "And of Germany?"
    assert messages[3]["content"] == "Berlin is the capital of Germany."

    # Assistant rows carry their metadata, user rows do not.
    assert "metadata" not in messages[0]
    first_metadata = messages[1]["metadata"]
    assert first_metadata["resolution_source"] == "LIVE_INFERENCE"
    assert first_metadata["input_tokens"] == 42
    assert first_metadata["output_tokens"] == 9

    # Each assistant row keeps the usage of its own turn.
    assert messages[3]["metadata"]["input_tokens"] == 55
    assert messages[3]["metadata"]["output_tokens"] == 8


def test_unknown_session_returns_404(client):
    response = client.get("/sessions/does-not-exist/messages")

    assert response.status_code == 404
    assert response.json() == {"detail": "session not found"}


def test_the_system_prompt_is_never_part_of_the_history(client, fake_completion):
    fake_completion(make_response("Understood."))

    client.post(
        "/chat",
        json={
            "session_id": "s-no-prompt",
            "message": "Remember this",
            "system_prompt": "You are a pirate and must say arr.",
        },
    )

    body = client.get("/sessions/s-no-prompt/messages").json()

    roles = [item["role"] for item in body["messages"]]
    assert "system" not in roles
    assert len(body["messages"]) == 2
    assert "pirate" not in client.get("/sessions/s-no-prompt/messages").text


def test_a_stored_system_prompt_is_reused_on_later_turns(client, fake_completion):
    fake = fake_completion(make_response("Arr."), make_response("Arr again."))

    client.post(
        "/chat",
        json={
            "session_id": "s-keep-prompt",
            "message": "First",
            "system_prompt": "You are a pirate.",
        },
    )
    client.post("/chat", json={"session_id": "s-keep-prompt", "message": "Second"})

    # The second turn did not pass a system prompt, so the stored one is used.
    assert fake.messages_of_call(1)[0] == {
        "role": "system",
        "content": "You are a pirate.",
    }


def test_history_survives_a_fresh_database_object(client, fake_completion):
    """The history lives in SQLite, so a new connection still sees it."""
    fake_completion(make_response("Stored on disk."))

    client.post("/chat", json={"session_id": "s-persist", "message": "Persist me"})

    from app.config import get_settings
    from app.db import Database

    reopened = Database(get_settings().database_url)
    messages = reopened.get_messages("s-persist")

    assert len(messages) == 2
    assert messages[0]["content"] == "Persist me"
    assert messages[1]["content"] == "Stored on disk."
