"""Tests for request validation on POST /chat.

Every invalid request must be rejected with a 422 and a JSON body, and it
must never reach the provider or the history.
"""

import pytest

from app.main import app


@pytest.mark.parametrize(
    "payload, case",
    [
        ({"session_id": "s-1", "message": ""}, "empty message"),
        ({"session_id": "s-1", "message": "      "}, "whitespace only message"),
        ({"session_id": "s-1", "message": "x" * 4001}, "message over 4000 chars"),
        ({"message": "hello"}, "missing session_id"),
        ({"session_id": "", "message": "hello"}, "empty session_id"),
        ({"session_id": "s" * 129, "message": "hello"}, "session_id over 128 chars"),
        ({"session_id": "s-1"}, "missing message"),
    ],
)
def test_invalid_payloads_return_422(client, fake_completion, payload, case):
    fake = fake_completion()

    response = client.post("/chat", json=payload)

    assert response.status_code == 422, case
    assert "detail" in response.json(), case
    # A rejected request never reaches the provider.
    assert fake.call_count == 0, case


def test_malformed_json_returns_422(client):
    response = client.post(
        "/chat",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert "detail" in response.json()


def test_message_of_exactly_4000_chars_is_accepted(client, fake_completion):
    fake_completion()

    response = client.post(
        "/chat", json={"session_id": "s-4000", "message": "x" * 4000}
    )

    assert response.status_code == 200


def test_surrounding_whitespace_is_trimmed_before_the_length_check(
    client, fake_completion
):
    fake_completion()

    # 4000 real characters plus padding is still valid after trimming.
    response = client.post(
        "/chat", json={"session_id": "s-trim", "message": "   " + "x" * 4000 + "   "}
    )

    assert response.status_code == 200
    stored = client.get("/sessions/s-trim/messages").json()["messages"]
    assert stored[0]["content"] == "x" * 4000


def test_a_422_is_not_written_to_history(client, fake_completion):
    fake_completion()

    assert client.post("/chat", json={"session_id": "s-422", "message": " "}).status_code == 422

    assert app.state.db.count_messages("s-422") == 0
    assert client.get("/sessions/s-422/messages").status_code == 404
