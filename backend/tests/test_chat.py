"""Tests for the chat flow: live inference, caching and conversation state."""

from tests.conftest import TEST_MODEL, make_response


def test_normal_chat_is_live_inference_on_the_first_key(client, fake_completion):
    fake = fake_completion(make_response("The capital of France is Paris."))

    response = client.post(
        "/chat",
        json={"session_id": "s-live", "message": "What is the capital of France?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "s-live"
    assert body["reply"] == "The capital of France is Paris."

    metadata = body["metadata"]
    assert metadata["resolution_source"] == "LIVE_INFERENCE"
    assert metadata["model"] == TEST_MODEL
    assert metadata["key_id"] == "key_1"
    assert metadata["fallback_attempts"] == 0
    assert metadata["input_tokens"] == 42
    assert metadata["output_tokens"] == 9
    assert isinstance(metadata["latency_ms"], int)
    assert metadata["latency_ms"] >= 0

    # The first key in the pool was the one actually used.
    assert fake.call_count == 1
    assert fake.api_key_of_call(0) == "test-key-one"


def test_the_system_prompt_leads_the_message_list(client, fake_completion):
    fake = fake_completion()

    client.post(
        "/chat",
        json={
            "session_id": "s-prompt",
            "message": "Hello there",
            "system_prompt": "You only answer in haiku.",
        },
    )

    messages = fake.messages_of_call(0)
    assert messages[0] == {"role": "system", "content": "You only answer in haiku."}
    assert messages[-1] == {"role": "user", "content": "Hello there"}


def test_default_system_prompt_is_used_when_none_is_given(client, fake_completion, settings):
    fake = fake_completion()

    client.post("/chat", json={"session_id": "s-default", "message": "Hello"})

    messages = fake.messages_of_call(0)
    assert messages[0]["content"] == settings.default_system_prompt


def test_same_first_message_in_a_new_session_is_served_from_cache(
    client, fake_completion
):
    fake = fake_completion(make_response("Paris is the capital of France."))

    first = client.post(
        "/chat",
        json={"session_id": "s-cache-a", "message": "What is the capital of France?"},
    ).json()
    second = client.post(
        "/chat",
        json={"session_id": "s-cache-b", "message": "What is the capital of France?"},
    ).json()

    assert first["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert second["metadata"]["resolution_source"] == "LLM_CACHE"

    # The identical stored reply and the stored usage come back.
    assert second["reply"] == first["reply"]
    assert second["metadata"]["input_tokens"] == first["metadata"]["input_tokens"]
    assert second["metadata"]["output_tokens"] == first["metadata"]["output_tokens"]

    # No key was used, and no failover happened, because nothing was called.
    assert second["metadata"]["key_id"] is None
    assert second["metadata"]["fallback_attempts"] == 0

    # The provider was contacted exactly once, for the first request.
    assert fake.call_count == 1


def test_a_cache_hit_is_faster_than_the_live_call(client, fake_completion):
    fake_completion(make_response("Cached answers are quick."))

    live = client.post(
        "/chat", json={"session_id": "s-fast-a", "message": "Why is caching useful?"}
    ).json()
    cached = client.post(
        "/chat", json={"session_id": "s-fast-b", "message": "Why is caching useful?"}
    ).json()

    assert cached["metadata"]["resolution_source"] == "LLM_CACHE"
    assert cached["metadata"]["latency_ms"] < live["metadata"]["latency_ms"]


def test_a_different_message_is_never_served_from_cache(client, fake_completion):
    fake = fake_completion(
        make_response("Paris."),
        make_response("William Shakespeare."),
    )

    first = client.post(
        "/chat", json={"session_id": "s-diff-a", "message": "Capital of France?"}
    ).json()
    second = client.post(
        "/chat", json={"session_id": "s-diff-b", "message": "Who wrote Hamlet?"}
    ).json()

    assert first["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert second["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert second["reply"] == "William Shakespeare."
    assert fake.call_count == 2


def test_the_cache_key_ignores_which_api_key_answered(client, fake_completion):
    """A reply produced by key_2 must still be found when key_1 is used."""
    fake = fake_completion(make_response("Cached across keys."))

    # key_1 is forced to fail, so key_2 produces and stores the reply.
    first = client.post(
        "/chat",
        json={"session_id": "s-xkey-a", "message": "Does the cache ignore keys?"},
        headers={"X-Simulate-Failure": "key_1=429"},
    ).json()
    assert first["metadata"]["key_id"] == "key_2"

    # The same question without the header hits the cache, even though the
    # healthy key_1 would have served it live.
    second = client.post(
        "/chat",
        json={"session_id": "s-xkey-b", "message": "Does the cache ignore keys?"},
    ).json()

    assert second["metadata"]["resolution_source"] == "LLM_CACHE"
    assert second["reply"] == first["reply"]
    assert fake.call_count == 1


def test_multi_turn_sends_the_earlier_exchange_back_to_the_model(
    client, fake_completion
):
    fake = fake_completion(
        make_response("Hello, how can I help?"),
        make_response("You said hello."),
    )

    client.post("/chat", json={"session_id": "s-multi", "message": "Hello"})
    client.post("/chat", json={"session_id": "s-multi", "message": "What did I say?"})

    second_call_messages = fake.messages_of_call(1)
    assert second_call_messages[0]["role"] == "system"
    assert second_call_messages[1] == {"role": "user", "content": "Hello"}
    assert second_call_messages[2] == {
        "role": "assistant",
        "content": "Hello, how can I help?",
    }
    assert second_call_messages[3] == {"role": "user", "content": "What did I say?"}


def test_two_sessions_stay_separate(client, fake_completion):
    fake = fake_completion(
        make_response("Noted, your name is Ada."),
        make_response("I do not know your name."),
    )

    client.post("/chat", json={"session_id": "s-one", "message": "My name is Ada"})
    client.post("/chat", json={"session_id": "s-two", "message": "What is my name?"})

    # The second session carries none of the first session's history.
    second_call_messages = fake.messages_of_call(1)
    assert len(second_call_messages) == 2
    assert second_call_messages[1] == {"role": "user", "content": "What is my name?"}

    first_history = client.get("/sessions/s-one/messages").json()["messages"]
    second_history = client.get("/sessions/s-two/messages").json()["messages"]
    assert len(first_history) == 2
    assert len(second_history) == 2
    assert first_history[0]["content"] == "My name is Ada"
    assert second_history[0]["content"] == "What is my name?"


def test_tokens_are_per_request_and_never_a_running_total(client, fake_completion):
    fake_completion(
        make_response("First answer.", prompt_tokens=10, completion_tokens=3),
        make_response("Second answer.", prompt_tokens=25, completion_tokens=7),
    )

    first = client.post(
        "/chat", json={"session_id": "s-tokens", "message": "First question"}
    ).json()
    second = client.post(
        "/chat", json={"session_id": "s-tokens", "message": "Second question"}
    ).json()

    assert first["metadata"]["input_tokens"] == 10
    assert first["metadata"]["output_tokens"] == 3
    # The second turn reports its own usage, not 10 + 25 or 3 + 7.
    assert second["metadata"]["input_tokens"] == 25
    assert second["metadata"]["output_tokens"] == 7


def test_the_response_never_contains_a_real_api_key(client, fake_completion):
    fake_completion()

    response = client.post(
        "/chat", json={"session_id": "s-secret", "message": "Keep secrets please"}
    )

    assert "test-key-one" not in response.text
    assert "test-key-two" not in response.text


def test_the_model_is_reported_with_its_provider_prefix(client, fake_completion):
    """Providers report a bare model name; the contract shows the LiteLLM id.

    Gemini answers with "gemini-3.1-flash-lite" in response.model, but the API
    contract (and the assignment sample) shows "gemini/gemini-2.0-flash", so the
    configured provider prefix is put back before the name is returned.
    """
    fake_completion(make_response(text="Paris", model="gemini-3.1-flash-lite"))

    response = client.post(
        "/chat", json={"session_id": "s-prefix", "message": "Capital of France?"}
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["model"] == "gemini/gemini-3.1-flash-lite"
