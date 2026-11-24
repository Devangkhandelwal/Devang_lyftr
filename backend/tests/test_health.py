"""Tests for GET /health."""


def test_health_returns_the_three_required_fields(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["keys_configured"] == 2
    assert body["cache_enabled"] is True


def test_health_never_exposes_a_real_key(client):
    response = client.get("/health")

    raw_body = response.text
    assert "test-key-one" not in raw_body
    assert "test-key-two" not in raw_body
