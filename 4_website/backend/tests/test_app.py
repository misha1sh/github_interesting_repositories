import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import engine as eng_module

@pytest.fixture
def client():
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealthEndpoint:
    def test_health_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["ok"] is True


class TestStatusEndpoint:
    def test_status_returns_loaded_field(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.get_json()
        assert "loaded" in data
        assert "repo_count" in data
        assert "tag_count" in data


class TestRecommendValidation:
    def test_missing_username_returns_400(self, client):
        r = client.post("/api/recommend", json={"top_k": 10})
        assert r.status_code == 400
        data = r.get_json()
        assert "errors" in data
        assert any("username" in e for e in data["errors"])

    def test_empty_username_returns_400(self, client):
        r = client.post("/api/recommend", json={"username": "", "top_k": 10})
        assert r.status_code == 400

    def test_invalid_top_k_returns_400(self, client):
        r = client.post("/api/recommend", json={"username": "test", "top_k": -5})
        assert r.status_code == 400
        data = r.get_json()
        assert any("top_k" in e for e in data["errors"])

    def test_invalid_model_returns_400(self, client):
        r = client.post("/api/recommend", json={"username": "test", "model": "gpt4"})
        assert r.status_code == 400
        data = r.get_json()
        assert any("model" in e for e in data["errors"])

    def test_device_param_ignored(self, client):
        # device is now auto-detected, not a user param — passing it should not cause an error
        r = client.post("/api/recommend", json={"username": "test", "device": "tpu"})
        # should not be a 400 from device validation; might still be 200 (streaming)
        assert r.status_code != 400 or "device" not in (r.get_json() or {}).get("errors", [""])[0]

    def test_invalid_clustering_type_returns_400(self, client):
        r = client.post("/api/recommend", json={"username": "test", "clustering_type": "spectral"})
        assert r.status_code == 400

    def test_negative_clusters_returns_400(self, client):
        r = client.post("/api/recommend", json={"username": "test", "clusters": -1})
        assert r.status_code == 400

    def test_no_body_returns_400(self, client):
        r = client.post("/api/recommend", content_type="application/json", data="")
        assert r.status_code == 400

    def test_multiple_errors_all_reported(self, client):
        r = client.post("/api/recommend", json={"username": "", "top_k": 0, "model": "bad"})
        assert r.status_code == 400
        data = r.get_json()
        assert len(data["errors"]) >= 2


class TestRecommendSSE:
    def _mock_engine_run(self, results=None):
        if results is None:
            results = [
                {
                    "rank": 1,
                    "name": "owner/repo",
                    "score": 0.95,
                    "calibrated": 0.8,
                    "pct_rank": 0.99,
                    "matching_tags": ["python", "ml"],
                    "cluster_id": 1,
                }
            ]
        return results

    def _parse_sse(self, raw_bytes: bytes) -> list[dict]:
        text = raw_bytes.decode("utf-8")
        events = []
        current_event = None
        for line in text.splitlines():
            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
                events.append({"event": current_event, "data": data})
                current_event = None
        return events

    def test_valid_request_returns_sse_stream(self, client):
        with patch.object(eng_module._static, "loaded", True), \
             patch("app.RecommendationEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.run.return_value = self._mock_engine_run()
            r = client.post(
                "/api/recommend",
                json={"username": "testuser", "top_k": 5},
            )
            assert r.status_code == 200
            assert "text/event-stream" in r.content_type

    def test_sse_stream_contains_done_event(self, client):
        with patch.object(eng_module._static, "loaded", True), \
             patch("app.RecommendationEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.run.return_value = self._mock_engine_run()
            r = client.post(
                "/api/recommend",
                json={"username": "testuser", "top_k": 5},
            )
            raw = b"".join(r.response)
            events = self._parse_sse(raw)
            event_types = [e["event"] for e in events]
            assert "done" in event_types

    def test_sse_done_event_has_success_flag(self, client):
        with patch.object(eng_module._static, "loaded", True), \
             patch("app.RecommendationEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.run.return_value = self._mock_engine_run()
            r = client.post(
                "/api/recommend",
                json={"username": "testuser", "top_k": 5},
            )
            raw = b"".join(r.response)
            events = self._parse_sse(raw)
            done_events = [e for e in events if e["event"] == "done"]
            assert len(done_events) == 1
            assert "success" in done_events[0]["data"]

    def test_engine_exception_yields_error_event(self, client):
        with patch.object(eng_module._static, "loaded", True), \
             patch("app.RecommendationEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.run.side_effect = RuntimeError("exploded")
            r = client.post(
                "/api/recommend",
                json={"username": "testuser", "top_k": 5},
            )
            raw = b"".join(r.response)
            events = self._parse_sse(raw)
            event_types = [e["event"] for e in events]
            assert "error" in event_types or "done" in event_types

    def test_default_params_used_when_omitted(self, client):
        with patch.object(eng_module._static, "loaded", True), \
             patch("app.RecommendationEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.run.return_value = []
            client.post("/api/recommend", json={"username": "testuser"})
            call_kwargs = instance.run.call_args[1]
            assert call_kwargs["top_k"] == 30
            assert call_kwargs["model_type"] == "nn"
            assert "device" not in call_kwargs
            assert call_kwargs["clusters"] == 0
            assert call_kwargs["clustering_type"] == "k-means"

    def test_custom_params_passed_to_engine(self, client):
        with patch.object(eng_module._static, "loaded", True), \
             patch("app.RecommendationEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.run.return_value = []
            client.post("/api/recommend", json={
                "username": "alice",
                "top_k": 50,
                "model": "simple",
                "include_forked": True,
                "include_user_repos": True,
                "clusters": 3,
                "clustering_type": "gmm",
            })
            call_kwargs = instance.run.call_args[1]
            assert call_kwargs["top_k"] == 50
            assert call_kwargs["model_type"] == "simple"
            assert call_kwargs["include_forked"] is True
            assert call_kwargs["include_user_repos"] is True
            assert call_kwargs["clusters"] == 3
            assert call_kwargs["clustering_type"] == "gmm"

    def test_sse_format_correct(self, client):
        with patch.object(eng_module._static, "loaded", True), \
             patch("app.RecommendationEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.run.return_value = []
            r = client.post("/api/recommend", json={"username": "testuser"})
            raw = b"".join(r.response)
            text = raw.decode("utf-8")
            for block in text.strip().split("\n\n"):
                if not block.strip():
                    continue
                lines = block.strip().splitlines()
                assert any(l.startswith("event:") for l in lines)
                assert any(l.startswith("data:") for l in lines)
