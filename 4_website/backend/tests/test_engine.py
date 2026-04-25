import sys
import os
import re
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine import (
    validate_params,
    _normalize_tags,
    _choose_k,
    _cluster_starred_embeddings,
    detect_device,
    VALID_MODELS,
    VALID_CLUSTERING,
)


class TestValidateParams:
    def _base(self, **overrides):
        p = {
            "username": "testuser",
            "top_k": 30,
            "model": "nn",
            "include_forked": False,
            "include_user_repos": False,
            "clusters": 0,
            "clustering_type": "k-means",
        }
        p.update(overrides)
        return p

    def test_valid_params_no_errors(self):
        assert validate_params(self._base()) == []

    def test_missing_username(self):
        errors = validate_params(self._base(username=""))
        assert any("username" in e for e in errors)

    def test_invalid_username_special_chars(self):
        errors = validate_params(self._base(username="user name!"))
        assert any("username" in e for e in errors)

    def test_valid_username_with_hyphens(self):
        assert validate_params(self._base(username="my-user.name_1")) == []

    def test_top_k_too_small(self):
        errors = validate_params(self._base(top_k=0))
        assert any("top_k" in e for e in errors)

    def test_top_k_too_large(self):
        errors = validate_params(self._base(top_k=201))
        assert any("top_k" in e for e in errors)

    def test_top_k_edge_cases(self):
        assert validate_params(self._base(top_k=1)) == []
        assert validate_params(self._base(top_k=200)) == []

    def test_invalid_model(self):
        errors = validate_params(self._base(model="transformer"))
        assert any("model" in e for e in errors)

    def test_valid_models(self):
        for m in VALID_MODELS:
            assert validate_params(self._base(model=m)) == []

    def test_detect_device_returns_string(self):
        d = detect_device()
        assert d in ("cpu", "cuda")

    def test_negative_clusters(self):
        errors = validate_params(self._base(clusters=-1))
        assert any("clusters" in e for e in errors)

    def test_zero_clusters_valid(self):
        assert validate_params(self._base(clusters=0)) == []

    def test_invalid_clustering_type(self):
        errors = validate_params(self._base(clustering_type="spectral"))
        assert any("clustering_type" in e for e in errors)

    def test_valid_clustering_types(self):
        for ct in VALID_CLUSTERING:
            assert validate_params(self._base(clustering_type=ct)) == []

    def test_multiple_errors(self):
        errors = validate_params({"username": "", "top_k": -1})
        assert len(errors) >= 2


class TestNormalizeTags:
    def test_basic_normalization(self):
        result = _normalize_tags(["Python", "Web-Dev", "API"])
        assert result == ["python", "web-dev", "api"]

    def test_strips_special_chars(self):
        result = _normalize_tags(["  hello world  "])
        assert "hello-world" in result

    def test_deduplication(self):
        result = _normalize_tags(["python", "python", "python"])
        assert result == ["python"]

    def test_max_7_tags(self):
        result = _normalize_tags([str(i) for i in range(10)])
        assert len(result) == 7

    def test_empty_list(self):
        assert _normalize_tags([]) == []

    def test_non_list_returns_empty(self):
        assert _normalize_tags(None) == []
        assert _normalize_tags("python") == []

    def test_non_string_items_skipped(self):
        result = _normalize_tags([1, "python", None, "js"])
        assert result == ["python", "js"]

    def test_empty_tag_skipped(self):
        result = _normalize_tags(["", "python"])
        assert result == ["python"]

    def test_plus_sign_preserved(self):
        result = _normalize_tags(["c++"])
        assert "c++" in result


class TestChooseK:
    def test_empty_returns_1(self):
        import numpy as np
        assert _choose_k(np.array([]).reshape(0, 4)) == 1

    def test_few_samples_returns_1(self):
        import numpy as np
        rng = np.random.default_rng(42)
        emb = rng.random((3, 8))
        assert _choose_k(emb) == 1

    def test_returns_at_least_1(self):
        import numpy as np
        rng = np.random.default_rng(42)
        emb = rng.random((20, 8))
        k = _choose_k(emb)
        assert k >= 1

    def test_respects_max_k(self):
        import numpy as np
        rng = np.random.default_rng(42)
        emb = rng.random((50, 8))
        k = _choose_k(emb, max_k=3)
        assert k <= 3


class TestClusterEmbeddings:
    def test_single_sample(self):
        import numpy as np
        emb = np.random.rand(1, 8)
        labels = _cluster_starred_embeddings(emb, "k-means", 1)
        assert labels == [0]

    def test_k1_returns_all_zeros(self):
        import numpy as np
        emb = np.random.rand(10, 8)
        labels = _cluster_starred_embeddings(emb, "k-means", 1)
        assert all(l == 0 for l in labels)

    def test_kmeans_returns_correct_length(self):
        import numpy as np
        rng = np.random.default_rng(42)
        emb = rng.random((20, 8))
        labels = _cluster_starred_embeddings(emb, "k-means", 3)
        assert len(labels) == 20

    def test_agglomerative_returns_correct_length(self):
        import numpy as np
        rng = np.random.default_rng(42)
        emb = rng.random((20, 8))
        labels = _cluster_starred_embeddings(emb, "agglomerative", 2)
        assert len(labels) == 20

    def test_dbscan_returns_correct_length(self):
        import numpy as np
        rng = np.random.default_rng(42)
        emb = rng.random((20, 8))
        labels = _cluster_starred_embeddings(emb, "dbscan", 3)
        assert len(labels) == 20

    def test_unknown_type_falls_back_to_kmeans(self):
        import numpy as np
        rng = np.random.default_rng(42)
        emb = rng.random((20, 8))
        labels = _cluster_starred_embeddings(emb, "unknown", 2)
        assert len(labels) == 20
