"""Unit tests for the multi-signal rerank module."""
from __future__ import annotations

import sys
import os
import pytest

# Allow imports from the parent package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nodes.rerank import rerank_frames, _expected_modalities


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_candidate(
    video_id: str,
    frame_id: int,
    fused_score: float,
    evidence: list[str],
    source_scores: dict[str, float] | None = None,
) -> dict:
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "fused_score": fused_score,
        "evidence": evidence,
        "source_scores": source_scores or {src: 0.8 for src in evidence},
        "branch": "agentic",
        "raw_payload": {},
    }


ROUTING_WEIGHTS_MIXED = {
    "keyframe": 0.25,
    "caption": 0.40,
    "object": 0.25,
    "ocr": 0.10,
    "metadata": 0.00,
}

INTENT_VISUAL_OBJECT = {
    "objects": ["car", "tree"],
    "attributes": ["red"],
    "actions": [],
    "scene": [],
    "text_cues": [],
    "metadata_cues": [],
    "query_type": "visual_object",
}

INTENT_TEXT_AND_OBJECT = {
    "objects": ["person"],
    "attributes": [],
    "actions": [],
    "scene": [],
    "text_cues": ["HELLO"],
    "metadata_cues": [],
    "query_type": "mixed",
}

INTENT_EVENT_SCENE = {
    "objects": [],
    "attributes": [],
    "actions": ["running"],
    "scene": ["outdoor park"],
    "text_cues": [],
    "metadata_cues": [],
    "query_type": "visual_event",
}

INTENT_EMPTY = {
    "objects": [],
    "attributes": [],
    "actions": [],
    "scene": [],
    "text_cues": [],
    "metadata_cues": [],
    "query_type": "mixed",
}


# ── Tests for _expected_modalities ────────────────────────────────────────────

class TestExpectedModalities:
    def test_objects_expect_object_source(self):
        result = _expected_modalities(INTENT_VISUAL_OBJECT)
        assert "object" in result

    def test_text_cues_expect_ocr(self):
        result = _expected_modalities(INTENT_TEXT_AND_OBJECT)
        assert "ocr" in result
        assert "object" in result

    def test_actions_expect_caption(self):
        result = _expected_modalities(INTENT_EVENT_SCENE)
        assert "caption" in result

    def test_empty_intent_expects_nothing(self):
        result = _expected_modalities(INTENT_EMPTY)
        assert len(result) == 0


# ── Tests for cross-source agreement (Tầng 1) ────────────────────────────────

class TestCrossSourceAgreement:
    def test_multi_source_beats_single_source(self):
        """A candidate with evidence from multiple strong sources should score
        higher than one from a single source, all else equal."""
        multi = _make_candidate("v1", 1, fused_score=0.5, evidence=["caption", "keyframe", "object"])
        single = _make_candidate("v1", 2, fused_score=0.5, evidence=["caption"])

        results = rerank_frames(
            [multi, single],
            query_intent=INTENT_EMPTY,
            routing_weights=ROUTING_WEIGHTS_MIXED,
        )

        assert results[0]["frame_id"] == 1, "Multi-source candidate should rank first"
        assert results[0]["agent_score"] > results[1]["agent_score"]

    def test_high_weight_source_contributes_more(self):
        """A candidate from a high-weight source should get a bigger agreement
        bonus than one from a low-weight source."""
        # caption weight = 0.40, ocr weight = 0.10
        high_weight = _make_candidate("v1", 1, fused_score=0.5, evidence=["caption"],
                                      source_scores={"caption": 1.0})
        low_weight = _make_candidate("v1", 2, fused_score=0.5, evidence=["ocr"],
                                     source_scores={"ocr": 1.0})

        results = rerank_frames(
            [high_weight, low_weight],
            query_intent=INTENT_EMPTY,
            routing_weights=ROUTING_WEIGHTS_MIXED,
        )

        assert results[0]["frame_id"] == 1
        # Verify the agreement bonus difference
        sig_1 = results[0]["rerank_signals"]["agreement_bonus"]
        sig_2 = results[1]["rerank_signals"]["agreement_bonus"]
        assert sig_1 > sig_2


# ── Tests for intent coverage (Tầng 2) ───────────────────────────────────────

class TestIntentCoverage:
    def test_coverage_bonus_when_expected_present(self):
        """Frame with object evidence should get coverage bonus when intent has objects."""
        candidate = _make_candidate("v1", 1, fused_score=0.5, evidence=["object", "caption"])

        results = rerank_frames(
            [candidate],
            query_intent=INTENT_VISUAL_OBJECT,
            routing_weights=ROUTING_WEIGHTS_MIXED,
        )

        signals = results[0]["rerank_signals"]
        assert signals["coverage_bonus"] > 0
        assert "object" in signals["present_modalities"]

    def test_missing_penalty_when_expected_absent(self):
        """Frame WITHOUT ocr evidence should be penalized when intent has text_cues."""
        candidate = _make_candidate("v1", 1, fused_score=0.5, evidence=["caption"])

        results = rerank_frames(
            [candidate],
            query_intent=INTENT_TEXT_AND_OBJECT,
            routing_weights=ROUTING_WEIGHTS_MIXED,
        )

        signals = results[0]["rerank_signals"]
        assert signals["missing_penalty"] > 0
        assert "ocr" in signals["missing_modalities"]

    def test_full_coverage_no_penalty(self):
        """Frame covering all expected modalities should have zero penalty."""
        candidate = _make_candidate("v1", 1, fused_score=0.5,
                                    evidence=["ocr", "object", "caption"])

        results = rerank_frames(
            [candidate],
            query_intent=INTENT_TEXT_AND_OBJECT,
            routing_weights=ROUTING_WEIGHTS_MIXED,
        )

        signals = results[0]["rerank_signals"]
        assert signals["missing_penalty"] == 0
        assert len(signals["missing_modalities"]) == 0

    def test_coverage_helps_rank_order(self):
        """Given same fused_score, the candidate covering expected modalities
        should rank higher."""
        covered = _make_candidate("v1", 1, fused_score=0.5,
                                  evidence=["object", "caption"])
        not_covered = _make_candidate("v1", 2, fused_score=0.5,
                                      evidence=["caption"])

        results = rerank_frames(
            [not_covered, covered],
            query_intent=INTENT_VISUAL_OBJECT,
            routing_weights=ROUTING_WEIGHTS_MIXED,
        )

        assert results[0]["frame_id"] == 1


# ── Tests for signal breakdown and sorting ────────────────────────────────────

class TestSignalBreakdown:
    def test_signal_keys_present(self):
        candidate = _make_candidate("v1", 1, fused_score=0.5, evidence=["caption"])
        results = rerank_frames(
            [candidate],
            query_intent=INTENT_EMPTY,
            routing_weights=ROUTING_WEIGHTS_MIXED,
        )
        signals = results[0]["rerank_signals"]
        assert "fused_score" in signals
        assert "agreement_bonus" in signals
        assert "coverage_bonus" in signals
        assert "missing_penalty" in signals
        assert "expected_modalities" in signals
        assert "present_modalities" in signals
        assert "missing_modalities" in signals

    def test_sorted_descending(self):
        c1 = _make_candidate("v1", 1, fused_score=0.9, evidence=["caption"])
        c2 = _make_candidate("v1", 2, fused_score=0.3, evidence=["caption"])
        c3 = _make_candidate("v1", 3, fused_score=0.6, evidence=["caption"])

        results = rerank_frames(
            [c1, c2, c3],
            query_intent=INTENT_EMPTY,
            routing_weights=ROUTING_WEIGHTS_MIXED,
        )

        scores = [r["agent_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_input_limits_shortlist(self):
        candidates = [
            _make_candidate("v1", i, fused_score=1.0 - i * 0.01, evidence=["caption"])
            for i in range(50)
        ]

        results = rerank_frames(
            candidates,
            query_intent=INTENT_EMPTY,
            routing_weights=ROUTING_WEIGHTS_MIXED,
            top_n_input=10,
        )

        assert len(results) == 10


# ── Test for empty intent (no expected modalities → no bonus/penalty) ────────

class TestEmptyIntent:
    def test_no_coverage_signals_on_empty_intent(self):
        candidate = _make_candidate("v1", 1, fused_score=0.5,
                                    evidence=["caption", "object", "ocr"])

        results = rerank_frames(
            [candidate],
            query_intent=INTENT_EMPTY,
            routing_weights=ROUTING_WEIGHTS_MIXED,
        )

        signals = results[0]["rerank_signals"]
        assert signals["coverage_bonus"] == 0
        assert signals["missing_penalty"] == 0
        assert len(signals["expected_modalities"]) == 0
