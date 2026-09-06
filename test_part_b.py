"""Focused tests for the Part B job matching engine."""

import json
from pathlib import Path

from benchmark_metrics import build_candidates
from job_matcher import JobDescription, JobMatcher, SearchConfig


def test_job_dataset():
    jobs = json.loads(Path("job_descriptions.json").read_text(encoding="utf-8"))
    assert len(jobs) >= 5
    assert all(job["required_skills"] for job in jobs)


def test_candidate_matching():
    candidates = build_candidates()
    assert len(candidates) >= 30
    job_data = json.loads(Path("job_descriptions.json").read_text(encoding="utf-8"))[0]
    job_data["experience_level"] = "senior"
    matcher = JobMatcher(SearchConfig(min_score_threshold=0.0, require_all_must_haves=False))
    job = JobDescription(**{key: value for key, value in job_data.items() if key != "id"})
    results = matcher.match_job_to_candidates(job, candidates)
    assert results
    assert results[0].match_score >= results[-1].match_score
    assert results[0].reasoning


if __name__ == "__main__":
    test_job_dataset()
    test_candidate_matching()
    print("Part B tests passed")
