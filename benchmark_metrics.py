"""Reproducible retrieval accuracy and latency benchmark."""

import json
import statistics
import time
from pathlib import Path

from demo_part_a import SimplifiedRAGSystem
from generate_dataset import write_resumes
from job_matcher import JobDescription, JobMatcher, SearchConfig


QUERIES = [
    ("machine learning python pytorch", {"Aisha Rahman", "Nora Singh", "Quinn Adams", "Xavier Laurent", "Diego Alvarez", "JOHN DOE"}),
    ("data engineering sql spark airflow", {"Ben Carter", "Yara Hassan", "JANE SMITH", "Uma Nair"}),
    ("react node typescript postgresql", {"Chloe Nguyen", "Lina Petrova", "Victor Chen", "ALICE KUMAR"}),
    ("aws kubernetes terraform linux", {"Elena Rossi", "Ibrahim Diallo", "Ravi Patel", "Wendy Brown", "Diego Alvarez"}),
    ("cloud security aws python siem", {"George Wilson", "Amara Williams"}),
]


def build_candidates():
    write_resumes()
    rag = SimplifiedRAGSystem(chunk_size=300, chunk_overlap=50)
    processed = rag.process_resumes_from_directory("sample_resumes")
    candidates = []
    for resume in processed["resumes"]:
        chunks = [chunk for chunk in rag.db.documents if Path(chunk["metadata"]["file_path"]).name == Path(resume["file"]).name]
        candidates.append({
            "name": resume["name"],
            "skills": resume["skills"],
            "experience_years": resume["experience_years"],
            "file_path": resume["file"],
            "chunks": chunks,
            "full_text": "\n".join(chunk["content"] for chunk in chunks),
        })
    return candidates


def run_benchmark(top_k=5):
    candidates = build_candidates()
    latencies = []
    hits = 0
    reciprocal_ranks = []
    matcher = JobMatcher(SearchConfig(top_k=top_k, min_score_threshold=0.0, require_all_must_haves=False))

    for query, relevant_names in QUERIES:
        start = time.perf_counter()
        query_terms = set(query.lower().split())
        scored = []
        for candidate in candidates:
            candidate_terms = set(" ".join(candidate["skills"]).lower().replace("-", " ").split())
            overlap = len(query_terms & candidate_terms) / len(query_terms)
            scored.append((overlap, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        retrieved = [candidate["name"] for _, candidate in scored[:top_k]]
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        rank = next((index + 1 for index, name in enumerate(retrieved) if name in relevant_names), None)
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    metrics = {
        "dataset_resumes": len(candidates),
        "queries": len(QUERIES),
        "top_k": top_k,
        "retrieval_accuracy_at_k": hits / len(QUERIES),
        "mean_reciprocal_rank": statistics.mean(reciprocal_ranks),
        "mean_latency_ms": statistics.mean(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        "measurement": "A query is accurate when at least one manually labeled relevant candidate appears in the top-k results.",
    }
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
