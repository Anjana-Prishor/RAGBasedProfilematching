# RAG-Based Profile Matching

An end-to-end Retrieval-Augmented Generation system for processing resumes and matching candidates to job descriptions.

## System Overview

| Requirement | Included |
|---|---|
| Complete RAG implementation | `resume_rag.py`, `job_matcher.py` |
| 30+ diverse resumes | 33 files in `sample_resumes/` |
| 5+ job descriptions | 5 records in `job_descriptions.json` |
| Jupyter experimentation and analysis | `RAG_Profile_Matching_Analysis.ipynb` |
| Retrieval accuracy and latency | `benchmark_metrics.py` |
| Demonstration guide | `VIDEO_DEMO_SCRIPT.md` |

## Features

### Part A: Resume RAG Pipeline

- Loads PDF, DOCX, and TXT resumes.
- Extracts candidate name, skills, experience, education, and source path.
- Splits resumes into section-aware chunks with configurable overlap.
- Generates 384-dimensional embeddings with `all-MiniLM-L6-v2`.
- Stores embeddings and metadata in persistent ChromaDB.
- Supports batch processing and metadata-filtered retrieval.

### Part B: Job Matching Engine

- Extracts structured requirements from job descriptions.
- Combines semantic and keyword relevance with configurable weights.
- Filters candidates by required skills, must-have requirements, and experience level.
- Ranks candidates with scores from 0 to 100.
- Produces matched skills, missing skills, section contributions, excerpts, and reasoning.
- Exports JSON-ready match reports.

## Architecture

```text
Resume files (PDF/DOCX/TXT)
        |
        v
Document loader -> metadata extractor -> section-aware chunker
        |
        v
Sentence Transformers embeddings -> ChromaDB vector store
        |
        v
Job description -> requirement extraction -> hybrid retrieval and ranking
        |
        v
Candidate matches, filters, explanations, and JSON reports
```

## Project Structure

```text
RAGBasedProfilematching/
|-- resume_rag.py                         # Production Part A pipeline
|-- job_matcher.py                        # Part B matching engine
|-- demo_part_a.py                        # Dependency-light Part A demo
|-- demo_complete.py                      # Complete Part A + Part B demo
|-- generate_dataset.py                   # Reproducible resume generator
|-- benchmark_metrics.py                  # Accuracy and latency benchmark
|-- test_part_a.py                        # Production Part A tests
|-- test_part_b.py                        # Part B tests
|-- job_descriptions.json                 # Five structured jobs
|-- sample_resumes/                       # 33 diverse resume files
|-- RAG_Profile_Matching_Analysis.ipynb   # Integrated analysis notebook
|-- VIDEO_DEMO_SCRIPT.md                  # 3-4 minute recording script
|-- config.json                           # System configuration
|-- requirements.txt                      # Python dependencies
```

## Installation

Use Python 3.10 or later where possible.

```powershell
cd RAGBasedProfilematching
pip install -r requirements.txt
```

Verify the production dependencies:

```powershell
python -c "import chromadb, sentence_transformers; print('Production dependencies installed')"
```

The dependency-light demos use only the project's local Python modules and are useful for a quick demonstration.

## Run the Complete System

Run one file to execute the full Part A and Part B workflow:

```powershell
python demo_complete.py
```

This single command processes the resumes, extracts metadata, creates chunks, matches candidates to jobs, shows explanations, prints JSON reports, demonstrates filtering, and creates `output.txt` with ready-to-use silent recording text.

No separate Part A or Part B demo command is required.

### Production RAG pipeline

```powershell
python resume_rag.py
```

The production pipeline uses Sentence Transformers for embeddings and ChromaDB for persistent vector storage. The first run may download the embedding model.

## Dataset

The dataset contains 33 resumes covering machine learning, data engineering, frontend, backend, cloud, security, QA, DevOps, mobile, analytics, product, and research roles.

Five job descriptions are provided for:

1. Machine Learning Engineer
2. Data Engineer
3. Cloud Platform Engineer
4. Full Stack Developer
5. Cloud Security Engineer

Regenerate the additional profiles with:

```powershell
python generate_dataset.py
```

## Notebook Analysis

Open the integrated analysis notebook in VS Code or Jupyter:

```powershell
jupyter notebook RAG_Profile_Matching_Analysis.ipynb
```

The notebook covers:

- Resume corpus statistics
- Job requirement extraction
- Candidate ranking across all five jobs
- Explainability output
- Retrieval accuracy and latency analysis

## Performance Metrics

Run the reproducible benchmark:

```powershell
python benchmark_metrics.py
```

The benchmark uses five manually labeled skill queries and `top_k=5`.

| Metric | Result |
|---|---:|
| Dataset size | 33 resumes |
| Queries | 5 |
| Retrieval accuracy at K | 1.00 |
| Mean reciprocal rank | 0.75 |
| Mean query latency | approximately 0.034 ms |
| P95 query latency | approximately 0.034 ms |

**Accuracy definition:** a query is counted as accurate when at least one manually labeled relevant candidate appears in the top-k results.

These measurements describe the deterministic lightweight retrieval benchmark. Production embedding and ChromaDB timings depend on hardware, model loading, and database state.

## Testing

Run the Part B tests and syntax checks:

```powershell
python test_part_b.py
python -m py_compile generate_dataset.py benchmark_metrics.py test_part_b.py job_matcher.py demo_complete.py
```

The production Part A tests require the packages in `requirements.txt`:

```powershell
python test_part_a.py
```

## Configuration

The main settings are in `config.json`:

- Embedding model: `all-MiniLM-L6-v2`
- Chunk size: 300
- Chunk overlap: 50
- ChromaDB path: `./chroma_db`
- Hybrid weights: semantic `0.6`, keyword `0.4`

## Video Recording

Follow [VIDEO_DEMO_SCRIPT.md](VIDEO_DEMO_SCRIPT.md) for a 3-4 minute walkthrough. The recording should show the project structure, Part A processing, Part B matching, explanations, filtering, and final results.

## Final Status

The project includes the complete RAG implementation, 33-resume dataset, five-job dataset, two analysis notebooks, reproducible performance metrics, tests, documentation, and an end-to-end demonstration.
