"""
Part B Demo - Job Matching Engine
Demonstrates complete RAG system with job matching

Combines Part A (RAG Setup) with Part B (Job Matching)
"""

import sys
import os
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo_part_a import SimplifiedRAGSystem, create_sample_resumes
from job_matcher import (
    JobMatcher,
    JobDescriptionExtractor,
    SearchConfig,
    create_sample_job_descriptions
)
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def write_recording_script(results, rag_system, sample_jobs, job_results):
    """Write concise on-screen narration prompts for a silent recording."""
    total_matches = sum(len(result['matches']) for result in job_results.values())
    lines = [
        "RAG-BASED PROFILE MATCHING - RECORDING SCRIPT",
        "",
        "[INTRODUCTION]",
        "This system processes resumes and matches candidates to job descriptions using a complete RAG workflow.",
        "",
        "[PART A: RESUME PROCESSING]",
        f"The system successfully processed {results['processed']} resumes and created {len(rag_system.db.documents)} document chunks.",
        "It extracts candidate names, skills, years of experience, education, and resume sections.",
        "The section-aware chunks preserve context for accurate retrieval and explanations.",
        "",
        "[PART B: JOB MATCHING]",
        f"The matching engine processed {len(sample_jobs)} job descriptions.",
        "It combines semantic relevance and keyword matching with configurable weights.",
        "It also applies skill filters, must-have requirements, and experience-level matching.",
        "",
        "[MATCH RESULTS]",
        "Each result includes an overall score, semantic score, keyword score, matched skills, missing skills, and experience compatibility.",
        "The section contribution analysis shows which parts of each resume influenced the match.",
        "The reasoning and relevant excerpts make the ranking explainable.",
        "",
        "[FILTERING]",
        "The system supports recruiter-style filtering by required skills and minimum experience.",
        "This helps narrow the candidate pool before reviewing ranked results.",
        "",
        "[JSON OUTPUT]",
        "The system also generates structured JSON reports suitable for APIs or downstream storage.",
        "",
        "[CONCLUSION]",
        f"The complete workflow generated {total_matches} candidate-job matches.",
        "This demonstrates resume ingestion, retrieval, ranking, filtering, and explainable job matching in one system.",
    ]
    Path("output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_match_result(match, index: int):
    """Format a single match result for display"""
    print(f"\n  {index}. {match.candidate_name}")
    print(f"     Match Score: {match.match_score:.1f}/100")
    print(f"     Semantic Score: {match.semantic_score:.1f}")
    print(f"     Keyword Score: {match.keyword_score:.1f}")
    print(f"     Resume: {match.resume_path}")
    
    print(f"     Matched Skills: {', '.join(match.matched_skills[:5])}")
    if match.missing_required_skills:
        print(f"     Missing: {', '.join(match.missing_required_skills)}")
    
    if match.nice_to_have_matched:
        print(f"     Nice-to-have: {', '.join(match.nice_to_have_matched)}")
    
    print(f"     Experience Match: {'YES' if match.experience_match else 'NO'}")
    
    if match.section_contributions:
        print(f"     Section Contributions:")
        for section, contrib in sorted(match.section_contributions.items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"       - {section}: {contrib:.1f}%")
    
    print(f"     Reasoning: {match.reasoning}")
    
    if match.relevant_excerpts:
        print(f"     Relevant Excerpts:")
        for excerpt in match.relevant_excerpts[:2]:
            print(f"       - \"{excerpt}\"")


def run_demo():
    """Run complete Part A + Part B demo"""
    print("\n" + "="*80)
    print("RAG-BASED PROFILE MATCHING - COMPLETE DEMO")
    print("Part A: RAG System Setup + Part B: Job Matching Engine")
    print("="*80)
    
    # ==================== PART A: Setup ====================
    print("\n" + "-"*80)
    print("PART A: RAG SYSTEM SETUP")
    print("-"*80)
    
    # Create sample resumes
    print("\n[Step 1] Creating sample resumes...")
    resume_dir = create_sample_resumes()
    print(f"[OK] Sample resumes created in: {resume_dir}")
    
    # Initialize RAG system
    print("\n[Step 2] Initializing RAG system...")
    rag_system = SimplifiedRAGSystem(chunk_size=300, chunk_overlap=50)
    print("[OK] RAG system initialized")
    
    # Process resumes
    print("\n[Step 3] Processing resumes...")
    results = rag_system.process_resumes_from_directory(resume_dir)
    
    print(f"\nProcessing Results:")
    print(f"  Total files: {results['total_files']}")
    print(f"  Successfully processed: {results['processed']}")
    print(f"  Failed: {results['failed']}")
    
    # Prepare candidate data for job matching
    candidates_data = []
    for resume in results['resumes']:
        # Find corresponding chunks
        chunks = [c for c in rag_system.db.documents 
                 if c['metadata']['file_path'].endswith(Path(resume['file']).name) 
                 or resume['name'] in c['metadata']['name']]
        
        candidates_data.append({
            'name': resume['name'],
            'skills': resume['skills'],
            'experience_years': resume['experience_years'],
            'education': resume['education'],
            'file_path': resume['file'],
            'chunks': chunks,
            'full_text': '\n'.join([c['content'] for c in chunks])
        })
    
    print(f"\n[Step 4] Candidates prepared for matching:")
    for cand in candidates_data:
        print(f"  - {cand['name']}: {cand['experience_years']} years, {len(cand['skills'])} skills")
    
    # ==================== PART B: Job Matching ====================
    print("\n" + "-"*80)
    print("PART B: JOB MATCHING ENGINE")
    print("-"*80)
    
    # Create sample jobs
    print("\n[Step 5] Creating sample job descriptions...")
    sample_jobs = create_sample_job_descriptions()
    print(f"[OK] Created {len(sample_jobs)} sample job descriptions")
    
    # Initialize job matcher
    print("\n[Step 6] Initializing job matcher...")
    config = SearchConfig(
        semantic_weight=0.6,
        keyword_weight=0.4,
        top_k=10,
        min_score_threshold=40.0,
        require_all_must_haves=True
    )
    job_matcher = JobMatcher(config=config)
    print("[OK] Job matcher initialized with hybrid search")
    
    # Process each job
    print("\n[Step 7] Processing job descriptions and matching candidates...")
    
    job_results = {}
    for job_key, job_text in sample_jobs.items():
        print(f"\n  Processing: {job_key.replace('_', ' ').title()}")
        
        # Extract job information
        job = job_matcher.process_job_description(job_text, job_key.replace('_', ' ').title())
        
        print(f"    Required Skills: {', '.join(job.required_skills[:5])}")
        print(f"    Required Experience: {job.required_experience}+ years")
        print(f"    Experience Level: {job.experience_level}")
        
        # Match candidates
        matches = job_matcher.match_job_to_candidates(job, candidates_data)
        job_results[job_key] = {
            'job': job,
            'matches': matches
        }
        
        print(f"    [OK] Found {len(matches)} matches")
    
    # ==================== RESULTS ====================
    print("\n" + "="*80)
    print("DETAILED MATCHING RESULTS")
    print("="*80)
    
    for job_key, result in job_results.items():
        job = result['job']
        matches = result['matches']
        
        print(f"\n{'='*80}")
        print(f"JOB: {job.title}")
        print(f"{'='*80}")
        
        print(f"\nJob Requirements:")
        print(f"  Required Skills: {', '.join(job.required_skills)}")
        print(f"  Experience: {job.required_experience}+ years ({job.experience_level} level)")
        print(f"  Nice-to-have: {', '.join(job.nice_to_have_skills[:5])}")
        
        if matches:
            print(f"\nTop Matches (found {len(matches)}):")
            for i, match in enumerate(matches, 1):
                format_match_result(match, i)
        else:
            print(f"\nNo suitable matches found for this position.")
    
    # ==================== JSON OUTPUT ====================
    print("\n" + "="*80)
    print("JSON OUTPUT (FOR API/STORAGE)")
    print("="*80)
    
    for job_key, result in job_results.items():
        job = result['job']
        matches = result['matches']
        
        report = job_matcher.generate_report(job, matches, include_reasoning=True)
        
        print(f"\n# Job: {job.title}\n")
        print(json.dumps(report, indent=2, default=str))
    
    # ==================== FILTERING EXAMPLES ====================
    print("\n" + "="*80)
    print("FILTERING EXAMPLES")
    print("="*80)
    
    # Example 1: Filter by specific skills
    print("\n[Example 1] ML Engineer role filtered by Python requirement:")
    ml_job = job_results['ml_engineer']['job']
    filtered_matches = job_matcher.match_job_to_candidates(
        ml_job, 
        candidates_data,
        filter_skills=['Python'],
        filter_experience_min=4.0
    )
    print(f"  Candidates with Python + 4+ years: {len(filtered_matches)}")
    for match in filtered_matches:
        print(f"    - {match.candidate_name}: {match.match_score:.1f}%")
    
    # Example 2: Filter by experience level
    print("\n[Example 2] Data Engineer role (mid-level, 5+ years):")
    data_job = job_results['data_engineer']['job']
    filtered_matches = job_matcher.match_job_to_candidates(
        data_job,
        candidates_data,
        filter_experience_min=5.0
    )
    print(f"  Candidates with 5+ years: {len(filtered_matches)}")
    for match in filtered_matches:
        print(f"    - {match.candidate_name}: {match.match_score:.1f}%")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n[OK] Part A: Processed {results['processed']} resumes")
    print(f"[OK] Part A: Created {len(rag_system.db.documents)} document chunks")
    print(f"[OK] Part A: Extracted metadata from all candidates")
    
    print(f"\n[OK] Part B: Processed {len(sample_jobs)} job descriptions")
    print(f"[OK] Part B: Implemented hybrid search (semantic + keyword)")
    print(f"[OK] Part B: Performed matching with multiple filters")
    print(f"[OK] Part B: Generated match reasoning and analysis")
    
    total_matches = sum(len(r['matches']) for r in job_results.values())
    print(f"\n[OK] Total matches generated: {total_matches}")
    
    print("\n" + "="*80)
    print("SUCCESS: COMPLETE RAG SYSTEM WITH JOB MATCHING - WORKING!")
    print("="*80 + "\n")
    write_recording_script(results, rag_system, sample_jobs, job_results)
    print("[OK] Silent recording script written to output.txt")


if __name__ == "__main__":
    run_demo()
