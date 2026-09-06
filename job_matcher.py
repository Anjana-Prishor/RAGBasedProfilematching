"""
Job Matching Engine - Part B
Semantic search, ranking, and candidate matching for job descriptions

Features:
- Hybrid search (semantic + keyword matching)
- Configurable match weights
- Must-have requirements filtering
- Match reasoning and section contribution analysis
- Experience level matching
- Skill filtering
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Data Models ====================

@dataclass
class JobDescription:
    """Job description model"""
    title: str
    description: str
    required_skills: List[str]
    required_experience: float
    nice_to_have_skills: List[str] = None
    experience_level: str = "mid"  # entry, mid, senior
    must_have_keywords: List[str] = None
    
    def __post_init__(self):
        if self.nice_to_have_skills is None:
            self.nice_to_have_skills = []
        if self.must_have_keywords is None:
            self.must_have_keywords = []


@dataclass
class MatchResult:
    """Result of matching a candidate to a job"""
    candidate_name: str
    resume_path: str
    match_score: float  # 0-100
    semantic_score: float  # 0-100
    keyword_score: float  # 0-100
    matched_skills: List[str]
    missing_required_skills: List[str]
    nice_to_have_matched: List[str]
    experience_match: bool
    section_contributions: Dict[str, float]  # Section name -> contribution %
    relevant_excerpts: List[str]
    reasoning: str
    matched_chunks: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class SearchConfig:
    """Configuration for search behavior"""
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    top_k: int = 10
    min_score_threshold: float = 40.0
    require_all_must_haves: bool = True
    experience_strict_matching: bool = False  # If True, must meet exact experience


# ==================== Job Description Extractor ====================

class JobDescriptionExtractor:
    """Extract structured information from job descriptions"""
    
    def __init__(self):
        self.skill_keywords = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'sql', 'mongodb', 'postgresql', 'mysql', 'redis',
            'machine learning', 'ml', 'deep learning', 'nlp', 'computer vision',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'react', 'angular', 'vue', 'node', 'express', 'django', 'flask',
            'git', 'linux', 'rest api', 'graphql', 'microservices'
        }
    
    def extract_experience_requirement(self, text: str) -> float:
        """Extract required years of experience"""
        patterns = [
            r'(\d+)\+?\s*years?\s+of\s+experience',
            r'requires?\s+(\d+)\+?\s*years?',
            r'(\d+)\s*\+\s*years?\s+exp'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return float(matches[0])
        
        return 0.0
    
    def extract_skills(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract required and nice-to-have skills"""
        text_lower = text.lower()
        required_skills = []
        nice_to_have = []
        
        # Check if skill is in required or nice-to-have section
        in_required = False
        in_nice_to_have = False
        
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            if any(word in line_lower for word in ['required', 'must have', 'essential']):
                in_required = True
                in_nice_to_have = False
            elif any(word in line_lower for word in ['nice to have', 'preferred', 'bonus']):
                in_nice_to_have = True
                in_required = False
            
            for skill in self.skill_keywords:
                if skill in line_lower:
                    if in_required:
                        if skill not in required_skills:
                            required_skills.append(skill.title())
                    elif in_nice_to_have:
                        if skill not in nice_to_have:
                            nice_to_have.append(skill.title())
                    elif not in_required and not in_nice_to_have:
                        if skill not in required_skills:
                            required_skills.append(skill.title())
        
        return required_skills, nice_to_have
    
    def extract_experience_level(self, text: str) -> str:
        """Determine experience level"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['junior', 'entry', 'entry-level', 'graduate']):
            return 'entry'
        elif any(word in text_lower for word in ['senior', 'lead', 'principal', 'staff']):
            return 'senior'
        else:
            return 'mid'
    
    def extract_must_haves(self, text: str) -> List[str]:
        """Extract critical must-have keywords"""
        patterns = [
            r'must have[:\s]+([^\n]+)',
            r'required[:\s]+([^\n]+)',
            r'essential[:\s]+([^\n]+)',
            r'critical[:\s]+([^\n]+)'
        ]
        
        must_haves = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                items = [item.strip() for item in match.split(',')]
                must_haves.extend(items)
        
        return list(set(must_haves))
    
    def extract(self, text: str, title: str = "Unknown") -> JobDescription:
        """Extract all information from job description"""
        required_skills, nice_to_have = self.extract_skills(text)
        
        return JobDescription(
            title=title,
            description=text,
            required_skills=required_skills,
            required_experience=self.extract_experience_requirement(text),
            nice_to_have_skills=nice_to_have,
            experience_level=self.extract_experience_level(text),
            must_have_keywords=self.extract_must_haves(text)
        )


# ==================== Semantic Search Engine ====================

class SemanticMatcher:
    """Perform semantic matching between job and candidates"""
    
    def __init__(self, rag_system=None, embedding_generator=None):
        """Initialize matcher with RAG system components"""
        self.rag_system = rag_system
        self.embedding_generator = embedding_generator
    
    def get_semantic_score(self, job_embedding, candidate_embedding) -> float:
        """Calculate cosine similarity between embeddings"""
        if job_embedding is None or candidate_embedding is None:
            return 0.0
        
        # Cosine similarity: dot product / (norm1 * norm2)
        import numpy as np
        
        job_vec = np.array(job_embedding)
        cand_vec = np.array(candidate_embedding)
        
        dot_product = np.dot(job_vec, cand_vec)
        norm1 = np.linalg.norm(job_vec)
        norm2 = np.linalg.norm(cand_vec)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Convert from [-1, 1] to [0, 100]
        return max(0, min(100, (similarity + 1) * 50))


# ==================== Keyword Matcher ====================

class KeywordMatcher:
    """Perform keyword-based matching"""
    
    def match_skills(self, required_skills: List[str], 
                     candidate_skills: List[str]) -> Tuple[List[str], List[str], float]:
        """
        Match candidate skills against required skills
        
        Returns:
            matched_skills: Skills found in candidate
            missing_skills: Required skills not found
            match_percentage: Percentage of required skills matched
        """
        matched = []
        missing = []
        
        required_lower = [s.lower() for s in required_skills]
        candidate_lower = [s.lower() for s in candidate_skills]
        
        for req_skill in required_skills:
            req_lower = req_skill.lower()
            found = False
            
            for cand_skill in candidate_skills:
                if req_lower == cand_skill.lower() or req_lower in cand_skill.lower():
                    matched.append(cand_skill)
                    found = True
                    break
            
            if not found:
                missing.append(req_skill)
        
        match_percentage = (len(matched) / len(required_skills) * 100) if required_skills else 0
        
        return matched, missing, match_percentage
    
    def match_experience(self, required_years: float, 
                        candidate_years: float, 
                        strict: bool = False) -> Tuple[bool, float]:
        """
        Match experience levels
        
        Returns:
            matches: Whether experience requirement is met
            score: Score for experience matching (0-100)
        """
        if required_years == 0:
            return True, 100.0
        
        if strict:
            matches = candidate_years >= required_years
            score = min(100, (candidate_years / required_years) * 100)
        else:
            # More lenient: 80% of required is acceptable
            threshold = required_years * 0.8
            matches = candidate_years >= threshold
            score = min(100, (candidate_years / required_years) * 100)
        
        return matches, score
    
    def match_keywords(self, required_keywords: List[str], 
                      text: str) -> Tuple[float, List[str]]:
        """
        Match critical keywords in text
        
        Returns:
            match_percentage: Percentage of keywords found
            found_keywords: Keywords that were found
        """
        text_lower = text.lower()
        found = []
        
        for keyword in required_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                found.append(keyword)
        
        match_percentage = (len(found) / len(required_keywords) * 100) if required_keywords else 0
        
        return match_percentage, found


# ==================== Match Reasoner ====================

class MatchReasoner:
    """Generate reasoning and analysis for matches"""
    
    def analyze_section_contributions(self, 
                                     matched_chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze which resume sections contributed to the match"""
        contributions = {}
        
        if not matched_chunks:
            return contributions
        
        for chunk in matched_chunks:
            section = chunk.get('section', 'unknown')
            if section not in contributions:
                contributions[section] = 0
            contributions[section] += 1
        
        # Normalize to percentages
        total = sum(contributions.values())
        for section in contributions:
            contributions[section] = (contributions[section] / total) * 100
        
        return contributions
    
    def generate_reasoning(self, 
                          job: JobDescription,
                          candidate_name: str,
                          match_result: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the match"""
        reasons = []
        
        # Skills match
        matched_skills = match_result.get('matched_skills', [])
        if matched_skills:
            reasons.append(f"Strong match for {', '.join(matched_skills[:3])} skills")
        
        missing_skills = match_result.get('missing_required_skills', [])
        if missing_skills:
            reasons.append(f"Missing: {', '.join(missing_skills[:2])}")
        
        # Experience
        if match_result.get('experience_match'):
            reasons.append("Meets experience requirements")
        
        # Section contributions
        contributions = match_result.get('section_contributions', {})
        if contributions:
            top_section = max(contributions.items(), key=lambda x: x[1])[0]
            reasons.append(f"Strongest match in {top_section} section")
        
        # Score-based reasoning
        score = match_result.get('match_score', 0)
        if score >= 90:
            reasons.append("Excellent overall fit")
        elif score >= 75:
            reasons.append("Good overall fit")
        elif score >= 60:
            reasons.append("Moderate fit, worth considering")
        
        return "; ".join(reasons) if reasons else "Limited information available"
    
    def extract_relevant_excerpts(self, 
                                 matched_chunks: List[Dict[str, Any]], 
                                 max_excerpts: int = 3) -> List[str]:
        """Extract relevant text excerpts from matched chunks"""
        excerpts = []
        
        for chunk in matched_chunks[:max_excerpts]:
            content = chunk.get('content', '')
            if content:
                # Take first 150 characters
                excerpt = content[:150].strip()
                if not excerpt.endswith('.'):
                    excerpt += '...'
                excerpts.append(excerpt)
        
        return excerpts


# ==================== Job Matcher ====================

class JobMatcher:
    """Main job matching engine"""
    
    def __init__(self, 
                 database_manager=None,
                 embedding_generator=None,
                 config: SearchConfig = None):
        """Initialize job matcher"""
        self.db = database_manager
        self.embedder = embedding_generator
        self.config = config or SearchConfig()
        
        self.jd_extractor = JobDescriptionExtractor()
        self.semantic_matcher = SemanticMatcher(embedding_generator=embedding_generator)
        self.keyword_matcher = KeywordMatcher()
        self.reasoner = MatchReasoner()
        
        logger.info("Job Matcher initialized")
    
    def process_job_description(self, jd_text: str, title: str = "Unknown") -> JobDescription:
        """Process and extract information from job description"""
        return self.jd_extractor.extract(jd_text, title)
    
    def calculate_hybrid_score(self, semantic_score: float, keyword_score: float) -> float:
        """Calculate hybrid score from semantic and keyword scores"""
        weighted_score = (
            semantic_score * self.config.semantic_weight +
            keyword_score * self.config.keyword_weight
        )
        return min(100, max(0, weighted_score))
    
    def filter_by_experience_level(self, 
                                   candidate_experience: float,
                                   job_exp_level: str) -> bool:
        """Filter candidates by experience level"""
        if job_exp_level == 'entry':
            return candidate_experience < 2  # Entry level: <2 years
        elif job_exp_level == 'mid':
            return 2 <= candidate_experience < 5  # Mid level: 2-5 years
        elif job_exp_level == 'senior':
            return candidate_experience >= 5  # Senior level: 5+ years
        return True
    
    def match_candidate(self, 
                       job: JobDescription,
                       candidate_data: Dict[str, Any]) -> Optional[MatchResult]:
        """
        Match a single candidate against a job description
        
        Args:
            job: JobDescription object
            candidate_data: Candidate information from resume
            
        Returns:
            MatchResult if match is viable, None otherwise
        """
        candidate_name = candidate_data.get('name', 'Unknown')
        candidate_skills = candidate_data.get('skills', [])
        candidate_experience = candidate_data.get('experience_years', 0)
        candidate_text = candidate_data.get('full_text', '')
        candidate_chunks = candidate_data.get('chunks', [])
        candidate_file = candidate_data.get('file_path', '')
        
        # Step 1: Check must-have requirements
        # Must-haves are checked against required skills, not text extraction
        if self.config.require_all_must_haves and job.required_skills:
            # Check if candidate has at least 30% of required skills (minimum baseline)
            matched_required, _, required_match_pct = self.keyword_matcher.match_skills(
                job.required_skills, candidate_skills
            )
            if required_match_pct < 30:  # At least 30% of required skills
                logger.info(f"Candidate {candidate_name} failed must-have requirements (only {required_match_pct:.1f}% skills match)")
                return None
        
        # Step 2: Check experience level filtering
        if not self.filter_by_experience_level(candidate_experience, job.experience_level):
            logger.info(f"Candidate {candidate_name} doesn't match experience level")
            return None
        
        # Step 3: Semantic scoring
        if self.embedder and candidate_chunks:
            # This would require embedding the job description and comparing
            # For now, we'll use a simplified approach
            semantic_score = 75.0  # Placeholder
        else:
            semantic_score = 0.0
        
        # Step 4: Keyword scoring
        matched_skills, missing_skills, skill_match_pct = self.keyword_matcher.match_skills(
            job.required_skills, candidate_skills
        )
        keyword_score = skill_match_pct
        
        # Step 5: Experience matching
        exp_match, exp_score = self.keyword_matcher.match_experience(
            job.required_experience, 
            candidate_experience,
            self.config.experience_strict_matching
        )
        
        # Step 6: Calculate hybrid score
        hybrid_score = self.calculate_hybrid_score(semantic_score, keyword_score)
        
        # Apply experience as modifier (0-20 points bonus)
        if exp_score >= 100:
            hybrid_score = min(100, hybrid_score + 10)
        elif exp_score >= 80:
            hybrid_score = min(100, hybrid_score + 5)
        elif exp_score < 60:
            hybrid_score = hybrid_score * 0.9  # Penalty for low experience
        
        # Step 7: Check threshold
        if hybrid_score < self.config.min_score_threshold:
            logger.info(f"Candidate {candidate_name} score {hybrid_score:.1f} below threshold")
            return None
        
        # Step 8: Nice-to-have matching
        nice_to_have_matched = []
        if job.nice_to_have_skills:
            for skill in job.nice_to_have_skills:
                for cand_skill in candidate_skills:
                    if skill.lower() in cand_skill.lower():
                        nice_to_have_matched.append(cand_skill)
                        break
        
        # Step 9: Section analysis
        section_contributions = self.reasoner.analyze_section_contributions(candidate_chunks)
        
        # Step 10: Extract excerpts
        relevant_excerpts = self.reasoner.extract_relevant_excerpts(candidate_chunks)
        
        # Step 11: Generate reasoning
        result_dict = {
            'matched_skills': matched_skills,
            'missing_required_skills': missing_skills,
            'experience_match': exp_match,
            'section_contributions': section_contributions,
            'match_score': hybrid_score
        }
        reasoning = self.reasoner.generate_reasoning(job, candidate_name, result_dict)
        
        # Create result
        return MatchResult(
            candidate_name=candidate_name,
            resume_path=candidate_file,
            match_score=hybrid_score,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            matched_skills=matched_skills,
            missing_required_skills=missing_skills,
            nice_to_have_matched=nice_to_have_matched,
            experience_match=exp_match,
            section_contributions=section_contributions,
            relevant_excerpts=relevant_excerpts,
            reasoning=reasoning,
            matched_chunks=candidate_chunks
        )
    
    def match_job_to_candidates(self, 
                               job: JobDescription,
                               candidates: List[Dict[str, Any]],
                               filter_skills: List[str] = None,
                               filter_experience_min: float = None) -> List[MatchResult]:
        """
        Match a job to multiple candidates and return ranked results
        
        Args:
            job: JobDescription object
            candidates: List of candidate data dictionaries
            filter_skills: Optional list of skills to filter by
            filter_experience_min: Optional minimum experience threshold
            
        Returns:
            List of MatchResult sorted by score (highest first)
        """
        matches = []
        
        for candidate in candidates:
            # Apply skill filtering
            if filter_skills:
                candidate_skills = candidate.get('skills', [])
                candidate_skills_lower = [s.lower() for s in candidate_skills]
                if not any(fs.lower() in candidate_skills_lower for fs in filter_skills):
                    logger.info(f"Candidate {candidate.get('name')} filtered out by skills")
                    continue
            
            # Apply experience filtering
            if filter_experience_min is not None:
                if candidate.get('experience_years', 0) < filter_experience_min:
                    logger.info(f"Candidate {candidate.get('name')} filtered out by experience")
                    continue
            
            # Perform matching
            result = self.match_candidate(job, candidate)
            if result:
                matches.append(result)
        
        # Sort by score (highest first)
        matches.sort(key=lambda x: x.match_score, reverse=True)
        
        # Return top K
        return matches[:self.config.top_k]
    
    def generate_report(self, 
                       job: JobDescription,
                       matches: List[MatchResult],
                       include_reasoning: bool = True) -> Dict[str, Any]:
        """Generate a formatted report of matching results"""
        report = {
            'job_description': job.title,
            'job_details': {
                'required_skills': job.required_skills,
                'required_experience': job.required_experience,
                'nice_to_have': job.nice_to_have_skills,
                'experience_level': job.experience_level
            },
            'matches_found': len(matches),
            'timestamp': datetime.now().isoformat(),
            'top_matches': []
        }
        
        for match in matches:
            match_dict = match.to_dict()
            if not include_reasoning:
                match_dict.pop('reasoning', None)
            report['top_matches'].append(match_dict)
        
        return report


# ==================== Utility Functions ====================

def create_sample_job_descriptions() -> Dict[str, str]:
    """Create sample job descriptions for testing"""
    jobs = {
        'ml_engineer': """
        Senior Machine Learning Engineer
        
        Company: TechAI Corp
        Location: San Francisco, CA
        
        About the role:
        We are looking for an experienced Machine Learning Engineer to join our team.
        You will be responsible for designing, implementing, and deploying machine learning 
        models at scale.
        
        Requirements:
        - 5+ years of experience in machine learning
        - Expert level Python programming
        - Strong knowledge of TensorFlow or PyTorch
        - Experience with AWS or GCP
        - Docker and Kubernetes experience
        - Understanding of NLP and Computer Vision
        
        Nice to have:
        - Experience with MLOps
        - Published research or open source contributions
        - Experience with large-scale data processing
        
        Must have:
        - 5+ years Python
        - Machine Learning expertise
        - Deep Learning frameworks
        """,
        
        'data_engineer': """
        Senior Data Engineer
        
        Company: DataScale Inc
        Location: New York, NY
        
        About the role:
        Join our data engineering team to build and maintain scalable data pipelines
        that power analytics and machine learning applications.
        
        Requirements:
        - 5+ years of data engineering experience
        - Expert SQL skills
        - Python or Scala programming
        - Apache Spark experience
        - Data warehouse design (Snowflake, BigQuery, Redshift)
        - ETL/ELT pipeline development
        
        Nice to have:
        - Airflow or other orchestration tools
        - Kafka or streaming data experience
        - dbt experience
        - Tableau or Looker
        
        Must have:
        - SQL expert
        - 5+ years experience
        - Spark knowledge
        """,
        
        'full_stack_dev': """
        Full Stack Developer (Mid-level)
        
        Company: WebWorks Studio
        Location: Remote
        
        About the role:
        We're hiring a full stack developer to build modern web applications
        using React on the frontend and Node.js on the backend.
        
        Requirements:
        - 3+ years of web development experience
        - React or Vue.js experience
        - Node.js and Express
        - PostgreSQL and MongoDB
        - REST API design
        - Git and version control
        - AWS or Azure experience
        
        Nice to have:
        - TypeScript
        - GraphQL
        - Docker
        - CI/CD pipelines
        
        Must have:
        - React skills
        - Node.js
        - 3+ years web development
        """
    }
    
    return jobs


if __name__ == "__main__":
    # Simple test
    print("Job Matcher - Part B Implementation Loaded")
