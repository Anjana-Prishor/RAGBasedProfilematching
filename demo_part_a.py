"""
Simplified RAG System Demo - Part A
Works without external dependencies installed
Demonstrates all core functionality of the RAG system
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleDocumentLoader:
    """Load documents from various formats"""
    
    def load_txt(self, file_path: str) -> str:
        """Load text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(f"Loaded TXT: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            return ""
    
    def load(self, file_path: str) -> str:
        """Load document"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.txt':
            return self.load_txt(file_path)
        else:
            logger.warning(f"For demo, only TXT files supported. Got: {file_ext}")
            return ""


class SimpleMetadataExtractor:
    """Extract metadata from resume text"""
    
    SKILLS_KEYWORDS = [
        'skills', 'technical skills', 'programming languages', 'tools'
    ]
    
    EXPERIENCE_KEYWORDS = [
        'experience', 'work experience', 'professional experience', 'employment'
    ]
    
    EDUCATION_KEYWORDS = [
        'education', 'degree', 'bachelor', 'master', 'phd', 'university'
    ]
    
    def __init__(self):
        self.technical_keywords = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'sql', 'mongodb', 'postgresql', 'mysql', 'redis',
            'machine learning', 'ml', 'deep learning', 'nlp', 'computer vision',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'react', 'angular', 'vue', 'node', 'express', 'django', 'flask',
            'git', 'linux', 'rest api', 'graphql'
        }
    
    def extract_name(self, text: str) -> str:
        """Extract candidate name"""
        lines = text.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) < 100:
                if not any(word.lower() in line.lower() for word in 
                          ['email', 'phone', 'linkedin', 'github', 'resume', '@']):
                    return line
        return "Unknown"
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume"""
        text_lower = text.lower()
        skills = set()
        
        for tech in self.technical_keywords:
            if tech in text_lower:
                skills.add(tech.title())
        
        return sorted(list(skills))
    
    def extract_experience_years(self, text: str) -> float:
        """Extract years of experience"""
        pattern = r'(\d+)\+?\s*years?\s+of\s+experience'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        if matches:
            return float(matches[0])
        return 0.0
    
    def extract_education(self, text: str) -> List[str]:
        """Extract education"""
        text_lower = text.lower()
        education = []
        
        for keyword in self.EDUCATION_KEYWORDS:
            if keyword in text_lower:
                idx = text_lower.find(keyword)
                section = text[idx:idx+500]
                
                patterns = [
                    r'(bachelor|master|phd|b\.?a\.?|b\.?s\.?|m\.?a\.?|m\.?s\.?)',
                    r'in\s+(\w+(?:\s+\w+)?)'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, section, re.IGNORECASE)
                    education.extend(matches)
                break
        
        return list(set(education))[:3]
    
    def extract(self, text: str, file_path: str) -> Dict[str, Any]:
        """Extract all metadata"""
        return {
            'name': self.extract_name(text),
            'skills': self.extract_skills(text),
            'experience_years': self.extract_experience_years(text),
            'education': self.extract_education(text),
            'file_path': file_path
        }


class SimpleDocumentChunker:
    """Chunk documents intelligently"""
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def identify_sections(self, text: str) -> Dict[int, str]:
        """Identify section boundaries"""
        section_keywords = {
            'education': ['education', 'academic', 'degree'],
            'experience': ['experience', 'employment', 'work history'],
            'skills': ['skills', 'technical skills', 'competencies'],
            'projects': ['projects', 'portfolio'],
            'summary': ['summary', 'objective', 'profile'],
        }
        
        sections = {}
        text_lower = text.lower()
        
        for section_name, keywords in section_keywords.items():
            for keyword in keywords:
                idx = text_lower.find(keyword)
                if idx != -1:
                    sections[idx] = section_name
                    break
        
        return sections
    
    def chunk(self, text: str, metadata: Dict) -> List[Dict[str, Any]]:
        """Chunk document"""
        if not text or len(text) < 10:
            logger.warning(f"Empty document for {metadata['name']}")
            return []
        
        sections = self.identify_sections(text)
        chunks = []
        chunk_index = 0
        
        sorted_sections = sorted(sections.items())
        
        for i, (start_pos, section_name) in enumerate(sorted_sections):
            if i + 1 < len(sorted_sections):
                end_pos = sorted_sections[i + 1][0]
            else:
                end_pos = len(text)
            
            section_text = text[start_pos:end_pos].strip()
            
            if not section_text:
                continue
            
            words = section_text.split()
            current_chunk = []
            
            for word in words:
                current_chunk.append(word)
                
                if len(' '.join(current_chunk)) >= self.chunk_size:
                    chunk_text = ' '.join(current_chunk)
                    
                    chunks.append({
                        'content': chunk_text,
                        'section': section_name,
                        'resume_id': metadata['file_path'],
                        'chunk_index': chunk_index,
                        'metadata': metadata
                    })
                    chunk_index += 1
                    
                    overlap_count = int(len(current_chunk) * self.overlap / self.chunk_size)
                    current_chunk = current_chunk[-overlap_count:] if overlap_count > 0 else []
            
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'section': section_name,
                    'resume_id': metadata['file_path'],
                    'chunk_index': chunk_index,
                    'metadata': metadata
                })
                chunk_index += 1
        
        logger.info(f"Created {len(chunks)} chunks for {metadata['name']}")
        return chunks


class InMemoryVectorDB:
    """Simple in-memory vector storage for demo"""
    
    def __init__(self):
        self.documents = []
        self.metadata_store = {}
    
    def add_chunks(self, chunks: List[Dict], embeddings: List = None):
        """Add chunks to database"""
        for i, chunk in enumerate(chunks):
            doc_id = f"{chunk['resume_id']}_{chunk['chunk_index']}"
            self.documents.append({
                'id': doc_id,
                'content': chunk['content'],
                'section': chunk['section'],
                'metadata': chunk['metadata']
            })
            self.metadata_store[doc_id] = chunk['metadata']
        
        logger.info(f"Added {len(chunks)} chunks to in-memory database")
    
    def get_info(self) -> Dict:
        """Get database info"""
        return {
            'total_documents': len(self.documents),
            'documents': len(set(d['metadata']['file_path'] for d in self.documents))
        }


class SimplifiedRAGSystem:
    """Simplified RAG system for demo"""
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """Initialize RAG system"""
        self.loader = SimpleDocumentLoader()
        self.extractor = SimpleMetadataExtractor()
        self.chunker = SimpleDocumentChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.db = InMemoryVectorDB()
        
        self.processed_resumes = []
        logger.info("RAG System initialized (Simplified Version)")
    
    def process_resume(self, file_path: str) -> Tuple[bool, Dict, List]:
        """Process a single resume"""
        logger.info(f"Processing resume: {file_path}")
        
        # Load document
        text = self.loader.load(file_path)
        if not text:
            logger.error(f"Failed to load resume: {file_path}")
            return False, None, []
        
        # Extract metadata
        metadata = self.extractor.extract(text, file_path)
        logger.info(f"Extracted metadata for {metadata['name']}")
        
        # Chunk document
        chunks = self.chunker.chunk(text, metadata)
        if not chunks:
            logger.error(f"Failed to chunk resume: {file_path}")
            return False, metadata, []
        
        # Store in database (embeddings simulated)
        self.db.add_chunks(chunks)
        
        self.processed_resumes.append({
            'file_path': file_path,
            'name': metadata['name'],
            'chunks': len(chunks)
        })
        
        return True, metadata, chunks
    
    def process_resumes_from_directory(self, directory: str) -> Dict[str, Any]:
        """Process all resumes from directory"""
        results = {
            'total_files': 0,
            'processed': 0,
            'failed': 0,
            'resumes': []
        }
        
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return results
        
        files = list(Path(directory).glob('*.txt'))
        results['total_files'] = len(files)
        
        for file_path in files:
            try:
                success, metadata, chunks = self.process_resume(str(file_path))
                if success:
                    results['processed'] += 1
                    results['resumes'].append({
                        'name': metadata['name'],
                        'file': str(file_path),
                        'skills': metadata['skills'],
                        'experience_years': metadata['experience_years'],
                        'education': metadata['education'],
                        'chunks': len(chunks)
                    })
                else:
                    results['failed'] += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results['failed'] += 1
        
        logger.info(f"Batch processing complete: {results['processed']} processed")
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        db_info = self.db.get_info()
        return {
            'chunk_size': self.chunker.chunk_size,
            'chunk_overlap': self.chunker.overlap,
            'database': db_info,
            'processed_resumes': len(self.processed_resumes)
        }


def create_sample_resumes(output_dir: str = "./sample_resumes"):
    """Create sample resumes for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    sample_resumes = [
        {
            'filename': 'john_doe_ml.txt',
            'content': """
JOHN DOE
john.doe@email.com

SUMMARY
Senior Machine Learning Engineer with 6+ years of experience in building and deploying ML models.

EXPERIENCE
Senior ML Engineer, TechCorp (2021-Present)
- Developed recommendation system using collaborative filtering and NLP
- Led team of 3 engineers in building production ML pipeline
- Tech Stack: Python, TensorFlow, PyTorch, AWS, Docker

ML Engineer, DataSolutions (2018-2021)
- Built computer vision models for image classification achieving 95% accuracy
- Implemented data pipeline processing 10M+ records daily
- Tech Stack: Python, OpenCV, Scikit-learn, PostgreSQL, GCP

EDUCATION
Master of Science in Computer Science, University of California (2016)
Bachelor of Science in Information Technology, State University (2014)

SKILLS
Programming Languages: Python, Java, JavaScript, SQL
ML/AI: TensorFlow, PyTorch, Scikit-learn, Keras, NLP, Computer Vision
Cloud & DevOps: AWS, GCP, Docker, Kubernetes
Databases: PostgreSQL, MongoDB, Redis
"""
        },
        {
            'filename': 'jane_smith_data.txt',
            'content': """
JANE SMITH
jane.smith@email.com

PROFILE
Data Engineer with 5+ years of experience building scalable data pipelines and analytics platforms.

EXPERIENCE
Senior Data Engineer, CloudData Inc (2022-Present)
- Architected data lake processing 500GB+ daily using Apache Spark
- Reduced data processing time by 40% through pipeline optimization
- Technologies: Spark, Airflow, Snowflake, AWS S3, Python

Data Engineer, Analytics Hub (2019-2022)
- Built ETL pipelines using Apache Airflow for real-time data processing
- Managed large-scale databases with millions of records
- Technologies: Python, Airflow, PostgreSQL, Tableau, SQL

EDUCATION
Master of Engineering in Data Science, Tech Institute (2017)
Bachelor of Science in Statistics, State College (2015)

SKILLS
Programming: Python, SQL, Java, Scala
Data Technologies: Spark, Hadoop, Kafka, Airflow, dbt
Databases: PostgreSQL, MongoDB, Cassandra, Snowflake
Cloud: AWS, Azure, GCP
"""
        },
        {
            'filename': 'alice_kumar_fullstack.txt',
            'content': """
ALICE KUMAR
alice.kumar@email.com

SUMMARY
Full Stack Developer with 4+ years of experience building responsive web applications.

EXPERIENCE
Full Stack Engineer, WebSolutions (2021-Present)
- Developed and maintained 5+ production React applications serving 100K+ users
- Designed and implemented Node.js backend services with REST and GraphQL APIs
- Improved application performance by optimizing bundle size and implementing lazy loading
- Tech Stack: React, TypeScript, Node.js, PostgreSQL, AWS Lambda

Frontend Developer, StartupIO (2019-2021)
- Built responsive UI components using React and Material-UI
- Reduced page load time by 50% through code splitting and optimization
- Tech Stack: React, Redux, CSS, JavaScript, Jest

EDUCATION
Bachelor of Technology in Computer Science, Engineering College (2017)

SKILLS
Frontend: React, Vue.js, HTML5, CSS3, TypeScript, Redux
Backend: Node.js, Express, Python, REST API, GraphQL
Databases: PostgreSQL, MongoDB, Firebase
DevOps: Docker, AWS, CI/CD, GitHub Actions
"""
        }
    ]
    
    for resume in sample_resumes:
        path = os.path.join(output_dir, resume['filename'])
        with open(path, 'w') as f:
            f.write(resume['content'])
        logger.info(f"Created sample resume: {path}")
    
    return output_dir


# ==================== Main Demo ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RAG-BASED PROFILE MATCHING - PART A DEMO")
    print("Simplified Version (No External Dependencies Required)")
    print("="*70)
    
    # Create sample resumes
    print("\n[Step 1] Creating sample resumes...")
    resume_dir = create_sample_resumes()
    print(f"✓ Sample resumes created in: {resume_dir}")
    
    # Initialize RAG system
    print("\n[Step 2] Initializing RAG system...")
    rag_system = SimplifiedRAGSystem(chunk_size=300, chunk_overlap=50)
    print("✓ RAG system initialized")
    
    # Process resumes
    print("\n[Step 3] Processing resumes...")
    results = rag_system.process_resumes_from_directory(resume_dir)
    
    print(f"\nProcessing Results:")
    print(f"  Total files: {results['total_files']}")
    print(f"  Successfully processed: {results['processed']}")
    print(f"  Failed: {results['failed']}")
    
    # Display results
    print("\n[Step 4] Processed Resumes Analysis:")
    print("-" * 70)
    
    for resume in results['resumes']:
        print(f"\n  Name: {resume['name']}")
        print(f"  File: {resume['file']}")
        print(f"  Experience: {resume['experience_years']} years")
        print(f"  Skills: {', '.join(resume['skills'][:6])}")
        if resume['education']:
            print(f"  Education: {', '.join(resume['education'])}")
        print(f"  Chunks created: {resume['chunks']}")
    
    # System info
    print("\n" + "="*70)
    print("RAG System Information:")
    print("="*70)
    
    info = rag_system.get_system_info()
    print(f"Chunk size: {info['chunk_size']} tokens")
    print(f"Chunk overlap: {info['chunk_overlap']} tokens")
    print(f"Total documents in database: {info['database']['total_documents']}")
    print(f"Total resumes processed: {info['database']['documents']}")
    
    # Summary
    print("\n" + "="*70)
    print("✓ PART A: RAG SYSTEM SETUP - COMPLETE")
    print("="*70)
    print("\nKey Features Implemented:")
    print("  ✓ Document Processing Pipeline")
    print("  ✓ Intelligent Document Chunking")
    print("  ✓ Metadata Extraction")
    print("  ✓ Vector Database (In-Memory Demo)")
    print("\nReady for Part B: Job Matching Engine")
    print("="*70 + "\n")
