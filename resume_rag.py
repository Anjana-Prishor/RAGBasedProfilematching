"""
RAG System for Resume Processing and Embedding

Part A: Document Processing Pipeline
- Load resumes from file system
- Intelligent document chunking preserving sections
- Generate embeddings using HuggingFace models
- Store in ChromaDB vector database
- Extract and store metadata
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
import pypdf
from docx import Document as DocxDocument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Data Models ====================

class ResumeMetadata(BaseModel):
    """Metadata extracted from resume"""
    name: str
    skills: List[str] = Field(default_factory=list)
    experience_years: float = 0.0
    education: List[str] = Field(default_factory=list)
    file_path: str
    total_chunks: int = 0


class DocumentChunk(BaseModel):
    """Represents a chunk of a document"""
    content: str
    section: str
    resume_id: str
    metadata: ResumeMetadata
    chunk_index: int


# ==================== Document Loader ====================

class DocumentLoader:
    """Load and parse resume documents"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx', '.txt'}
    
    def load_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            logger.info(f"Successfully loaded PDF: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return ""
    
    def load_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            logger.info(f"Successfully loaded DOCX: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            return ""
    
    def load_txt(self, file_path: str) -> str:
        """Load text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(f"Successfully loaded TXT: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            return ""
    
    def load(self, file_path: str) -> str:
        """Load document based on file type"""
        file_ext = Path(file_path).suffix.lower()
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""
        
        if file_ext == '.pdf':
            return self.load_pdf(file_path)
        elif file_ext == '.docx':
            return self.load_docx(file_path)
        elif file_ext == '.txt':
            return self.load_txt(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_ext}")
            return ""


# ==================== Metadata Extractor ====================

class MetadataExtractor:
    """Extract key fields from resume text"""
    
    # Keywords for different sections
    EDUCATION_KEYWORDS = [
        'education', 'degree', 'bachelor', 'master', 'phd', 'university', 
        'college', 'institute', 'certification', 'certified'
    ]
    
    EXPERIENCE_KEYWORDS = [
        'experience', 'work experience', 'professional experience', 
        'employment', 'position', 'role', 'worked as', 'worked at'
    ]
    
    SKILLS_KEYWORDS = [
        'skills', 'technical skills', 'programming languages', 'tools', 
        'technologies', 'proficiencies', 'core competencies'
    ]
    
    def __init__(self):
        # Common programming languages and tools
        self.technical_keywords = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis',
            'machine learning', 'ml', 'deep learning', 'nlp', 'computer vision',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'cloud',
            'react', 'angular', 'vue', 'node', 'express', 'django', 'flask',
            'git', 'linux', 'windows', 'rest api', 'graphql', 'microservices'
        }
    
    def extract_name(self, text: str) -> str:
        """Extract candidate name from resume"""
        lines = text.split('\n')
        
        # Usually the first non-empty line or within first 5 lines
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) < 100 and len(line.split()) <= 3:
                # Skip common header words
                if not any(word.lower() in line.lower() for word in 
                          ['email', 'phone', 'linkedin', 'github', 'resume']):
                    return line
        
        return "Unknown"
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        text_lower = text.lower()
        skills = set()
        
        # Find skills section
        for keyword in self.SKILLS_KEYWORDS:
            if keyword in text_lower:
                # Extract content after skills section
                idx = text_lower.find(keyword)
                section = text_lower[idx:idx+1000]
                
                # Look for technical keywords in this section
                for tech in self.technical_keywords:
                    if tech in section:
                        skills.add(tech.title())
        
        # Also search entire text for technical keywords
        for tech in self.technical_keywords:
            if tech in text_lower:
                skills.add(tech.title())
        
        return list(skills)
    
    def extract_experience_years(self, text: str) -> float:
        """Extract years of experience from resume"""
        # Look for patterns like "X years of experience"
        pattern = r'(\d+)\+?\s*years?\s+of\s+experience'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        if matches:
            return float(matches[0])
        
        # Alternative pattern: look for years in dates
        date_pattern = r'(20\d{2})\s*-\s*(20\d{2}|present|current)'
        matches = re.findall(date_pattern, text, re.IGNORECASE)
        
        if matches:
            try:
                years = []
                for start, end in matches:
                    end_year = 2024 if end.lower() in ['present', 'current'] else int(end)
                    years.append(end_year - int(start))
                return float(sum(years)) / len(years)
            except:
                pass
        
        return 0.0
    
    def extract_education(self, text: str) -> List[str]:
        """Extract education details from resume"""
        text_lower = text.lower()
        education = []
        
        # Find education section
        for keyword in self.EDUCATION_KEYWORDS:
            if keyword in text_lower:
                idx = text_lower.find(keyword)
                # Extract some context after finding education keyword
                section = text[idx:idx+800]
                
                # Look for degree patterns
                degree_patterns = [
                    r'(bachelor|master|phd|b\.?a\.?|b\.?s\.?|m\.?a\.?|m\.?s\.?|b\.?tech|m\.?tech)',
                    r'in\s+(\w+(?:\s+\w+)?)'  # "in Computer Science"
                ]
                
                for pattern in degree_patterns:
                    matches = re.findall(pattern, section, re.IGNORECASE)
                    education.extend(matches)
                break
        
        return list(set(education))[:3]  # Return top 3 education entries
    
    def extract(self, text: str, file_path: str) -> ResumeMetadata:
        """Extract all metadata from resume"""
        return ResumeMetadata(
            name=self.extract_name(text),
            skills=self.extract_skills(text),
            experience_years=self.extract_experience_years(text),
            education=self.extract_education(text),
            file_path=file_path
        )


# ==================== Document Chunker ====================

class DocumentChunker:
    """Intelligently chunk documents while preserving sections"""
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def identify_sections(self, text: str) -> Dict[int, Tuple[str, int]]:
        """Identify section boundaries in resume"""
        section_keywords = {
            'education': ['education', 'academic', 'degree', 'qualification'],
            'experience': ['experience', 'employment', 'work history', 'professional'],
            'skills': ['skills', 'technical skills', 'proficiencies', 'competencies'],
            'projects': ['projects', 'portfolio', 'achievements'],
            'certifications': ['certification', 'certified', 'license'],
            'summary': ['summary', 'objective', 'profile', 'introduction'],
            'contact': ['contact', 'email', 'phone', 'linkedin', 'github']
        }
        
        sections = {}
        text_lower = text.lower()
        
        for section_name, keywords in section_keywords.items():
            for keyword in keywords:
                idx = text_lower.find(keyword)
                if idx != -1:
                    sections[idx] = (section_name, idx)
                    break
        
        return sections
    
    def chunk(self, text: str, metadata: ResumeMetadata) -> List[DocumentChunk]:
        """Chunk document intelligently"""
        if not text or len(text) < 10:
            logger.warning(f"Empty or very short document for {metadata.name}")
            return []
        
        sections = self.identify_sections(text)
        chunks = []
        chunk_index = 0
        
        # Sort sections by position
        sorted_sections = sorted(sections.items())
        
        # Process document section by section
        for i, (start_pos, (section_name, _)) in enumerate(sorted_sections):
            # Determine end position
            if i + 1 < len(sorted_sections):
                end_pos = sorted_sections[i + 1][0]
            else:
                end_pos = len(text)
            
            section_text = text[start_pos:end_pos].strip()
            
            if not section_text:
                continue
            
            # Split long sections into chunks
            words = section_text.split()
            current_chunk = []
            
            for word in words:
                current_chunk.append(word)
                
                # Check if chunk reaches desired size
                if len(' '.join(current_chunk)) >= self.chunk_size:
                    chunk_text = ' '.join(current_chunk)
                    
                    chunk = DocumentChunk(
                        content=chunk_text,
                        section=section_name,
                        resume_id=metadata.file_path,
                        metadata=metadata,
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Keep overlap
                    overlap_count = int(len(current_chunk) * self.overlap / self.chunk_size)
                    current_chunk = current_chunk[-overlap_count:] if overlap_count > 0 else []
            
            # Add remaining words as chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = DocumentChunk(
                    content=chunk_text,
                    section=section_name,
                    resume_id=metadata.file_path,
                    metadata=metadata,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
        
        metadata.total_chunks = len(chunks)
        logger.info(f"Created {len(chunks)} chunks for {metadata.name}")
        
        return chunks


# ==================== Embedding Generator ====================

class EmbeddingGenerator:
    """Generate embeddings for text using HuggingFace models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Popular models:
        - all-MiniLM-L6-v2: Fast, 384 dimensions
        - all-mpnet-base-v2: Better quality, 768 dimensions
        - paraphrase-MiniLM-L6-v2: Good for paraphrasing
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)


# ==================== Vector Database Manager ====================

class VectorDatabaseManager:
    """Manage ChromaDB for storing embeddings and metadata"""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "resumes"):
        """Initialize ChromaDB"""
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Create a persistent database using the current ChromaDB client API.
        settings = Settings(anonymized_telemetry=False)
        self.client = chromadb.PersistentClient(path=db_path, settings=settings)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ChromaDB at {db_path}")
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[np.ndarray]):
        """Add document chunks and embeddings to database"""
        if len(chunks) == 0 or len(embeddings) == 0:
            logger.warning("No chunks or embeddings to add")
            return
        
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = f"{chunk.resume_id}_{chunk.chunk_index}"
            ids.append(chunk_id)
            documents.append(chunk.content)
            embeddings_list.append(embedding)
            
            # Store metadata
            metadata = {
                "section": chunk.section,
                "resume_id": chunk.resume_id,
                "chunk_index": chunk.chunk_index,
                "candidate_name": chunk.metadata.name,
                "skills": ",".join(chunk.metadata.skills),
                "experience_years": str(chunk.metadata.experience_years),
                "education": ",".join(chunk.metadata.education),
                "file_path": chunk.metadata.file_path
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to database")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "metadata_sample": self.collection.get(limit=1)
        }
    
    def persist(self):
        """Persist database to disk when supported by the client."""
        persist_method = getattr(self.client, "persist", None)
        if callable(persist_method):
            persist_method()
        logger.info(f"Database persisted to {self.db_path}")


# ==================== RAG System ====================

class ResumeRAGSystem:
    """Complete RAG system for resume processing"""
    
    def __init__(
        self,
        db_path: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50
    ):
        """Initialize RAG system"""
        self.loader = DocumentLoader()
        self.extractor = MetadataExtractor()
        self.chunker = DocumentChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedding_gen = EmbeddingGenerator(model_name=embedding_model)
        self.db = VectorDatabaseManager(db_path=db_path)
        
        self.processed_resumes = []
        logger.info("RAG System initialized")
    
    def process_resume(self, file_path: str) -> Tuple[bool, ResumeMetadata, List[DocumentChunk]]:
        """Process a single resume"""
        logger.info(f"Processing resume: {file_path}")
        
        # Load document
        text = self.loader.load(file_path)
        if not text:
            logger.error(f"Failed to load resume: {file_path}")
            return False, None, []
        
        # Extract metadata
        metadata = self.extractor.extract(text, file_path)
        logger.info(f"Extracted metadata for {metadata.name}")
        
        # Chunk document
        chunks = self.chunker.chunk(text, metadata)
        if not chunks:
            logger.error(f"Failed to chunk resume: {file_path}")
            return False, metadata, []
        
        # Generate embeddings
        texts_to_embed = [chunk.content for chunk in chunks]
        embeddings = self.embedding_gen.encode_batch(texts_to_embed)
        
        # Store in database
        self.db.add_chunks(chunks, embeddings)
        
        self.processed_resumes.append({
            'file_path': file_path,
            'name': metadata.name,
            'chunks': len(chunks)
        })
        
        return True, metadata, chunks
    
    def process_resumes_from_directory(self, directory: str) -> Dict[str, Any]:
        """Process all resumes from a directory"""
        results = {
            'total_files': 0,
            'processed': 0,
            'failed': 0,
            'resumes': []
        }
        
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return results
        
        files = []
        for ext in ['.pdf', '.docx', '.txt']:
            files.extend(Path(directory).glob(f'*{ext}'))
        
        results['total_files'] = len(files)
        
        for file_path in files:
            try:
                success, metadata, chunks = self.process_resume(str(file_path))
                if success:
                    results['processed'] += 1
                    results['resumes'].append({
                        'name': metadata.name,
                        'file': str(file_path),
                        'skills': metadata.skills,
                        'experience_years': metadata.experience_years,
                        'education': metadata.education,
                        'chunks': len(chunks)
                    })
                else:
                    results['failed'] += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results['failed'] += 1
        
        # Persist database
        self.db.persist()
        
        logger.info(f"Batch processing complete: {results['processed']} processed, {results['failed']} failed")
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get RAG system information"""
        return {
            'embedding_dimension': self.embedding_gen.embedding_dim,
            'chunk_size': self.chunker.chunk_size,
            'chunk_overlap': self.chunker.overlap,
            'database': self.db.get_collection_info(),
            'processed_resumes': len(self.processed_resumes)
        }


# ==================== Utility Functions ====================

def create_sample_resumes(output_dir: str = "./sample_resumes"):
    """Create sample resumes for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    sample_resumes = [
        {
            'filename': 'john_doe_ml.txt',
            'content': """
            JOHN DOE
            john.doe@email.com | LinkedIn: /in/johndoe | GitHub: github.com/johndoe
            
            SUMMARY
            Senior Machine Learning Engineer with 6+ years of experience in building and deploying ML models.
            Strong background in Python, TensorFlow, and cloud technologies.
            
            EXPERIENCE
            Senior ML Engineer, TechCorp (2021-Present)
            - Developed recommendation system using collaborative filtering and NLP
            - Led team of 3 engineers in building production ML pipeline
            - Improved model performance by 25% using ensemble methods
            - Tech Stack: Python, TensorFlow, PyTorch, AWS, Docker
            
            ML Engineer, DataSolutions (2018-2021)
            - Built computer vision models for image classification achieving 95% accuracy
            - Implemented data pipeline processing 10M+ records daily
            - Deployed models using Docker and Kubernetes
            - Tech Stack: Python, OpenCV, Scikit-learn, PostgreSQL, GCP
            
            Junior Developer, StartupXYZ (2016-2018)
            - Contributed to backend services using Python and Django
            - Worked with SQL databases and REST APIs
            - Tech Stack: Python, Django, MySQL, JavaScript
            
            EDUCATION
            Master of Science in Computer Science, University of California (2016)
            Bachelor of Science in Information Technology, State University (2014)
            
            SKILLS
            Programming Languages: Python, Java, JavaScript, SQL
            ML/AI: TensorFlow, PyTorch, Scikit-learn, Keras, NLP, Computer Vision
            Cloud & DevOps: AWS, GCP, Docker, Kubernetes
            Databases: PostgreSQL, MongoDB, Redis
            Tools: Git, JIRA, Jupyter, VS Code
            """
        },
        {
            'filename': 'jane_smith_data.txt',
            'content': """
            JANE SMITH
            jane.smith@email.com | LinkedIn: /in/janesmith | GitHub: github.com/janesmith
            
            PROFILE
            Data Engineer with 5+ years of experience building scalable data pipelines and analytics platforms.
            Expertise in distributed systems, big data technologies, and cloud data warehousing.
            
            PROFESSIONAL EXPERIENCE
            Senior Data Engineer, CloudData Inc (2022-Present)
            - Architected data lake processing 500GB+ daily using Apache Spark
            - Reduced data processing time by 40% through pipeline optimization
            - Mentored junior engineers on best practices in data engineering
            - Technologies: Spark, Airflow, Snowflake, AWS S3, Python
            
            Data Engineer, Analytics Hub (2019-2022)
            - Built ETL pipelines using Apache Airflow for real-time data processing
            - Managed large-scale databases with millions of records
            - Created data models for business intelligence and reporting
            - Technologies: Python, Airflow, PostgreSQL, Tableau, SQL
            
            Junior Data Analyst, Financial Corp (2017-2019)
            - Analyzed financial data using SQL and Python
            - Created dashboards for executive reporting
            - Technologies: SQL, Python, Tableau, Excel
            
            EDUCATION
            Master of Engineering in Data Science, Tech Institute (2017)
            Bachelor of Science in Statistics, State College (2015)
            
            SKILLS
            Programming: Python, SQL, Java, Scala
            Data Technologies: Spark, Hadoop, Kafka, Airflow, dbt
            Databases: PostgreSQL, MongoDB, Cassandra, Snowflake
            Cloud: AWS, Azure, GCP
            BI Tools: Tableau, Looker, Power BI
            """
        },
        {
            'filename': 'alice_kumar_fullstack.txt',
            'content': """
            ALICE KUMAR
            alice.kumar@email.com | LinkedIn: /in/alicekumar | Portfolio: alicekumar.dev
            
            SUMMARY
            Full Stack Developer with 4+ years of experience building responsive web applications.
            Proficient in React, Node.js, and modern web technologies. Strong focus on user experience.
            
            EXPERIENCE
            Full Stack Engineer, WebSolutions (2021-Present)
            - Developed and maintained 5+ production React applications serving 100K+ users
            - Designed and implemented Node.js backend services with REST and GraphQL APIs
            - Improved application performance by optimizing bundle size and implementing lazy loading
            - Tech Stack: React, TypeScript, Node.js, PostgreSQL, AWS Lambda
            
            Frontend Developer, StartupIO (2019-2021)
            - Built responsive UI components using React and Material-UI
            - Reduced page load time by 50% through code splitting and optimization
            - Collaborated with designers to implement pixel-perfect designs
            - Tech Stack: React, Redux, CSS, JavaScript, Jest
            
            Junior Web Developer, LocalTech (2017-2019)
            - Developed static and dynamic websites using HTML, CSS, JavaScript
            - Maintained WordPress sites for multiple clients
            - Tech Stack: HTML, CSS, JavaScript, WordPress, PHP
            
            EDUCATION
            Bachelor of Technology in Computer Science, Engineering College (2017)
            
            SKILLS
            Frontend: React, Vue.js, HTML5, CSS3, TypeScript, Redux
            Backend: Node.js, Express, Python, REST API, GraphQL
            Databases: PostgreSQL, MongoDB, Firebase
            DevOps: Docker, AWS, CI/CD, GitHub Actions
            Tools: Git, Webpack, Jest, VS Code
            """
        }
    ]
    
    for resume in sample_resumes:
        path = os.path.join(output_dir, resume['filename'])
        with open(path, 'w') as f:
            f.write(resume['content'])
        logger.info(f"Created sample resume: {path}")
    
    return output_dir


# ==================== Main Example ====================

if __name__ == "__main__":
    # Create sample resumes
    resume_dir = create_sample_resumes()
    
    # Initialize RAG system
    rag_system = ResumeRAGSystem(
        db_path="./chroma_db",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=300,
        chunk_overlap=50
    )
    
    # Process all resumes
    print("\n" + "="*60)
    print("Processing resumes...")
    print("="*60)
    
    results = rag_system.process_resumes_from_directory(resume_dir)
    
    # Print results
    print("\nProcessing Results:")
    print(f"Total files: {results['total_files']}")
    print(f"Successfully processed: {results['processed']}")
    print(f"Failed: {results['failed']}")
    
    print("\nProcessed Resumes:")
    for resume in results['resumes']:
        print(f"\n  Name: {resume['name']}")
        print(f"  File: {resume['file']}")
        print(f"  Experience: {resume['experience_years']} years")
        print(f"  Skills: {', '.join(resume['skills'][:5])}...")
        print(f"  Education: {', '.join(resume['education'])}")
        print(f"  Chunks created: {resume['chunks']}")
    
    # Print system info
    print("\n" + "="*60)
    print("RAG System Information:")
    print("="*60)
    
    info = rag_system.get_system_info()
    print(f"Embedding dimension: {info['embedding_dimension']}")
    print(f"Chunk size: {info['chunk_size']}")
    print(f"Chunk overlap: {info['chunk_overlap']}")
    print(f"Total documents in database: {info['database']['total_documents']}")
    print(f"Total resumes processed: {info['processed_resumes']}")
