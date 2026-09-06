"""
Test script to verify Part A implementation
Demonstrates the RAG system working end-to-end
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resume_rag import (
    ResumeRAGSystem,
    DocumentLoader,
    MetadataExtractor,
    DocumentChunker,
    EmbeddingGenerator,
    create_sample_resumes
)
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_document_loader():
    """Test document loading functionality"""
    print("\n" + "="*70)
    print("TEST 1: Document Loader")
    print("="*70)
    
    loader = DocumentLoader()
    
    # Create test resume
    sample_text = """
    JOHN DOE
    john@example.com
    
    EXPERIENCE
    Senior Python Developer at TechCorp (2020-2024)
    - Developed ML models using TensorFlow
    - 4 years experience with Python
    
    EDUCATION
    Bachelor of Science in Computer Science
    """
    
    test_file = "test_resume.txt"
    with open(test_file, 'w') as f:
        f.write(sample_text)
    
    # Load document
    text = loader.load(test_file)
    assert len(text) > 0, "Failed to load document"
    print("[OK] Document loading works")
    
    # Cleanup
    os.remove(test_file)
    
    return True


def test_metadata_extractor():
    """Test metadata extraction"""
    print("\n" + "="*70)
    print("TEST 2: Metadata Extractor")
    print("="*70)
    
    sample_text = """
    JANE SMITH
    jane.smith@email.com
    
    PROFESSIONAL SUMMARY
    Data Engineer with 5+ years of experience in building scalable data pipelines.
    
    EXPERIENCE
    Senior Data Engineer, CloudData Inc (2022-Present)
    - Worked with Python, Spark, and Airflow
    - Managed large scale databases
    
    Data Engineer, Analytics Hub (2019-2022)
    - Experience in PostgreSQL and Tableau
    
    EDUCATION
    Master of Engineering in Data Science
    Bachelor of Science in Statistics
    
    SKILLS
    Programming: Python, SQL, Scala
    Data: Spark, Hadoop, Airflow, dbt
    Databases: PostgreSQL, MongoDB, Snowflake
    """
    
    extractor = MetadataExtractor()
    metadata = extractor.extract(sample_text, "test.txt")
    
    assert metadata.name != "Unknown", "Failed to extract name"
    assert len(metadata.skills) > 0, "Failed to extract skills"
    assert metadata.experience_years > 0, "Failed to extract experience"
    assert len(metadata.education) > 0, "Failed to extract education"
    
    print(f"[OK] Name: {metadata.name}")
    print(f"[OK] Skills: {', '.join(metadata.skills[:5])}")
    print(f"[OK] Experience: {metadata.experience_years} years")
    print(f"[OK] Education: {', '.join(metadata.education)}")
    
    return True


def test_document_chunker():
    """Test document chunking"""
    print("\n" + "="*70)
    print("TEST 3: Document Chunker")
    print("="*70)
    
    from resume_rag import ResumeMetadata
    
    long_text = """
    EDUCATION SECTION
    This is a long education section with details about degrees.
    Bachelor of Science in Computer Science from University.
    Master of Technology in Machine Learning.
    
    EXPERIENCE SECTION
    Senior Engineer at Company (2020-2024)
    Worked on various projects including machine learning systems.
    Led team of engineers and delivered multiple projects.
    
    SKILLS SECTION
    Python, Java, JavaScript, C++
    Machine Learning, Deep Learning, NLP
    AWS, Docker, Kubernetes
    """ * 3  # Make it long
    
    extractor = MetadataExtractor()
    metadata = extractor.extract(long_text, "test.txt")
    
    chunker = DocumentChunker(chunk_size=200, overlap=30)
    chunks = chunker.chunk(long_text, metadata)
    
    assert len(chunks) > 0, "Failed to create chunks"
    assert all(c.content for c in chunks), "Some chunks are empty"
    assert all(c.section for c in chunks), "Chunks missing section info"
    
    print(f"[OK] Created {len(chunks)} chunks")
    print(f"[OK] Sample chunk size: {len(chunks[0].content)} chars")
    print(f"[OK] Sections identified: {set(c.section for c in chunks)}")
    
    return True


def test_embedding_generator():
    """Test embedding generation"""
    print("\n" + "="*70)
    print("TEST 4: Embedding Generator")
    print("="*70)
    
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    
    texts = [
        "Python is a programming language",
        "Machine learning is a subset of AI",
        "Data engineering builds data pipelines"
    ]
    
    # Test single embedding
    embedding = generator.encode(texts[0])
    assert embedding is not None, "Failed to generate single embedding"
    assert len(embedding) == 384, f"Wrong embedding dimension: {len(embedding)}"
    
    # Test batch embedding
    embeddings = generator.encode_batch(texts)
    assert len(embeddings) == 3, "Wrong number of embeddings"
    assert all(len(e) == 384 for e in embeddings), "Inconsistent embedding dimensions"
    
    print(f"[OK] Single embedding dimension: {len(embedding)}")
    print(f"[OK] Batch embedding generation: {len(embeddings)} embeddings")
    print(f"[OK] Sample embedding values (first 5): {embedding[:5]}")
    
    return True


def test_full_rag_pipeline():
    """Test complete RAG pipeline"""
    print("\n" + "="*70)
    print("TEST 5: Complete RAG Pipeline")
    print("="*70)
    
    # Create sample resumes
    print("Creating sample resumes...")
    sample_dir = create_sample_resumes()
    
    # Initialize RAG system
    print("Initializing RAG system...")
    import shutil
    if os.path.exists("./test_chroma_db"):
        shutil.rmtree("./test_chroma_db", ignore_errors=True)
    rag_system = ResumeRAGSystem(
        db_path="./test_chroma_db",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=300,
        chunk_overlap=50
    )
    
    # Process resumes
    print("Processing resumes...")
    results = rag_system.process_resumes_from_directory(sample_dir)
    
    assert results['processed'] > 0, "Failed to process any resumes"
    assert len(results['resumes']) > 0, "No resume metadata extracted"
    
    print(f"[OK] Processed: {results['processed']} resumes")
    print(f"[OK] Total chunks: {sum(r['chunks'] for r in results['resumes'])}")
    
    # Get system info
    info = rag_system.get_system_info()
    print(f"[OK] Database documents: {info['database']['total_documents']}")
    print(f"[OK] Embedding dimension: {info['embedding_dimension']}")
    
    # Cleanup
    import gc
    del rag_system
    gc.collect()
    if os.path.exists("./test_chroma_db"):
        shutil.rmtree("./test_chroma_db", ignore_errors=True)
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("RAG SYSTEM - PART A VERIFICATION TESTS")
    print("="*70)
    
    tests = [
        ("Document Loader", test_document_loader),
        ("Metadata Extractor", test_metadata_extractor),
        ("Document Chunker", test_document_chunker),
        ("Embedding Generator", test_embedding_generator),
        ("Full RAG Pipeline", test_full_rag_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            results.append((test_name, f"FAILED: {str(e)}"))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, status in results:
        symbol = "[OK]" if status == "PASSED" else "[FAIL]"
        print(f"{symbol} {test_name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "[SUCCESS] "*3)
        print("ALL TESTS PASSED! Part A implementation is complete and working correctly.")
        print("[SUCCESS] "*3)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
