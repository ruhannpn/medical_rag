import os
import sys
import shutil
import tempfile
import numpy as np

# Add src and project root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, _ROOT)

from database import DatabaseManager
from auth import AuthManager
from extractor import ClinicalExtractor
from retriever import RAGIndex
from embedding import load_embedding_model

def run_tests():
    print("=== Starting Integration Tests ===")
    
    # 1. Database Setup
    db_path = "test_medical_rag.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    index_dir = "test_data/indices"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir, ignore_errors=True)
        
    db = DatabaseManager(db_path=db_path)
    auth = AuthManager(db)
    
    # 2. Test User Authentication
    print("1. Testing User Registration & Authentication...")
    username = "test_doc"
    password = "secure_password_123"
    
    # Registration
    assert auth.register_user(username, password) is True, "Failed to register user"
    assert auth.register_user(username, password) is False, "Duplicate user registration should fail"
    
    # Authentication
    user = auth.authenticate_user(username, password)
    assert user is not None, "Failed to authenticate user"
    assert user["username"] == username, "Username mismatch"
    user_id = user["id"]
    
    # Invalid password
    assert auth.authenticate_user(username, "wrong_pw") is None, "Wrong password authenticated successfully"
    print("   [OK] Authentication tests passed")

    # 3. Test Clinical Entity Extraction (Regex)
    print("2. Testing Regex Extraction...")
    mock_report = """
    Patient Report
    Name: John Doe
    Age: 45
    Gender: Male
    DOB: 1981-05-12
    Visit Date: 2026-07-06
    Diagnosis: Type 2 Diabetes Mellitus
    Medications:
    - Metformin 500mg BID
    - Lisinopril 10mg QD
    Symptoms:
    - Frequent urination
    - Fatigue
    Allergies: NKDA
    """
    
    meta = ClinicalExtractor.extract_with_regex(mock_report)
    assert meta["name"] == "John Doe", f"Expected 'John Doe', got {meta['name']}"
    assert meta["age"] == 45, f"Expected 45, got {meta['age']}"
    assert meta["gender"] == "Male", f"Expected 'Male', got {meta['gender']}"
    assert meta["dob"] == "1981-05-12", f"Expected '1981-05-12', got {meta['dob']}"
    assert meta["visit_date"] == "2026-07-06", f"Expected '2026-07-06', got {meta['visit_date']}"
    assert meta["diagnosis"] == "Type 2 Diabetes Mellitus", f"Expected diagnosis, got {meta['diagnosis']}"
    assert "Metformin 500mg BID" in meta["medications"], "Metformin missing"
    assert "Lisinopril 10mg QD" in meta["medications"], "Lisinopril missing"
    assert "Frequent urination" in meta["symptoms"], "Symptoms missing"
    assert len(meta["allergies"]) == 0, "Expected empty allergies list"
    print("   [OK] Regex extraction tests passed")

    # 4. Test DB Insertions & Chunks
    print("3. Testing Patient Database & Chunks insertion...")
    patient_id = db.add_patient(user_id, "john_doe.pdf", mock_report, meta)
    assert patient_id is not None, "Failed to save patient"
    
    mock_chunks = [
        "Patient John Doe is a 45-year-old male with diabetes.",
        "He is taking Metformin 500mg twice daily and Lisinopril 10mg daily."
    ]
    db.add_chunks(patient_id, mock_chunks)
    
    patients = db.get_patients(user_id)
    assert len(patients) == 1, "Patient not found in DB"
    assert patients[0]["name"] == "John Doe", "Patient name mismatch"
    
    chunks = db.get_patient_chunks(user_id)
    assert len(chunks) == 2, "Chunks not found in DB"
    assert chunks[0]["content"] == mock_chunks[0], "Chunk content mismatch"
    print("   [OK] Database insertion tests passed")

    # 5. Test RAGIndex & FAISS Serialization
    print("4. Testing RAGIndex Build, Search, Save, & Load...")
    embed_model = load_embedding_model()
    rag_index = RAGIndex(index_dir=index_dir)
    
    # Build
    db_chunks = db.get_patient_chunks(user_id)
    rag_index.build_or_update(db_chunks, embed_model)
    assert not rag_index.is_empty(), "RAG index is empty after build"
    
    # Save
    rag_index.save(user_id)
    faiss_path = os.path.join(index_dir, str(user_id), "index.faiss")
    assert os.path.exists(faiss_path), "FAISS index file not created"
    
    # Search
    query = "What drugs is John Doe taking?"
    top_indices = rag_index.search(query, embed_model, top_k=2)
    assert len(top_indices) >= 1, "Expected search results"
    matched_chunks = [rag_index.chunks[i]["content"] for i in top_indices]
    assert any("Metformin" in c for c in matched_chunks), f"Expected Metformin in matches: {matched_chunks}"
    
    # Re-Load in a new index instance
    new_rag_index = RAGIndex(index_dir=index_dir)
    new_rag_index.load(user_id, db_chunks)
    assert not new_rag_index.is_empty(), "Reloaded RAG index is empty"
    
    # Validate search on reloaded index matches
    new_top_indices = new_rag_index.search(query, embed_model, top_k=2)
    assert new_top_indices == top_indices, "Search results differ on reloaded index"
    print("   [OK] RAG index save and reload tests passed")

    # 6. Purging Data
    print("5. Testing User Data Purging...")
    db.clear_user_data(user_id)
    rag_index.clear(user_id)
    
    assert len(db.get_patients(user_id)) == 0, "Patients not cleared from DB"
    assert len(db.get_patient_chunks(user_id)) == 0, "Chunks not cleared from DB"
    assert not os.path.exists(faiss_path), "FAISS index file not deleted on clear"
    print("   [OK] Purging tests passed")
    
    # Cleanup files
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(index_dir):
        shutil.rmtree(os.path.dirname(index_dir), ignore_errors=True)
        
    print("\n=== ALL INTEGRATION TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
