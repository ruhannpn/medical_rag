import os
import json
import sqlite3
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="medical_rag.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        # Enable foreign keys on connection
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Users Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Patients Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT,
                    age INTEGER,
                    gender TEXT,
                    dob TEXT,
                    visit_date TEXT,
                    diagnosis TEXT,
                    symptoms_json TEXT,
                    allergies_json TEXT,
                    raw_text TEXT,
                    doc_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)

            # Chunks Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
                );
            """)

            # Chat History Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT CHECK(role IN ('user', 'assistant')) NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT,
                    confidence_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)

            conn.commit()

    # =================================================
    # USER OPERATIONS
    # =================================================
    def create_user(self, username, password_hash, salt):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?);",
                    (username.strip().lower(), password_hash, salt)
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_user_credentials(self, username):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, password_hash, salt FROM users WHERE username = ?;",
                (username.strip().lower(),)
            )
            row = cursor.fetchone()
            if row:
                return {"id": row[0], "password_hash": row[1], "salt": row[2]}
            return None

    # =================================================
    # PATIENT OPERATIONS
    # =================================================
    def add_patient(self, user_id, doc_id, raw_text, meta):
        """
        Add a patient report and return patient_id.
        meta should be a dictionary containing:
        name, age, gender, dob, visit_date, diagnosis, symptoms (list), allergies (list)
        """
        symptoms_json = json.dumps(meta.get("symptoms", []))
        allergies_json = json.dumps(meta.get("allergies", []))
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO patients (
                    user_id, name, age, gender, dob, visit_date, diagnosis, 
                    symptoms_json, allergies_json, raw_text, doc_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                user_id,
                meta.get("name"),
                meta.get("age"),
                meta.get("gender"),
                meta.get("dob"),
                meta.get("visit_date"),
                meta.get("diagnosis"),
                symptoms_json,
                allergies_json,
                raw_text,
                doc_id
            ))
            patient_id = cursor.lastrowid
            conn.commit()
            return patient_id

    def add_chunks(self, patient_id, chunks):
        """
        Insert text chunks associated with a patient.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO chunks (patient_id, content) VALUES (?, ?);",
                [(patient_id, chunk) for chunk in chunks]
            )
            conn.commit()

    def get_patients(self, user_id):
        """
        Retrieve all patients for a given user.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, age, gender, dob, visit_date, diagnosis, 
                       symptoms_json, allergies_json, raw_text, doc_id
                FROM patients WHERE user_id = ? ORDER BY name ASC;
            """, (user_id,))
            rows = cursor.fetchall()
            
            patients = []
            for r in rows:
                patients.append({
                    "id": r[0],
                    "name": r[1],
                    "age": r[2],
                    "gender": r[3],
                    "dob": r[4],
                    "visit_date": r[5],
                    "diagnosis": r[6],
                    "symptoms": json.loads(r[7] or "[]"),
                    "allergies": json.loads(r[8] or "[]"),
                    "raw_text": r[9],
                    "doc_id": r[10]
                })
            return patients

    def get_patient_chunks(self, user_id):
        """
        Get all text chunks for all patients belonging to a user.
        Useful for building/loading the RAG index.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chunks.content, patients.doc_id, patients.id
                FROM chunks
                JOIN patients ON chunks.patient_id = patients.id
                WHERE patients.user_id = ?;
            """, (user_id,))
            rows = cursor.fetchall()
            return [{"content": r[0], "doc_id": r[1], "patient_id": r[2]} for r in rows]

    # =================================================
    # CHAT HISTORY OPERATIONS
    # =================================================
    def save_chat_message(self, user_id, role, content, source=None, confidence=None):
        confidence_json = json.dumps(confidence) if confidence else None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (user_id, role, content, source, confidence_json)
                VALUES (?, ?, ?, ?, ?);
            """, (user_id, role, content, source, confidence_json))
            conn.commit()

    def get_chat_history(self, user_id, limit=50):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content, source, confidence_json
                FROM chat_history 
                WHERE user_id = ? 
                ORDER BY id ASC;
            """, (user_id,))
            rows = cursor.fetchall()
            
            history = []
            for r in rows:
                history.append({
                    "role": r[0],
                    "content": r[1],
                    "source": r[2],
                    "confidence": json.loads(r[3]) if r[3] else None
                })
            return history

    def clear_chat_history(self, user_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history WHERE user_id = ?;", (user_id,))
            conn.commit()

    # =================================================
    # DATA PURGING
    # =================================================
    def clear_user_data(self, user_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Cascade deletes handle patients, chunks, and chat history automatically
            cursor.execute("DELETE FROM patients WHERE user_id = ?;", (user_id,))
            cursor.execute("DELETE FROM chat_history WHERE user_id = ?;", (user_id,))
            conn.commit()
