from __future__ import annotations
import os
import hashlib
import streamlit as st
from typing import Tuple, Optional, Dict
from database import DatabaseManager

class AuthManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    @staticmethod
    def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
        """
        Hash password using PBKDF2-HMAC-SHA256.
        If salt is not provided, generate a new one.
        Returns (password_hash, salt).
        """
        if salt is None:
            salt = os.urandom(16).hex()
        
        # PBKDF2 with 100,000 iterations
        pw_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        return pw_hash, salt

    def register_user(self, username, password) -> bool:
        """
        Register a new user with hashed credentials.
        """
        if not username or not password:
            return False
        
        # Check if user already exists
        existing = self.db.get_user_credentials(username)
        if existing:
            return False
            
        pw_hash, salt = self.hash_password(password)
        user_id = self.db.create_user(username, pw_hash, salt)
        return user_id is not None

    def authenticate_user(self, username, password) -> Optional[Dict]:
        """
        Authenticate user and return their user_id if successful.
        """
        if not username or not password:
            return None
            
        creds = self.db.get_user_credentials(username)
        if not creds:
            return None
            
        # Hash entered password with stored salt
        pw_hash, _ = self.hash_password(password, creds["salt"])
        
        # Secure comparison
        if pw_hash == creds["password_hash"]:
            return {"id": creds["id"], "username": username}
        return None

    def render_auth_page(self):
        """
        Renders an elegant glassmorphism login / registration card in Streamlit.
        """
        # Inject styling for auth forms
        st.markdown("""
        <style>
        .auth-container {
            max-width: 450px;
            margin: 80px auto;
            padding: 2.5rem;
            background: linear-gradient(135deg, #0d1929 0%, #0a1520 100%);
            border: 1px solid #1a2e47;
            border-radius: 16px;
            box-shadow: 0 0 40px rgba(0, 163, 255, 0.08);
            position: relative;
            overflow: hidden;
        }
        .auth-container::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #00a3ff, #0044cc);
        }
        .auth-title {
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            color: #ffffff;
            margin-bottom: 0.5rem;
        }
        .auth-title span {
            background: linear-gradient(135deg, #00a3ff 0%, #0066ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .auth-sub {
            font-size: 0.88rem;
            color: #4a6080;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

        if "auth_mode" not in st.session_state:
            st.session_state.auth_mode = "login"

        # Center layout
        _, center_col, _ = st.columns([1, 2, 1])

        with center_col:
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            
            if st.session_state.auth_mode == "login":
                st.markdown('<div class="auth-title">Welcome to <span>MedRAG</span></div>', unsafe_allow_html=True)
                st.markdown('<div class="auth-sub">Sign in to query clinical medical records</div>', unsafe_allow_html=True)
                
                with st.form("login_form", clear_on_submit=False):
                    username = st.text_input("Username", key="login_username", placeholder="Enter username")
                    password = st.text_input("Password", type="password", key="login_password", placeholder="Enter password")
                    submit = st.form_submit_button("Sign In", use_container_width=True)
                    
                    if submit:
                        if not username or not password:
                            st.error("Please fill in all fields.")
                        else:
                            user = self.authenticate_user(username, password)
                            if user:
                                st.session_state.logged_in = True
                                st.session_state.user_id = user["id"]
                                st.session_state.username = user["username"]
                                st.session_state.chat_history = []
                                st.session_state.indexed = False
                                st.success("Authentication successful!")
                                st.rerun()
                            else:
                                st.error("Invalid username or password.")
                                
                # Link to Register
                if st.button("New to MedRAG? Register here", use_container_width=True):
                    st.session_state.auth_mode = "register"
                    st.rerun()
                    
            else:
                st.markdown('<div class="auth-title">Create <span>Account</span></div>', unsafe_allow_html=True)
                st.markdown('<div class="auth-sub">Register a secure clinical account</div>', unsafe_allow_html=True)
                
                with st.form("register_form", clear_on_submit=False):
                    username = st.text_input("Username", key="reg_username", placeholder="Choose username")
                    password = st.text_input("Password", type="password", key="reg_password", placeholder="Choose password")
                    confirm_pw = st.text_input("Confirm Password", type="password", key="reg_confirm", placeholder="Confirm password")
                    submit = st.form_submit_button("Create Account", use_container_width=True)
                    
                    if submit:
                        if not username or not password:
                            st.error("Please fill in all fields.")
                        elif len(password) < 6:
                            st.error("Password must be at least 6 characters.")
                        elif password != confirm_pw:
                            st.error("Passwords do not match.")
                        else:
                            success = self.register_user(username, password)
                            if success:
                                st.success("Account created successfully! Please log in.")
                                st.session_state.auth_mode = "login"
                                st.rerun()
                            else:
                                st.error("Username already exists.")
                                
                # Link to Login
                if st.button("Already have an account? Sign In", use_container_width=True):
                    st.session_state.auth_mode = "login"
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
