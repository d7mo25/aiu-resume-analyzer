from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
import base64
import pandas as pd
import io
import re
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import time
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage, exceptions
import requests
import json
import random
from datetime import datetime, timedelta
import jwt
import os
from pathlib import Path
import tempfile
import matplotlib
import bcrypt
import urllib.request
import urllib.parse
import csv
import logging
matplotlib.use('Agg')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AIU Smart Resume Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# Static files
# Vercel serves files from the project root differently than Railway/Docker.
# We still mount /static so the dev server (uvicorn) works locally.
# On Vercel the route in vercel.json forwards /static/* to this app too.
# ---------------------------------------------------------------------------
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Configuration
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "AIzaSyBSBSkxJoya3yk4JA8wXp6BgF99GQJplrs")
BUCKET_NAME = os.getenv("FIREBASE_STORAGE_BUCKET", "resume-analyzer-d58fd")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secure-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
PORT = int(os.getenv("PORT", 8000))

# Firebase initialization
firebase_initialized = False
db = None
bucket = None
firebase_app = None

def initialize_firebase():
    global firebase_initialized, db, bucket, firebase_app

    # Already initialised (e.g. hot-reload / second request on same worker)
    try:
        firebase_app = firebase_admin.get_app()
        firebase_initialized = True
        db = firestore.client()
        bucket = storage.bucket(BUCKET_NAME)
        logger.info("✅ Firebase already initialized")
        return True
    except ValueError:
        pass

    try:
        raw = os.getenv("FIREBASE_SERVICE_ACCOUNT", "").strip()

        if not raw:
            # ----------------------------------------------------------------
            # FIREBASE_SERVICE_ACCOUNT is not set — give a clear action item.
            # ----------------------------------------------------------------
            logger.error(
                "❌ FIREBASE_SERVICE_ACCOUNT environment variable is not set.\n"
                "   Go to Firebase Console → Project Settings → Service Accounts\n"
                "   → Generate new private key, then paste the entire JSON into\n"
                "   Vercel → Settings → Environment Variables as FIREBASE_SERVICE_ACCOUNT."
            )
            raise Exception("FIREBASE_SERVICE_ACCOUNT env var is missing")

        # Vercel sometimes double-escapes the private key newlines (\\n → \n).
        # Fix that before parsing so json.loads always succeeds.
        raw = raw.replace("\\n", "\n")

        try:
            service_account_info = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(f"❌ FIREBASE_SERVICE_ACCOUNT is not valid JSON: {exc}")
            logger.error(
                "   Make sure you pasted the complete .json file content, "
                "including the outer { } braces."
            )
            raise Exception("FIREBASE_SERVICE_ACCOUNT contains invalid JSON") from exc

        required_keys = {"type", "project_id", "private_key", "client_email"}
        missing = required_keys - service_account_info.keys()
        if missing:
            logger.error(f"❌ FIREBASE_SERVICE_ACCOUNT JSON is missing keys: {missing}")
            raise Exception(f"Service account JSON missing required fields: {missing}")

        cred = credentials.Certificate(service_account_info)
        logger.info(f"✅ Firebase credentials loaded for project: {service_account_info.get('project_id')}")

        firebase_app = firebase_admin.initialize_app(cred, {
            "storageBucket": BUCKET_NAME,
            "projectId": service_account_info.get("project_id", "resume-analyzer-d58fd"),
        })

        db = firestore.client()
        bucket = storage.bucket(BUCKET_NAME)
        
        firebase_initialized = True
        create_default_admin()
        logger.info("✅ Firebase initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Firebase initialization failed: {e}")
        logger.warning("⚠️ Falling back to development mode (in-memory storage)")
        firebase_initialized = False
        return False

def create_default_admin():
    if not db:
        return
    
    try:
        admins_ref = db.collection('admins')
        query = admins_ref.where('email', '==', "admin@aiu.edu.my").limit(1).get()
        
        if not query:
            hashed_password = bcrypt.hashpw("Admin123!".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            admin_doc = {
                "email": "admin@aiu.edu.my",
                "password_hash": hashed_password,
                "created_at": datetime.now(),
                "last_login": None,
                "created_by": "system"
            }
            admins_ref.add(admin_doc)
            logger.info(f"✅ Created default admin: admin@aiu.edu.my / Admin123!")
    except Exception as e:
        logger.error(f"❌ Error creating default admin: {str(e)}")

# Initialize Firebase
firebase_status = initialize_firebase()

if not firebase_initialized:
    logger.warning("⚠️ Firebase not available, using in-memory storage")
    IN_MEMORY_ADMIN = {
        "email": "admin@aiu.edu.my",
        "password_hash": bcrypt.hashpw("Admin123!".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        "created_at": datetime.now(),
        "last_login": None
    }
    IN_MEMORY_USERS = {}
    IN_MEMORY_RESUMES = {}
    IN_MEMORY_ADMINS = {
        "dev_admin": IN_MEMORY_ADMIN
    }

# Pydantic models
class TokenRequest(BaseModel):
    token: str

class UserRegistration(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    confirm_password: str
    phone: str
    agree_terms: bool

class UserLogin(BaseModel):
    email: str
    password: str
    rememberMe: bool = False

class AdminLogin(BaseModel):
    email: str
    password: str

class AdminCreate(BaseModel):
    email: EmailStr
    password: str
    confirm_password: str

class CheckEmailRequest(BaseModel):
    email: EmailStr

class EmailResetRequest(BaseModel):
    email: EmailStr

class PasswordResetRequest(BaseModel):
    email: EmailStr
    action_url: Optional[str] = None

class PasswordResetComplete(BaseModel):
    oobCode: str
    newPassword: str

class DatabasePasswordUpdate(BaseModel):
    email: EmailStr
    newPassword: str

class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    bio: Optional[str] = None

class UserProfileResponse(BaseModel):
    user_id: str
    email: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    bio: Optional[str] = None
    created_at: str
    last_login: Optional[str] = None
    role: str

class DashboardStats(BaseModel):
    total_users: int
    total_resumes: int
    average_score: float

# Updated Keyword Categories aligned with Streamlit version
KEYWORD_CATEGORIES = {
    "Technical Skills": [
        "LMS", "Moodle", "Blackboard", "management",
        "E-learning platforms", "SPSS", "Statistical software", "Excel", "Microsoft Excel",
        "Microsoft Office", "Qualitative analysis", "Research methodology",
        "Teaching tools", "Academic software", "Learning analytics"
    ],
    "Soft Skills": [
        "Communication", "Leadership", "Teamwork", "Adaptability", "Problem-solving"
    ],
    "Work Experience": [
        "University lecturer", "Professor role",
        "Course development", "Lecture preparation",
        "Student mentorship", "Research supervision", "Teaching assistant",
        "Administrative experience", "Technical support", "Project management", "Office management",
        "Data analysis", "Systems administrator", "Technical documentation",
        "Operations management", "Fresh graduate", "Internship",
        "Industrial training"
    ],
    "Language Proficiency": [
        "English"
    ],
    "Achievements": [
        "Research grants", "Employee recognition", "Dean's list", "Competition achievements",
        "Certifications"
    ],
    "Publications": [
        "Peer-reviewed journal articles", "Conference proceedings", "Books", "Blog",
        "Academic articles", "Research papers"
    ],
    "Candidate Profile": [
        "Full Name", "Email", "Phone", "Address", "LinkedIn"
    ]
}

SECTION_PATTERNS = {
    "Candidate Profile": [
        r'\b(personal\s+information|contact\s+information|profile|about\s+me|summary)\b',
        r'\b(name|email|phone|address|linkedin)\b'
    ],
    "Education": [
        r'\b(education|academic\s+background|qualifications|degree|university|college|school)\b',
        r'\b(bachelor|master|phd|diploma|certificate|graduation)\b'
    ],
    "Skills": [
        r'\b(skills|competencies|technical\s+skills|core\s+competencies|abilities)\b',
        r'\b(programming|software|tools|languages|proficiency)\b'
    ],
    "Experience": [
        r'\b(experience|work\s+experience|employment|career|professional\s+experience)\b',
        r'\b(job|position|role|internship|worked\s+at|employed)\b'
    ]
}

REQUIRED_SECTIONS = [
    "Candidate Profile", "Education", "Skills", "Experience"
]

MAX_SCORES = {
    "Technical Skills": 10,
    "Soft Skills": 15,
    "Work Experience": 10,
    "Language Proficiency": 10,
    "Achievements": 5,
    "Publications": 10,
    "Sections Presence": 25,
    "Candidate Profile": 15
}

# Utility functions
def create_jwt_token(user_data: dict) -> str:
    payload = {
        "user_id": user_data.get("user_id"),
        "email": user_data.get("email"),
        "role": user_data.get("role", "user"),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return verify_jwt_token(credentials.credentials)

def pdf_reader(file_buffer):
    try:
        resource_manager = PDFResourceManager()
        output_string = io.StringIO()
        laparams = LAParams(char_margin=2.0, line_margin=0.5, boxes_flow=0.5, detect_vertical=True)
        converter = TextConverter(resource_manager, output_string, laparams=laparams)
        interpreter = PDFPageInterpreter(resource_manager, converter)
        file_buffer.seek(0)
        
        for page in PDFPage.get_pages(file_buffer, caching=True, check_extractable=True, maxpages=0):
            interpreter.process_page(page)
                
        converter.close()
        text = output_string.getvalue()
        output_string.close()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_full_name(text: str) -> str:
    """Extract full name using regex — no spaCy required.

    Scans the first 20 non-empty lines for a line that looks like a person
    name: 2–4 title-cased words, no digits, no common section headers.
    """
    SECTION_HEADERS = {
        "curriculum vitae", "resume", "cv", "work experience", "education",
        "skills", "summary", "objective", "references", "contact information",
        "personal information", "professional summary", "career objective",
        "profile", "about me", "cover letter",
    }
    # 2–4 words, each starting uppercase, only letters/hyphens/apostrophes
    name_re = re.compile(r"^[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3}$")

    for line in text.splitlines()[:20]:
        line = line.strip()
        if not line:
            continue
        if line.lower() in SECTION_HEADERS:
            continue
        if re.search(r"[@/\\0-9]", line):   # definitely not a name
            continue
        if name_re.match(line):
            return line
    return "N/A"

def extract_basic_info_from_text(text):
    name = extract_full_name(text)
    
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    
    phone_pattern = re.compile(r'(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}|\+?6?0?1?[-.\s]?[0-9]{1,2}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{4})')
    phone_matches = list(phone_pattern.finditer(text))
    phone = "N/A"
    if phone_matches:
        earliest_match = min(phone_matches, key=lambda m: m.start())
        phone = earliest_match.group(0).strip()
    
    address = "N/A"
    linkedin_match = re.search(r"(https?://)?(www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+", text, re.IGNORECASE)
    github_match = re.search(r"(https?://)?(www\.)?github\.com/[a-zA-Z0-9_-]+", text, re.IGNORECASE)
    
    return {
        'name': name,
        'email': email_match.group(0) if email_match else "N/A",
        'phone': phone,
        'address': address,
        'linkedin': linkedin_match.group(0) if linkedin_match else "N/A",
        'github': github_match.group(0) if github_match else "N/A",
    }

def extract_keywords(text, keywords):
    text_lower = text.lower()
    found = []
    for kw in keywords:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.append(kw)
    return found

def extract_candidate_profile_keywords(basic_info):
    profile_keywords = []
    if basic_info.get('name') and basic_info['name'] != "N/A":
        profile_keywords.append("Full Name")
    if basic_info.get('email') and basic_info['email'] != "N/A":
        profile_keywords.append("Email")
    if basic_info.get('phone') and basic_info['phone'] != "N/A":
        profile_keywords.append("Phone")
    if basic_info.get('address') and basic_info['address'] != "N/A":
        profile_keywords.append("Address")
    if basic_info.get('linkedin') and basic_info['linkedin'] != "N/A":
        profile_keywords.append("LinkedIn")
    return profile_keywords

def detect_sections_presence(text):
    text_lower = text.lower()
    found_sections = []
    for section, patterns in SECTION_PATTERNS.items():
        section_found = False
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                section_found = True
                break
        if section_found:
            found_sections.append(section)
    return found_sections

def calculate_ats_score(extraction_results, sections_found):
    category_scores = {}
    for category, found_keywords in extraction_results.items():
        max_score = MAX_SCORES.get(category, 0)
        found_count = len(found_keywords)
        total_possible = len(KEYWORD_CATEGORIES.get(category, []))
        
        if total_possible > 0:
            category_scores[category] = round((found_count / total_possible) * max_score, 2)
        else:
            category_scores[category] = 0

    sections_score = (len(sections_found) / len(REQUIRED_SECTIONS)) * MAX_SCORES["Sections Presence"]
    category_scores["Sections Presence"] = round(sections_score, 2)

    total_score = sum(category_scores.values())
    return round(min(total_score, 100), 2), category_scores

def create_keyword_chart(extraction_results, sections_found, applied_role: str):
    categories = list(extraction_results.keys()) + ["Sections Presence"]
    found_counts = [len(extraction_results[cat]) for cat in extraction_results.keys()] + [len(sections_found)]
    total_keywords = [len(KEYWORD_CATEGORIES[cat]) for cat in extraction_results.keys()] + [len(REQUIRED_SECTIONS)]
    
    _, category_scores = calculate_ats_score(extraction_results, sections_found)
    scores = [category_scores[cat] for cat in categories]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(categories))

    ax.barh(y_pos, found_counts, color='#667eea', alpha=0.8, label='Keywords Found')
    ax.barh(y_pos, total_keywords, color='none', edgecolor='gray', linewidth=1.5, label='Total Possible')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    ax.set_title(f'Resume Analysis Results for {applied_role}', fontsize=16, fontweight='bold')

    max_scores_list = [MAX_SCORES[cat] for cat in categories]
    for i, (score, max_score) in enumerate(zip(scores, max_scores_list)):
        ax.text(max(found_counts[i], 1) + 0.3, i, f'{score} / {max_score}', va='center', fontsize=9, color='black')

    ax.legend()
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return chart_base64

def get_improvement_suggestions(extraction_results, applied_role: str):
    suggestions = {}
    for category, keywords in KEYWORD_CATEGORIES.items():
        found_set = set(extraction_results.get(category, []))
        all_keywords = set(keywords)
        missing = all_keywords - found_set
        if missing:
            suggestions[category] = list(missing)[:5]
    return suggestions

async def save_resume_to_firestore(user_id: str, basic_info: dict, file_name: str, file_url: str, applied_role: str):
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    upload_date = datetime.now().strftime("%Y-%m-%d")
    
    resume_data = {
        "user_id": user_id,
        "name": basic_info.get('name', 'N/A'),
        "email": basic_info.get('email', 'N/A'),
        "phone": basic_info.get('phone', 'N/A'),
        "address": basic_info.get('address', 'N/A'),
        "linkedin": basic_info.get('linkedin', 'N/A'),
        "github": basic_info.get('github', 'N/A'),
        "upload_date": upload_date,
        "file_name": file_name,
        "file_url": file_url,
        "applied_role": applied_role,
        "status": "uploaded",
        "ats_score": None,
        "analysis_date": None,
        "analysis_data": None
    }
    
    if db:
        try:
            doc_ref = db.collection('resumes').add(resume_data)
            return doc_ref[1].id
        except Exception as e:
            logger.error(f"Error saving to Firestore: {str(e)}")
            return None
    else:
        doc_id = f"resume_{len(IN_MEMORY_RESUMES) + 1}_{int(time.time())}"
        resume_data['id'] = doc_id
        IN_MEMORY_RESUMES[doc_id] = resume_data
        return doc_id

async def update_resume_with_analysis(resume_id: str, ats_score: float, extraction_results: dict, suggestions: dict, chart_base64: str = None):
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    analysis_data = {
        "extraction_results": extraction_results,
        "suggestions": suggestions,
        "chart": chart_base64
    }
    update_data = {
        "ats_score": ats_score,
        "analysis_date": current_date,
        "analysis_data": analysis_data,
        "status": "analyzed"
    }
    
    if db:
        try:
            db.collection('resumes').document(resume_id).update(update_data)
            return True
        except Exception as e:
            logger.error(f"Error updating resume analysis: {str(e)}")
            return False
    else:
        if resume_id in IN_MEMORY_RESUMES:
            IN_MEMORY_RESUMES[resume_id].update(update_data)
            return True
        return False

async def upload_to_firebase_storage(content: bytes, filename: str, content_type: str) -> str:
    if not bucket:
        raise HTTPException(status_code=500, detail="File storage service unavailable")
    
    try:
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{filename}"
        blob_path = f"resumes/{safe_filename}"
        
        blob = bucket.blob(blob_path)
        blob.metadata = {
            'contentType': content_type,
            'uploadedAt': datetime.now().isoformat(),
            'originalFilename': filename
        }
        
        blob.upload_from_string(content, content_type=content_type)
        blob.make_public()
        return blob.public_url
        
    except Exception as e:
        logger.error(f"❌ Firebase Storage upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

async def ensure_user_in_firebase_auth(email: str, password: str = None):
    if not firebase_initialized:
        return None
    
    try:
        user_record = auth.get_user_by_email(email)
        return user_record
    except auth.UserNotFoundError:
        try:
            if password:
                user_record = auth.create_user(email=email, password=password, email_verified=True)
            else:
                import secrets, string
                temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))
                user_record = auth.create_user(email=email, password=temp_password, email_verified=True)
            return user_record
        except Exception as create_error:
            logger.error(f"❌ Failed to create user in Firebase Auth: {create_error}")
            return None
    except Exception as e:
        logger.error(f"❌ Error checking Firebase Auth user: {e}")
        return None

# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/resume-upload", response_class=HTMLResponse)
async def resume_upload_page(request: Request):
    return templates.TemplateResponse("resume-upload.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/admin-login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    return templates.TemplateResponse("admin-login.html", {"request": request})

@app.get("/admin-dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    return templates.TemplateResponse("admin-dashboard.html", {"request": request})

@app.get("/profile", response_class=HTMLResponse)
async def user_profile_page(request: Request):
    return templates.TemplateResponse("profile.html", {"request": request})

@app.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot-password.html", {"request": request})

# ---------------------------------------------------------------------------
# API routes — identical to original, no changes needed below
# ---------------------------------------------------------------------------

@app.post("/api/forgot-password/check-email")
async def check_email_exists(email_data: CheckEmailRequest):
    try:
        if db:
            users_ref = db.collection('users')
            query = users_ref.where('email', '==', email_data.email).limit(1).get()
            exists = len(query) > 0
            if exists:
                await ensure_user_in_firebase_auth(email_data.email)
                return {"exists": True, "message": "Account found", "email": email_data.email}
            return {"exists": False, "message": "No account found with this email address"}
        else:
            exists = any(user['email'] == email_data.email for user in IN_MEMORY_USERS.values())
            return {"exists": exists, "message": "Account found" if exists else "No account found with this email address"}
    except Exception as e:
        logger.error(f"❌ Error checking email: {str(e)}")
        raise HTTPException(status_code=500, detail="Error checking email address")

@app.post("/api/forgot-password/send-reset")
async def send_password_reset_email(email_data: PasswordResetRequest, request: Request):
    try:
        if db:
            users_ref = db.collection('users')
            query = users_ref.where('email', '==', email_data.email).limit(1).get()
            if len(query) == 0:
                raise HTTPException(status_code=404, detail="No account found with this email address")
        else:
            exists = any(user['email'] == email_data.email for user in IN_MEMORY_USERS.values())
            if not exists:
                raise HTTPException(status_code=404, detail="No account found with this email address")
        
        await ensure_user_in_firebase_auth(email_data.email)
        
        if firebase_initialized:
            try:
                action_code_settings = auth.ActionCodeSettings(
                    url=email_data.action_url or f"{os.getenv('FRONTEND_URL', request.url.scheme + '://' + request.url.netloc)}/forgot-password",
                    handle_code_in_app=False
                )
                reset_link = auth.generate_password_reset_link(email_data.email, action_code_settings)
                return {"success": True, "message": "Password reset email sent successfully", "email": email_data.email}
            except Exception as firebase_error:
                logger.error(f"❌ Firebase Admin SDK error: {str(firebase_error)}")
                return {"success": True, "message": "Please use client-side reset", "use_client_auth": True, "email": email_data.email}
        else:
            return {"success": True, "message": "Password reset email sent successfully (development mode)", "email": email_data.email}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error sending password reset: {str(e)}")
        raise HTTPException(status_code=500, detail="Error sending password reset email")

@app.post("/api/forgot-password/update-database")
async def update_database_password(password_data: DatabasePasswordUpdate):
    try:
        email = password_data.email
        new_password = password_data.newPassword
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        if db:
            users_ref = db.collection('users')
            query = users_ref.where('email', '==', email).limit(1).get()
            if len(query) == 0:
                raise HTTPException(status_code=404, detail="User not found in database")
            user_doc = query[0]
            users_ref.document(user_doc.id).update({
                "password_hash": hashed_password,
                "password_updated_at": datetime.now(),
                "last_login": None
            })
        else:
            user_found = False
            for user_id, user_data in IN_MEMORY_USERS.items():
                if user_data['email'] == email:
                    user_data['password_hash'] = hashed_password
                    user_data['password_updated_at'] = datetime.now()
                    user_data['last_login'] = None
                    user_found = True
                    break
            if not user_found:
                raise HTTPException(status_code=404, detail="User not found in database")
        
        return {"success": True, "message": "Database password updated successfully", "email": email}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating database password: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating database password")

@app.post("/api/register")
async def register_user(user_data: UserRegistration):
    if user_data.password != user_data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    
    hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user_doc = {
        "full_name": user_data.full_name,
        "email": user_data.email,
        "phone": user_data.phone,
        "password_hash": hashed_password,
        "created_at": datetime.now(),
        "role": "user",
        "last_login": None,
        "address": None,
        "linkedin": None,
        "github": None,
        "bio": None
    }
    
    if db:
        try:
            users_ref = db.collection('users')
            query = users_ref.where('email', '==', user_data.email).limit(1).stream()
            if any(query):
                raise HTTPException(status_code=400, detail="Email already registered")
            doc_ref = users_ref.add(user_doc)
            user_id = doc_ref[1].id
            await ensure_user_in_firebase_auth(user_data.email, user_data.password)
        except Exception as e:
            logger.error(f"❌ Error creating user: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")
    else:
        if any(u['email'] == user_data.email for u in IN_MEMORY_USERS.values()):
            raise HTTPException(status_code=400, detail="Email already registered")
        user_id = f"dev_user_{len(IN_MEMORY_USERS) + 1}"
        IN_MEMORY_USERS[user_id] = user_doc
    
    user_info = {"user_id": user_id, "email": user_data.email, "role": "user"}
    token = create_jwt_token(user_info)
    return {"message": "User registered successfully", "token": token, "user": user_info}

@app.post("/api/login")
async def user_login(user_data: UserLogin):
    try:
        if db:
            users_ref = db.collection('users')
            docs = users_ref.where('email', '==', user_data.email).limit(1).get()
            if len(docs) == 0:
                raise HTTPException(status_code=401, detail="Invalid email or password")
            user_doc = docs[0].to_dict()
            user_id = docs[0].id
        else:
            user_found = None
            for uid, user in IN_MEMORY_USERS.items():
                if user['email'] == user_data.email:
                    user_found = user
                    user_id = uid
                    break
            if user_found is None:
                raise HTTPException(status_code=401, detail="Invalid email or password")
            user_doc = user_found
        
        if not bcrypt.checkpw(user_data.password.encode('utf-8'), user_doc['password_hash'].encode('utf-8')):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        if db:
            db.collection('users').document(user_id).update({"last_login": datetime.now()})
        else:
            IN_MEMORY_USERS[user_id]["last_login"] = datetime.now()
        
        user_info = {"user_id": user_id, "email": user_doc['email'], "role": user_doc.get('role', 'user')}
        token = create_jwt_token(user_info)
        return {"message": "Login successful", "token": token, "user": user_info}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.post("/api/admin/login")
async def admin_login(admin_data: AdminLogin):
    try:
        if db:
            admins_ref = db.collection('admins')
            docs = admins_ref.where('email', '==', admin_data.email).limit(1).get()
            if len(docs) == 0:
                raise HTTPException(status_code=401, detail="Invalid admin credentials")
            doc = docs[0]
            admin_doc = doc.to_dict()
            admin_doc['id'] = doc.id
            if not bcrypt.checkpw(admin_data.password.encode('utf-8'), admin_doc['password_hash'].encode('utf-8')):
                raise HTTPException(status_code=401, detail="Invalid admin credentials")
            admins_ref.document(admin_doc['id']).update({"last_login": datetime.now()})
            admin_info = {"user_id": admin_doc['id'], "email": admin_data.email, "role": "admin"}
            token = create_jwt_token(admin_info)
            return {"message": "Admin login successful", "token": token, "user": admin_info}
        else:
            if not IN_MEMORY_ADMIN:
                raise HTTPException(status_code=500, detail="Admin system not initialized")
            if admin_data.email != IN_MEMORY_ADMIN['email']:
                raise HTTPException(status_code=401, detail="Invalid admin credentials")
            if not bcrypt.checkpw(admin_data.password.encode('utf-8'), IN_MEMORY_ADMIN['password_hash'].encode('utf-8')):
                raise HTTPException(status_code=401, detail="Invalid admin credentials")
            admin_info = {"user_id": "dev_admin", "email": admin_data.email, "role": "admin"}
            token = create_jwt_token(admin_info)
            return {"message": "Admin login successful (development mode)", "token": token, "user": admin_info}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Admin login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.put("/api/user/profile")
async def update_user_profile(profile_data: UserProfileUpdate, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    update_fields = {k: v for k, v in profile_data.dict().items() if v is not None}
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    if db:
        try:
            db.collection('users').document(user_id).update(update_fields)
            return {"message": "Profile updated successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating profile: {str(e)}")
    else:
        if user_id not in IN_MEMORY_USERS:
            raise HTTPException(status_code=404, detail="User not found")
        IN_MEMORY_USERS[user_id].update(update_fields)
        return {"message": "Profile updated successfully"}

@app.get("/api/user/profile", response_model=UserProfileResponse)
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    
    if db:
        try:
            user_doc = db.collection('users').document(user_id).get()
            if not user_doc.exists:
                raise HTTPException(status_code=404, detail="User not found")
            user_data = user_doc.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching profile: {str(e)}")
    else:
        if user_id not in IN_MEMORY_USERS:
            raise HTTPException(status_code=404, detail="User not found")
        user_data = IN_MEMORY_USERS[user_id]
    
    return {
        "user_id": user_id,
        "email": user_data.get('email', ''),
        "full_name": user_data.get('full_name', ''),
        "phone": user_data.get('phone', ''),
        "address": user_data.get('address', ''),
        "linkedin": user_data.get('linkedin', ''),
        "github": user_data.get('github', ''),
        "bio": user_data.get('bio', ''),
        "created_at": user_data.get('created_at', datetime.now()).isoformat(),
        "last_login": user_data.get('last_login', datetime.now()).isoformat() if user_data.get('last_login') else None,
        "role": user_data.get('role', 'user')
    }

@app.get("/api/user/resumes")
async def get_user_resumes(current_user: dict = Depends(get_current_user)):
    if db:
        try:
            docs = db.collection('resumes').where('user_id', '==', current_user['user_id']).stream()
            results = []
            for doc in docs:
                resume_data = doc.to_dict()
                resume_data['id'] = doc.id
                results.append(resume_data)
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching user resumes: {str(e)}")
    else:
        return [r for r in IN_MEMORY_RESUMES.values() if r.get('user_id') == current_user['user_id']]

@app.post("/api/resume/upload")
async def upload_resume_only(file: UploadFile = File(...), role: str = Form(...), current_user: dict = Depends(get_current_user)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        pdf_content = await file.read()
        pdf_buffer = io.BytesIO(pdf_content)
        
        try:
            resume_text = pdf_reader(pdf_buffer)
            if not resume_text.strip():
                raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {str(e)}")
        
        basic_info = extract_basic_info_from_text(resume_text)
        timestamp = int(time.time())
        filename = f"resumes/{current_user['user_id']}/{timestamp}_{file.filename}"
        file_url = await upload_to_firebase_storage(content=pdf_content, filename=filename, content_type=file.content_type or "application/pdf")
        doc_id = await save_resume_to_firestore(current_user['user_id'], basic_info, file.filename, file_url, role)
        
        return {
            "message": "Resume uploaded successfully",
            "doc_id": doc_id,
            "basic_info": basic_info,
            "file_url": file_url,
            "applied_role": role,
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "uploaded"
        }
    except Exception as e:
        logger.error(f"❌ Resume upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading resume: {str(e)}")

@app.get("/api/admin/stats", response_model=DashboardStats)
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        if db:
            total_users = len(list(db.collection('users').stream()))
            resume_docs = list(db.collection('resumes').stream())
            total_resumes = len(resume_docs)
            analyzed_resumes = [doc for doc in resume_docs if doc.to_dict().get('ats_score') is not None]
            average_score = round(sum(doc.to_dict().get('ats_score', 0) for doc in analyzed_resumes) / len(analyzed_resumes), 1) if analyzed_resumes else 0.0
        else:
            total_users = len(IN_MEMORY_USERS)
            total_resumes = len(IN_MEMORY_RESUMES)
            analyzed_resumes = [r for r in IN_MEMORY_RESUMES.values() if r.get('ats_score') is not None]
            average_score = round(sum(r.get('ats_score', 0) for r in analyzed_resumes) / len(analyzed_resumes), 1) if analyzed_resumes else 0.0
        
        return DashboardStats(total_users=total_users, total_resumes=total_resumes, average_score=average_score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard stats: {str(e)}")

@app.get("/api/admin/admins")
async def get_all_admins(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if db:
        try:
            results = []
            for doc in db.collection('admins').stream():
                admin_data = doc.to_dict()
                admin_data['id'] = doc.id
                admin_data.pop('password_hash', None)
                results.append(admin_data)
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching admins: {str(e)}")
    else:
        results = []
        for admin_id, admin_data in IN_MEMORY_ADMINS.items():
            admin_copy = admin_data.copy()
            admin_copy['id'] = admin_id
            admin_copy.pop('password_hash', None)
            results.append(admin_copy)
        return results

@app.post("/api/admin/admins")
async def create_admin(admin_data: AdminCreate, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if admin_data.password != admin_data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    
    hashed_password = bcrypt.hashpw(admin_data.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_admin_doc = {
        "email": admin_data.email,
        "password_hash": hashed_password,
        "created_at": datetime.now(),
        "last_login": None,
        "created_by": current_user['email']
    }
    
    if db:
        try:
            admins_ref = db.collection('admins')
            if any(admins_ref.where('email', '==', admin_data.email).limit(1).stream()):
                raise HTTPException(status_code=400, detail="Admin email already exists")
            doc_ref = admins_ref.add(new_admin_doc)
            return {"message": "Admin created successfully", "admin_id": doc_ref[1].id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating admin: {str(e)}")
    else:
        if any(a['email'] == admin_data.email for a in IN_MEMORY_ADMINS.values()):
            raise HTTPException(status_code=400, detail="Admin email already exists")
        admin_id = f"dev_admin_{len(IN_MEMORY_ADMINS) + 1}"
        IN_MEMORY_ADMINS[admin_id] = new_admin_doc
        return {"message": "Admin created successfully", "admin_id": admin_id}

@app.delete("/api/admin/admins/{admin_id}")
async def delete_admin(admin_id: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if admin_id == current_user['user_id']:
        raise HTTPException(status_code=400, detail="Cannot delete your own admin account")
    
    if db:
        try:
            doc_ref = db.collection('admins').document(admin_id)
            if not doc_ref.get().exists:
                raise HTTPException(status_code=404, detail="Admin not found")
            doc_ref.delete()
            return {"message": "Admin deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting admin: {str(e)}")
    else:
        if admin_id not in IN_MEMORY_ADMINS:
            raise HTTPException(status_code=404, detail="Admin not found")
        del IN_MEMORY_ADMINS[admin_id]
        return {"message": "Admin deleted successfully"}

@app.get("/api/admin/resumes")
async def get_all_resumes(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if db:
        try:
            results = []
            for doc in db.collection('resumes').stream():
                resume_data = doc.to_dict()
                resume_data['id'] = doc.id
                for field, default in [
                    ('upload_date', resume_data.get('analysis_date', 'N/A')),
                    ('analysis_date', None), ('name', 'N/A'), ('email', 'N/A'),
                    ('ats_score', None), ('file_name', 'N/A'), ('file_url', ''),
                    ('applied_role', 'Not specified'), ('status', 'uploaded')
                ]:
                    if field not in resume_data:
                        resume_data[field] = default
                results.append(resume_data)
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching resumes: {str(e)}")
    else:
        results = []
        for resume_data in IN_MEMORY_RESUMES.values():
            for field, default in [
                ('upload_date', resume_data.get('analysis_date', 'N/A')),
                ('analysis_date', None), ('name', 'N/A'), ('email', 'N/A'),
                ('ats_score', None), ('file_name', 'N/A'), ('file_url', ''),
                ('applied_role', 'Not specified'), ('status', 'uploaded')
            ]:
                if field not in resume_data:
                    resume_data[field] = default
            results.append(resume_data)
        return results

@app.get("/api/admin/resumes/{resume_id}")
async def get_resume_by_id(resume_id: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if db:
        try:
            doc = db.collection('resumes').document(resume_id).get()
            if not doc.exists:
                raise HTTPException(status_code=404, detail="Resume not found")
            resume_data = doc.to_dict()
            resume_data['id'] = doc.id
            return resume_data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching resume: {str(e)}")
    else:
        if resume_id not in IN_MEMORY_RESUMES:
            raise HTTPException(status_code=404, detail="Resume not found")
        return IN_MEMORY_RESUMES[resume_id]

@app.post("/api/admin/resumes/{resume_id}/analyze")
async def analyze_resume_by_admin(resume_id: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        if db:
            doc = db.collection('resumes').document(resume_id).get()
            if not doc.exists:
                raise HTTPException(status_code=404, detail="Resume not found")
            resume_data = doc.to_dict()
        else:
            if resume_id not in IN_MEMORY_RESUMES:
                raise HTTPException(status_code=404, detail="Resume not found")
            resume_data = IN_MEMORY_RESUMES[resume_id]
        
        if resume_data.get('status') == 'analyzed':
            raise HTTPException(status_code=400, detail="Resume already analyzed")
        
        file_url = resume_data.get('file_url')
        if not file_url:
            raise HTTPException(status_code=400, detail="Resume file URL not found")
        
        response = requests.get(file_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not download resume file")
        
        pdf_buffer = io.BytesIO(response.content)
        resume_text = pdf_reader(pdf_buffer)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        basic_info = extract_basic_info_from_text(resume_text)
        extraction_results = {}
        for category, keywords in KEYWORD_CATEGORIES.items():
            if category == "Candidate Profile":
                extraction_results[category] = extract_candidate_profile_keywords(basic_info)
            else:
                extraction_results[category] = extract_keywords(resume_text, keywords)
        
        sections_found = detect_sections_presence(resume_text)
        ats_score, category_scores = calculate_ats_score(extraction_results, sections_found)
        suggestions = get_improvement_suggestions(extraction_results, resume_data.get('applied_role', 'General'))
        chart_base64 = create_keyword_chart(extraction_results, sections_found, resume_data.get('applied_role', 'General'))
        
        success = await update_resume_with_analysis(resume_id, ats_score, extraction_results, suggestions, chart_base64)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save analysis results")
        
        return {
            "basic_info": basic_info,
            "extraction_results": extraction_results,
            "ats_score": ats_score,
            "category_scores": category_scores,
            "sections_found": sections_found,
            "chart": chart_base64,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "suggestions": suggestions,
            "applied_role": resume_data.get('applied_role', 'General'),
            "status": "analyzed"
        }
    except Exception as e:
        logger.error(f"❌ Resume analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing resume: {str(e)}")

@app.get("/api/admin/resumes/{resume_id}/download")
async def download_resume_file(resume_id: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if db:
        try:
            doc = db.collection('resumes').document(resume_id).get()
            if not doc.exists:
                raise HTTPException(status_code=404, detail="Resume not found")
            resume_data = doc.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching resume: {str(e)}")
    else:
        if resume_id not in IN_MEMORY_RESUMES:
            raise HTTPException(status_code=404, detail="Resume not found")
        resume_data = IN_MEMORY_RESUMES[resume_id]
    
    file_url = resume_data.get('file_url')
    if not file_url:
        raise HTTPException(status_code=404, detail="Resume file URL not found")
    return {"file_url": file_url, "file_name": resume_data.get('file_name', 'resume.pdf')}

@app.delete("/api/admin/resumes/{resume_id}")
async def delete_resume(resume_id: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if db:
        try:
            doc_ref = db.collection('resumes').document(resume_id)
            doc = doc_ref.get()
            if not doc.exists:
                raise HTTPException(status_code=404, detail="Resume not found")
            doc_ref.delete()
            return {"message": "Resume deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting resume: {str(e)}")
    else:
        if resume_id not in IN_MEMORY_RESUMES:
            raise HTTPException(status_code=404, detail="Resume not found")
        del IN_MEMORY_RESUMES[resume_id]
        return {"message": "Resume deleted successfully"}

@app.get("/api/admin/resumes/export")
async def export_resumes_csv(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        data = []
        if db:
            for doc in db.collection('resumes').stream():
                resume_data = doc.to_dict()
                resume_data['id'] = doc.id
                data.append(resume_data)
        else:
            data = list(IN_MEMORY_RESUMES.values())
        
        csv_headers = ['id', 'name', 'email', 'phone', 'address', 'linkedin', 'github',
                       'ats_score', 'upload_date', 'analysis_date', 'file_name',
                       'applied_role', 'user_id', 'status', 'file_url']
        
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(csv_headers)
        
        for resume in data or []:
            row_data = []
            for field in csv_headers:
                value = resume.get(field, 'N/A')
                try:
                    if value and hasattr(value, 'strftime'):
                        value = value.strftime('%Y-%m-%d %H:%M:%S')
                    elif value and hasattr(value, 'isoformat'):
                        value = value.isoformat()
                    elif value is None:
                        value = 'N/A'
                    elif isinstance(value, (dict, list)):
                        value = str(value) if value else 'N/A'
                    elif isinstance(value, bool):
                        value = 'Yes' if value else 'No'
                    else:
                        value = str(value)
                except Exception:
                    value = 'Error'
                row_data.append(value)
            writer.writerow(row_data)
        
        if not data:
            writer.writerow(['No data available'] + [''] * (len(csv_headers) - 1))
        
        csv_content = output.getvalue()
        output.close()
        
        filename = f"resume_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            content=csv_content.encode('utf-8'),
            media_type="text/csv",
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Type': 'text/csv; charset=utf-8',
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")

@app.get("/api/health")
async def health_check():
    missing_vars = []
    if not os.getenv("FIREBASE_SERVICE_ACCOUNT"):
        missing_vars.append("FIREBASE_SERVICE_ACCOUNT")
    if not os.getenv("JWT_SECRET"):
        missing_vars.append("JWT_SECRET")

    return {
        "status": "healthy" if firebase_initialized else "degraded",
        "firebase_initialized": firebase_initialized,
        "firestore": db is not None,
        "storage": bucket is not None,
        "environment": "production" if firebase_initialized else "development",
        "missing_env_vars": missing_vars,
        "hint": (
            "Set the listed env vars in Vercel → Settings → Environment Variables, "
            "then redeploy."
        ) if missing_vars else None,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
