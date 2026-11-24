
"""
Resume-Job Matching Streamlit Application

Features:
1. Single Resume vs Job Description Matching
2. Batch Resume Analysis
3. Live Job Search with AI Resume Matching
4. Match History
5. PDF, DOCX and TXT resume support
"""

import io
import os
import json
import sqlite3

from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st

from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Resume Job Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TEST_DIR = DATA_DIR / "test"

DB_PATH = BASE_DIR / "match_history.db"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

MAX_WORDS = 350

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATABASE FUNCTIONS
# ============================================================

def get_db_connection():
    """Create and return a SQLite database connection."""

    return sqlite3.connect(DB_PATH)


def init_history_db():
    """Create the match history table if it does not exist."""

    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS match_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_type TEXT,
            resume_name TEXT,
            job_description TEXT,
            resume_text TEXT,
            match_score REAL,
            status TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_match_history(
    match_type,
    resume_name,
    job_description,
    resume_text,
    match_score,
    status
):
    """Save a matching result to SQLite."""

    conn = get_db_connection()

    conn.execute("""
        INSERT INTO match_history (
            match_type,
            resume_name,
            job_description,
            resume_text,
            match_score,
            status,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        match_type,
        resume_name,
        job_description,
        resume_text,
        float(match_score),
        status,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_match_history():
    """Return all saved matching records."""

    conn = get_db_connection()

    df = pd.read_sql_query(
        "SELECT * FROM match_history ORDER BY id DESC",
        conn
    )

    conn.close()

    return df


def delete_history_record(record_id):
    """Delete one selected history record."""

    conn = get_db_connection()

    conn.execute(
        "DELETE FROM match_history WHERE id = ?",
        (int(record_id),)
    )

    conn.commit()
    conn.close()


def clear_match_history():
    """Delete all records from match history."""

    conn = get_db_connection()

    conn.execute("DELETE FROM match_history")

    conn.commit()
    conn.close()


# ============================================================
# 3. RESUME TEXT EXTRACTION
# ============================================================

def extract_resume_text(uploaded_file):
    """
    Extract text from PDF, DOCX or TXT files.
    Returns extracted text as a string.
    """

    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()

    try:

        if file_name.endswith(".pdf"):

            pdf_reader = PdfReader(uploaded_file)

            pages_text = []

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:
                    pages_text.append(page_text)

            return "\n".join(pages_text)

        elif file_name.endswith(".docx"):

            document = Document(uploaded_file)

            text_parts = []

            for paragraph in document.paragraphs:

                if paragraph.text.strip():
                    text_parts.append(paragraph.text)

            for table in document.tables:

                for row in table.rows:

                    row_text = [
                        cell.text.strip()
                        for cell in row.cells
                    ]

                    text_parts.append(" ".join(row_text))

            return "\n".join(text_parts)

        elif file_name.endswith(".txt"):

            content = uploaded_file.getvalue()

            try:
                return content.decode("utf-8")

            except UnicodeDecodeError:
                return content.decode("latin-1")

        else:

            st.error(
                "Unsupported file format. "
                "Please upload PDF, DOCX or TXT."
            )

            return ""

    except Exception as error:

        st.error(f"Error reading resume: {error}")

        return ""


# ============================================================
# 4. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    """Clean and normalize text."""

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    text = " ".join(text.split())

    return text


def truncate_text(text, max_words=MAX_WORDS):
    """Limit text to a maximum number of words."""

    words = str(text).split()

    return " ".join(words[:max_words])


def clean_and_truncate(text, max_words=MAX_WORDS):
    """Clean text and limit its length."""

    cleaned_text = preprocess_text(text)

    return truncate_text(cleaned_text, max_words)


# ============================================================
# 5. SENTENCE TRANSFORMER MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the Sentence Transformer model."""

    return SentenceTransformer(MODEL_NAME)


def compute_embedding(text, model):
    """Generate a semantic embedding for text."""

    cleaned_text = clean_and_truncate(text)

    if not cleaned_text.strip():
        return None

    embedding = model.encode(
        cleaned_text,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embedding


# ============================================================
# 6. MATCH SCORE CALCULATION
# ============================================================

def calculate_match_score(resume_text, job_description, model):
    """
    Calculate semantic similarity between a resume
    and a job description.

    Returns a percentage score.
    """

    resume_embedding = compute_embedding(
        resume_text,
        model
    )

    job_embedding = compute_embedding(
        job_description,
        model
    )

    if resume_embedding is None or job_embedding is None:
        return 0.0

    similarity = cosine_similarity(
        resume_embedding.reshape(1, -1),
        job_embedding.reshape(1, -1)
    )[0][0]

    score = float(similarity) * 100

    score = max(0.0, min(100.0, score))

    return round(score, 2)


def get_score_label(score):
    """Return a label based on the matching score."""

    if score >= 80:
        return "Excellent Match"

    elif score >= 65:
        return "Strong Match"

    elif score >= 50:
        return "Moderate Match"

    elif score >= 35:
        return "Low Match"

    else:
        return "Very Low Match"


def get_score_color(score):
    """Return a display color based on the score."""

    if score >= 80:
        return "green"

    elif score >= 65:
        return "blue"

    elif score >= 50:
        return "orange"

    else:
        return "red"


# ============================================================
# 7. SINGLE RESUME MATCHING
# ============================================================

def render_single_match(model):

    st.header("🎯 Single Resume Matching")

    st.write(
        "Compare one resume with a job description "
        "using semantic similarity."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📋 Job Description")

        job_description = st.text_area(
            "Paste Job Description",
            height=300,
            placeholder="Paste the complete job description here...",
            key="single_job_description"
        )

    with col2:

        st.subheader("📄 Upload Resume")

        uploaded_file = st.file_uploader(
            "Upload Resume",
            type=["pdf", "docx", "txt"],
            key="single_resume_upload"
        )

        resume_text_input = st.text_area(
            "Or Paste Resume Text",
            height=200,
            placeholder="Paste resume text here...",
            key="single_resume_text"
        )

    if st.button(
        "🚀 Analyze Match",
        type="primary",
        use_container_width=True,
        key="single_analyze_button"
    ):

        resume_text = ""

        resume_name = "Pasted Resume"

        if uploaded_file is not None:

            resume_text = extract_resume_text(uploaded_file)

            resume_name = uploaded_file.name

        elif resume_text_input.strip():

            resume_text = resume_text_input

        if not job_description.strip():

            st.warning("Please enter a job description.")

        elif not resume_text.strip():

            st.warning("Please upload a resume or paste resume text.")

        else:

            with st.spinner(
                "Analyzing resume and calculating match..."
            ):

                score = calculate_match_score(
                    resume_text,
                    job_description,
                    model
                )

                label = get_score_label(score)

                save_match_history(
                    match_type="Single Match",
                    resume_name=resume_name,
                    job_description=job_description,
                    resume_text=resume_text,
                    match_score=score,
                    status=label
                )

            st.success("Resume analysis completed!")

            st.divider()

            st.subheader("📊 Match Results")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Match Score",
                    f"{score:.2f}%"
                )

            with col2:

                st.metric(
                    "Match Status",
                    label
                )

            with col3:

                st.metric(
                    "Resume Words",
                    len(resume_text.split())
                )

            st.progress(
                min(score / 100, 1.0)
            )

            if score >= 80:

                st.success(
                    "Excellent! Your resume has strong semantic "
                    "similarity with this job description."
                )

            elif score >= 65:

                st.info(
                    "Good match! Your resume aligns with many "
                    "aspects of the job description."
                )

            elif score >= 50:

                st.warning(
                    "Moderate match. Consider improving your resume "
                    "to better reflect the job requirements."
                )

            else:

                st.warning(
                    "Your resume has a relatively low match score. "
                    "Review the required skills and experience."
                )

            with st.expander("View Resume Text"):

                st.write(resume_text)

            with st.expander("View Job Description"):

                st.write(job_description)


# ============================================================
# 8. BATCH RESUME ANALYSIS
# ============================================================

def render_batch_analysis(model):

    st.header("📚 Batch Resume Analysis")

    st.write(
        "Upload multiple resumes and compare them "
        "against one common job description."
    )

    job_description = st.text_area(
        "Enter Job Description",
        height=220,
        placeholder="Paste the job description here...",
        key="batch_job_description"
    )

    uploaded_files = st.file_uploader(
        "Upload Multiple Resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="batch_resume_upload"
    )

    if st.button(
        "🔍 Analyze All Resumes",
        type="primary",
        use_container_width=True,
        key="batch_analyze_button"
    ):

        if not job_description.strip():

            st.warning("Please enter a job description.")

        elif not uploaded_files:

            st.warning("Please upload at least one resume.")

        else:

            results = []

            progress_bar = st.progress(0)

            status_text = st.empty()

            total_files = len(uploaded_files)

            for index, uploaded_file in enumerate(uploaded_files):

                status_text.write(
                    f"Analyzing {uploaded_file.name}..."
                )

                resume_text = extract_resume_text(uploaded_file)

                if not resume_text.strip():

                    results.append({
                        "Resume Name": uploaded_file.name,
                        "Match Score": 0.0,
                        "Status": "Could not extract text"
                    })

                else:

                    score = calculate_match_score(
                        resume_text,
                        job_description,
                        model
                    )

                    label = get_score_label(score)

                    save_match_history(
                        match_type="Batch Match",
                        resume_name=uploaded_file.name,
                        job_description=job_description,
                        resume_text=resume_text,
                        match_score=score,
                        status=label
                    )

                    results.append({
                        "Resume Name": uploaded_file.name,
                        "Match Score": score,
                        "Status": label
                    })

                progress_bar.progress(
                    (index + 1) / total_files
                )

            status_text.success(
                "Batch analysis completed!"
            )

            results_df = pd.DataFrame(results)

            valid_results = results_df[
                results_df["Status"] != "Could not extract text"
            ]


            if not valid_results.empty:

                valid_results = valid_results.sort_values(
                    by="Match Score",
                    ascending=False
                ).reset_index(drop=True)

                results_df = valid_results.copy()

                best_match = valid_results.iloc[0]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total Resumes",
                        len(results_df)
                    )

                with col2:
                    st.metric(
                        "Best Match",
                        best_match["Resume Name"]
                    )

                with col3:
                    st.metric(
                        "Best Score",
                        f"{best_match['Match Score']:.2f}%"
                    )

                st.subheader("🏆 Top Matching Resumes")

                top_results = valid_results.head(5)

                for _, row in top_results.iterrows():

                    st.write(
                        f"**{row['Resume Name']}** — "
                        f"{row['Match Score']:.2f}% "
                        f"({row['Status']})"
                    )

                    st.progress(
                        min(float(row["Match Score"]) / 100, 1.0)
                    )



            csv_data = results_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Batch Results CSV",
                data=csv_data,
                file_name="batch_resume_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )


# ============================================================
# 9. LIVE JOB SEARCH - API HELPERS
# ============================================================

def fetch_json_api(url, headers=None):

    try:

        request = Request(
            url,
            headers=headers or {
                "User-Agent": "ResumeJobMatcher/1.0"
            }
        )

        with urlopen(request, timeout=25) as response:

            data = response.read().decode("utf-8")

            return json.loads(data)

    except Exception as error:

        st.warning(f"Unable to fetch jobs: {error}")

        return None


# ============================================================
# 10. ADZUNA API CREDENTIALS
# ============================================================

def get_adzuna_credentials():

    app_id = os.getenv("ADZUNA_APP_ID")

    app_key = os.getenv("ADZUNA_APP_KEY")

    try:

        if not app_id:

            app_id = st.secrets.get("ADZUNA_APP_ID")

        if not app_key:

            app_key = st.secrets.get("ADZUNA_APP_KEY")

    except Exception:

        pass

    return app_id, app_key


# ============================================================
# 11. FETCH JOBS FROM ADZUNA
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_adzuna_jobs(search_term, location, app_id, app_key):

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": search_term,
        "where": location,
        "results_per_page": 30,
        "content-type": "application/json"
    }

    url = (
        "https://api.adzuna.com/v1/api/jobs/in/search/1?"
        + urlencode(params)
    )

    data = fetch_json_api(url)

    jobs = []

    if not data:

        return jobs

    for item in data.get("results", []):

        company = item.get("company", {})

        job_location = item.get("location", {})

        salary_min = item.get("salary_min")

        salary_max = item.get("salary_max")

        salary = "Not disclosed"

        if salary_min and salary_max:

            salary = (
                f"{salary_min:,.0f} - "
                f"{salary_max:,.0f}"
            )

        elif salary_min:

            salary = f"From {salary_min:,.0f}"

        elif salary_max:

            salary = f"Up to {salary_max:,.0f}"

        jobs.append({

            "Job Title": item.get(
                "title",
                "Not specified"
            ),

            "Company": company.get(
                "display_name",
                "Not specified"
            ),

            "Location": job_location.get(
                "display_name",
                location
            ),

            "Description": item.get(
                "description",
                ""
            ),

            "Application Link": item.get(
                "redirect_url",
                ""
            ),

            "Source": "Adzuna",

            "Salary": salary,

            "Date Posted": item.get(
                "created",
                "Not specified"
            )

        })

    return jobs


# ============================================================
# 12. FETCH REMOTE JOBS FROM JOBICY
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_jobicy_jobs(search_term):

    params = {
        "count": 100,
        "geo": "anywhere"
    }

    url = (
        "https://jobicy.com/api/jobs?"
        + urlencode(params)
    )

    data = fetch_json_api(url)

    jobs = []

    if not data:

        return jobs

    search_words = [
        word.lower()
        for word in search_term.split()
        if len(word) > 2
    ]

    for item in data.get("jobs", []):

        title = item.get(
            "jobTitle",
            ""
        )

        description = item.get(
            "jobDescription",
            ""
        )

        combined_text = (
            title + " " + description
        ).lower()

        if search_words and not any(
            word in combined_text
            for word in search_words
        ):

            continue

        salary_min = item.get("annualSalaryMin")

        salary_max = item.get("annualSalaryMax")

        salary = "Not disclosed"

        if salary_min and salary_max:

            salary = (
                f"{salary_min} - {salary_max}"
            )

        elif salary_min:

            salary = f"From {salary_min}"

        elif salary_max:

            salary = f"Up to {salary_max}"

        jobs.append({

            "Job Title": title or "Not specified",

            "Company": item.get(
                "companyName",
                "Not specified"
            ),

            "Location": item.get(
                "jobGeo",
                "Remote"
            ),

            "Description": description,

            "Application Link": item.get(
                "url",
                ""
            ),

            "Source": "Jobicy - Remote",

            "Salary": salary,

            "Date Posted": item.get(
                "pubDate",
                "Not specified"
            )

        })

    return jobs


# ============================================================
# 13. SUGGEST JOB SEARCH KEYWORDS FROM RESUME
# ============================================================

def extract_job_keywords(resume_text):

    skills = [

        "data analyst",
        "power bi",
        "tableau",
        "python",
        "sql",
        "machine learning",
        "data science",
        "java",
        "javascript",
        "react",
        "web development",
        "software developer",
        "excel",
        "pandas",
        "numpy",
        "artificial intelligence",
        "business analyst",
        "data visualization",
        "nlp",
        "flask",
        "deep learning",
        "statistics",
        "data engineer"

    ]

    resume_lower = resume_text.lower()

    matched_skills = []

    for skill in skills:

        if skill in resume_lower:

            matched_skills.append(skill)

    if not matched_skills:

        return "data analyst"

    priority_skills = [
        "data analyst",
        "power bi",
        "python",
        "sql",
        "tableau",
        "machine learning",
        "data science",
        "business analyst"
    ]

    for skill in priority_skills:

        if skill in matched_skills:

            return skill

    return matched_skills[0]


# ============================================================
# 14. REMOVE DUPLICATE JOB LISTINGS
# ============================================================

def remove_duplicate_jobs(jobs):

    unique_jobs = []

    seen = set()

    for job in jobs:

        key = (
            job.get("Job Title", "").lower().strip(),
            job.get("Company", "").lower().strip(),
            job.get("Location", "").lower().strip()
        )

        if key not in seen:

            seen.add(key)

            unique_jobs.append(job)

    return unique_jobs


# ============================================================
# 15. LIVE JOB SEARCH PAGE
# ============================================================

def render_search_jobs(model):

    st.header("🔎 Live Internet Job Search")

    st.write(
        "Upload your resume to discover job opportunities "
        "and calculate how closely each job matches your profile."
    )

    st.info(
        "Adzuna provides India job listings when API credentials "
        "are configured. Jobicy provides remote job listings."
    )

    col1, col2 = st.columns(2)

    with col1:

        uploaded_resume = st.file_uploader(
            "Upload Your Resume",
            type=["pdf", "docx", "txt"],
            key="live_search_resume"
        )

    with col2:

        location = st.selectbox(
            "Select Job Location",
            [
                "Pune",
                "Mumbai",
                "Bengaluru",
                "Hyderabad",
                "Delhi",
                "India",
                "Remote"
            ],
            key="live_search_location"
        )

    resume_text_input = st.text_area(
        "Or Paste Your Resume Text",
        height=180,
        placeholder="Paste your resume content here...",
        key="live_search_resume_text"
    )

    job_role = st.text_input(
        "Job Role / Search Keywords",
        placeholder="Example: Data Analyst, Python Developer, Power BI",
        key="live_search_role"
    )

    max_jobs = st.slider(
        "Maximum Jobs to Display",
        min_value=5,
        max_value=30,
        value=15,
        key="live_search_limit"
    )

    if st.button(
        "🌐 Search Live Jobs",
        type="primary",
        use_container_width=True,
        key="live_search_button"
    ):

        resume_text = ""

        if uploaded_resume is not None:

            resume_text = extract_resume_text(
                uploaded_resume
            )

        elif resume_text_input.strip():

            resume_text = resume_text_input

        if not resume_text.strip():

            st.warning(
                "Please upload your resume or paste resume text."
            )

            return

        if job_role.strip():

            search_term = job_role.strip()

        else:

            search_term = extract_job_keywords(
                resume_text
            )

        st.write(
            f"**Search keywords:** {search_term}"
        )

        all_jobs = []

        app_id, app_key = get_adzuna_credentials()

        if location != "Remote":

            if app_id and app_key:

                with st.spinner(
                    "Searching India job listings on Adzuna..."
                ):

                    adzuna_jobs = fetch_adzuna_jobs(
                        search_term,
                        location,
                        app_id,
                        app_key
                    )

                    all_jobs.extend(adzuna_jobs)

            else:

                st.warning(
                    "Adzuna API credentials are not configured. "
                    "India location listings will be unavailable. "
                    "You can still view remote jobs from Jobicy."
                )

        with st.spinner(
            "Searching remote job listings on Jobicy..."
        ):

            remote_jobs = fetch_jobicy_jobs(
                search_term
            )

            all_jobs.extend(remote_jobs)

        all_jobs = remove_duplicate_jobs(
            all_jobs
        )

        if not all_jobs:

            st.warning(
                "No matching jobs were returned by the job sources. "
                "Try another job role or search again later."
            )

            return

        st.success(
            f"Found {len(all_jobs)} job listings."
        )

        results = []

        with st.spinner(
            "Calculating resume-to-job match scores..."
        ):

            for job in all_jobs:

                job_description = (
                    job.get("Job Title", "")
                    + " "
                    + job.get("Description", "")
                )

                score = calculate_match_score(
                    resume_text,
                    job_description,
                    model
                )

                job_result = job.copy()

                job_result["Match Score"] = round(
                    score,
                    2
                )

                job_result["Match Status"] = (
                    get_score_label(score)
                )

                results.append(job_result)

        results.sort(
            key=lambda item: item["Match Score"],
            reverse=True
        )

        results = results[:max_jobs]

        st.divider()

        st.subheader("📊 Job Search Results")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Jobs Displayed",
                len(results)
            )

        with col2:

            best_score = max(
                item["Match Score"]
                for item in results
            )

            st.metric(
                "Best Match",
                f"{best_score:.2f}%"
            )

        with col3:

            average_score = sum(
                item["Match Score"]
                for item in results
            ) / len(results)

            st.metric(
                "Average Match",
                f"{average_score:.2f}%"
            )

        for index, job in enumerate(results, start=1):

            with st.container(border=True):

                st.subheader(
                    f"{index}. {job['Job Title']}"
                )

                st.write(
                    f"**Company:** {job['Company']}"
                )

                st.write(
                    f"**Location:** {job['Location']}"
                )

                st.write(
                    f"**Source:** {job['Source']}"
                )

                st.write(
                    f"**Salary:** {job['Salary']}"
                )

                st.write(
                    f"**Date Posted:** {job['Date Posted']}"
                )

                st.write(
                    f"**Resume Match:** "
                    f"{job['Match Score']:.2f}% "
                    f"— {job['Match Status']}"
                )

                st.progress(
                    min(
                        job["Match Score"] / 100,
                        1.0
                    )
                )

                with st.expander(
                    "View Job Description"
                ):

                    st.write(
                        job["Description"]
                    )

                if job["Application Link"]:

                    st.link_button(
                        "Apply for this Job",
                        job["Application Link"],
                        use_container_width=True
                    )

        export_df = pd.DataFrame(results)

        csv_data = export_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Job Search Results CSV",
            data=csv_data,
            file_name="live_job_search_results.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# 16. MATCH HISTORY PAGE
# ============================================================

def render_match_history():

    st.header("📜 Match History")

    st.write(
        "View and manage your previously saved resume matching results."
    )

    history = get_match_history()

    if history.empty:

        st.info(
            "No match history available yet. "
            "Perform a resume analysis to see results here."
        )

        return

    history_df = history.copy()

    st.subheader("📊 Match History Overview")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Matches",
            len(history_df)
        )

    with col2:

        average_score = history_df["match_score"].mean()

        st.metric(
            "Average Match Score",
            f"{average_score:.2f}%"
        )

    with col3:

        best_score = history_df["match_score"].max()

        st.metric(
            "Best Match Score",
            f"{best_score:.2f}%"
        )

    st.divider()

    st.subheader("Previous Matching Records")

    display_df = history_df[
        [
            "id",
            "match_type",
            "resume_name",
            "match_score",
            "status",
            "created_at"
        ]
    ].copy()

    display_df.columns = [
        "ID",
        "Match Type",
        "Resume Name",
        "Match Score",
        "Status",
        "Created At"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = display_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Match History CSV",
        data=csv_data,
        file_name="match_history.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("View Individual Match")

    record_ids = history_df["id"].tolist()

    selected_id = st.selectbox(
        "Select Match Record",
        record_ids,
        key="history_record_select"
    )

    selected_record = history_df[
        history_df["id"] == selected_id
    ].iloc[0]

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"**Resume:** {selected_record['resume_name']}"
        )

        st.write(
            f"**Match Type:** {selected_record['match_type']}"
        )

    with col2:

        st.metric(
            "Match Score",
            f"{selected_record['match_score']:.2f}%"
        )

        st.write(
            f"**Status:** {selected_record['status']}"
        )

    with st.expander("View Saved Resume Text"):

        st.write(
            selected_record["resume_text"]
        )

    with st.expander("View Saved Job Description"):

        st.write(
            selected_record["job_description"]
        )

    st.divider()

    st.subheader("Manage Match History")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🗑️ Delete Selected Record",
            use_container_width=True
        ):

            delete_history_record(
                selected_id
            )

            st.success(
                "Selected record deleted."
            )

            st.rerun()

    with col2:

        if st.button(
            "⚠️ Clear All History",
            use_container_width=True
        ):

            st.session_state["confirm_clear_history"] = True

    if st.session_state.get(
        "confirm_clear_history",
        False
    ):

        st.warning(
            "Are you sure you want to delete all match history?"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "Yes, Delete All",
                type="primary"
            ):

                clear_match_history()

                st.session_state["confirm_clear_history"] = False

                st.success(
                    "All match history has been cleared."
                )

                st.rerun()

        with col2:

            if st.button(
                "Cancel"
            ):

                st.session_state["confirm_clear_history"] = False

                st.rerun()


# ============================================================
# 17. MAIN APPLICATION
# ============================================================

def main():

    init_history_db()

    st.set_page_config(
        page_title="Resume Job Matcher",
        page_icon="🎯",
        layout="wide"
    )

    st.title("🎯 Resume Job Matcher")

    st.caption(
        "AI-powered resume analysis and live job discovery"
    )

    st.sidebar.title("Navigation")

    selected_page = st.sidebar.radio(
        "Choose a Feature",
        [
            "Home",
            "Single Match",
            "Batch Analysis",
            "Search Jobs",
            "Match History"
        ]
    )

    st.sidebar.divider()

    st.sidebar.info(
        "Match your resume with job descriptions, "
        "analyze multiple resumes, discover live jobs, "
        "and review your matching history."
    )

    if selected_page == "Home":

        st.header("Welcome to Resume Job Matcher 👋")

        st.write(
            "An AI-powered application that helps you analyze "
            "resume relevance and discover job opportunities."
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("🎯 Single Resume Matching")

            st.write(
                "Compare one resume against a job description "
                "using semantic similarity."
            )

            st.subheader("📚 Batch Resume Analysis")

            st.write(
                "Upload multiple resumes and rank them "
                "against one job description."
            )

        with col2:

            st.subheader("🌐 Live Job Search")

            st.write(
                "Fetch job listings from Adzuna and Jobicy "
                "and calculate resume-to-job match scores."
            )

            st.subheader("📜 Match History")

            st.write(
                "View, download, and manage previous "
                "resume matching results."
            )

        st.divider()

        st.subheader("⚙️ Technology Stack")

        st.write(
            "- Python\n"
            "- Streamlit\n"
            "- Sentence Transformers\n"
            "- Semantic Similarity\n"
            "- Cosine Similarity\n"
            "- SQLite Database\n"
            "- Adzuna Jobs API\n"
            "- Jobicy Jobs API\n"
            "- Pandas and NumPy"
        )

        st.divider()

        st.caption(
            "Resume Job Matcher | AI-powered job matching application"
        )

    elif selected_page == "Single Match":

        model = load_model()

        render_single_match(model)

    elif selected_page == "Batch Analysis":

        model = load_model()

        render_batch_analysis(model)

    elif selected_page == "Search Jobs":

        model = load_model()

        render_search_jobs(model)

    elif selected_page == "Match History":

        render_match_history()


# ============================================================
# 18. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
