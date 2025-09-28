from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
import json
import os

from app.services.nlp_processor import EnhancedNLPProcessor
from app.services.matcher import EnhancedJobMatcher

app = FastAPI(title="Resume Job Matcher", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
if os.path.exists("../frontend"):
    app.mount("/static", StaticFiles(directory="../frontend"), name="static")

from app.services.resume_parser import ResumeParser

# Initialize services
resume_parser = ResumeParser()
nlp_processor = EnhancedNLPProcessor()
job_matcher = EnhancedJobMatcher(nlp_processor)


@app.get("/")
async def root():
    # Serve the frontend if it exists, otherwise return API message
    if os.path.exists("../frontend/index.html"):
        return FileResponse("../frontend/index.html")
    return {"message": "Resume Job Matcher API is running!"}


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Upload and parse a resume"""
    try:
        # Read file content
        content = await file.read()

        # Parse resume text
        resume_text = resume_parser.extract_text_from_pdf(content)

        # Extract skills and generate embeddings
        skills = nlp_processor.extract_skills(resume_text)
        embeddings = nlp_processor.get_embeddings(resume_text)

        return {
            "filename": file.filename,
            "text_preview": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
            "skills": skills,
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.post("/match-job")
async def match_job(
        job_description: str = Form(...),
        resume_files: List[UploadFile] = File(...)
):
    """Match resumes against a job description"""
    try:
        results = []

        for resume_file in resume_files:
            content = await resume_file.read()
            resume_text = resume_parser.extract_text_from_pdf(content)

            # Calculate match score
            match_score = job_matcher.calculate_match_score(resume_text, job_description)

            results.append({
                "filename": resume_file.filename,
                "match_score": match_score,
                "resume_skills": nlp_processor.extract_skills(resume_text),
                "job_skills": nlp_processor.extract_skills(job_description)
            })

        # Sort by match score (descending)
        results.sort(key=lambda x: x["match_score"], reverse=True)

        return {
            "job_description": job_description,
            "matches": results,
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": nlp_processor.is_loaded()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)