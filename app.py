from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import PyPDF2
import pdfplumber
import io
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

app = FastAPI(title="Resume Job Matcher", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)


# Resume Parser Class
class ResumeParser:
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        text = ""
        try:
            text = self._extract_with_pdfplumber(file_content)
            if text.strip():
                return self._clean_text(text)
        except:
            pass

        try:
            text = self._extract_with_pypdf2(file_content)
            if text.strip():
                return self._clean_text(text)
        except:
            pass

        return "Could not extract text from PDF"

    def _extract_with_pdfplumber(self, file_content: bytes) -> str:
        text = ""
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def _extract_with_pypdf2(self, file_content: bytes) -> str:
        text = ""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def _clean_text(self, text: str) -> str:
        text = ' '.join(text.split())
        cleaned_text = ''.join(char for char in text if char.isalnum() or char in ' .,;:!?()-\n')
        return cleaned_text.strip()


# NLP Processor Class
class NLPProcessor:
    def __init__(self):
        self.is_loaded_flag = True

    def is_loaded(self) -> bool:
        return self.is_loaded_flag

    def extract_skills(self, text: str) -> List[str]:
        text_lower = text.lower()

        technical_skills = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
            'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'xml', 'json',
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'node.js', 'express',
            'laravel', 'rails', 'pytorch', 'tensorflow', 'keras', 'pandas', 'numpy',
            'scikit-learn', 'opencv', 'selenium', 'junit', 'jest',
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sqlite',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
            'gitlab', 'terraform', 'ansible', 'linux', 'unix', 'bash', 'powershell',
            'machine learning', 'deep learning', 'data science', 'artificial intelligence',
            'nlp', 'computer vision', 'data analysis', 'statistics', 'big data', 'hadoop',
            'spark', 'tableau', 'power bi', 'excel',
            'leadership', 'communication', 'teamwork', 'problem solving', 'project management',
            'agile', 'scrum', 'critical thinking', 'creativity', 'adaptability'
        ]

        found_skills = []
        for skill in technical_skills:
            if skill in text_lower:
                found_skills.append(skill.title())

        return list(set(found_skills))

    def get_embeddings(self, text: str) -> np.ndarray:
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            return tfidf_matrix.toarray()[0]
        except:
            return np.zeros(1000)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0

    def extract_experience_years(self, text: str) -> int:
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'experience\s*[:.]?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?',
        ]

        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    years.append(int(match))
                except:
                    continue

        return max(years) if years else 0


# Job Matcher Class
class JobMatcher:
    def __init__(self, nlp_processor):
        self.nlp_processor = nlp_processor

    def calculate_match_score(self, resume_text: str, job_description: str) -> float:
        resume_skills = set(skill.lower() for skill in self.nlp_processor.extract_skills(resume_text))
        job_skills = set(skill.lower() for skill in self.nlp_processor.extract_skills(job_description))

        if not job_skills:
            skill_match_score = 0.0
        else:
            matching_skills = resume_skills.intersection(job_skills)
            skill_match_score = len(matching_skills) / len(job_skills)

        text_similarity = self.nlp_processor.calculate_similarity(resume_text, job_description)

        job_experience_required = self.nlp_processor.extract_experience_years(job_description)
        resume_experience = self.nlp_processor.extract_experience_years(resume_text)

        if job_experience_required == 0:
            experience_score = 1.0
        else:
            experience_score = min(resume_experience / job_experience_required, 1.0)

        final_score = (
                0.5 * skill_match_score +
                0.3 * text_similarity +
                0.2 * experience_score
        )

        return round(final_score * 100, 2)


# Initialize services
resume_parser = ResumeParser()
nlp_processor = NLPProcessor()
job_matcher = JobMatcher(nlp_processor)


@app.get("/")
async def root():
    return {"message": "Resume Job Matcher API is running!"}


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        content = await file.read()
        resume_text = resume_parser.extract_text_from_pdf(content)
        skills = nlp_processor.extract_skills(resume_text)

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
    try:
        results = []

        for resume_file in resume_files:
            content = await resume_file.read()
            resume_text = resume_parser.extract_text_from_pdf(content)
            match_score = job_matcher.calculate_match_score(resume_text, job_description)

            results.append({
                "filename": resume_file.filename,
                "match_score": match_score,
                "resume_skills": nlp_processor.extract_skills(resume_text),
                "job_skills": nlp_processor.extract_skills(job_description)
            })

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