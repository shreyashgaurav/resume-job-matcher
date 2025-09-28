import PyPDF2
import pdfplumber
import io
from typing import Optional


class ResumeParser:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF using multiple methods for better accuracy"""
        text = ""

        # Try pdfplumber first (better for complex layouts)
        try:
            text = self._extract_with_pdfplumber(file_content)
            if text.strip():
                return self._clean_text(text)
        except Exception as e:
            print(f"pdfplumber failed: {e}")

        # Fallback to PyPDF2
        try:
            text = self._extract_with_pypdf2(file_content)
            if text.strip():
                return self._clean_text(text)
        except Exception as e:
            print(f"PyPDF2 failed: {e}")

        raise Exception("Could not extract text from PDF")

    def _extract_with_pdfplumber(self, file_content: bytes) -> str:
        """Extract text using pdfplumber"""
        text = ""
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def _extract_with_pypdf2(self, file_content: bytes) -> str:
        """Extract text using PyPDF2"""
        text = ""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove special characters that might interfere with NLP
        # Keep alphanumeric, common punctuation, and spaces
        cleaned_text = ''.join(char for char in text if char.isalnum() or char in ' .,;:!?()-\n')

        return cleaned_text.strip()

    def extract_basic_info(self, text: str) -> dict:
        """Extract basic information like email, phone (basic regex patterns)"""
        import re

        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)

        # Phone pattern (simple)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, text)

        return {
            "emails": emails,
            "phones": phones
        }