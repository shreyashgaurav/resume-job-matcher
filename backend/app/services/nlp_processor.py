import re
import nltk
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EnhancedNLPProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.is_loaded_flag = True
        self.sentence_model = None

        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            print("Warning: Could not download NLTK data")

        # Initialize sentence transformer
        self._load_sentence_transformer()

    def _load_sentence_transformer(self):
        """Load sentence transformer model with fallback"""
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Sentence transformer loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load sentence transformer: {e}")
            print("Falling back to TF-IDF similarity")
            self.sentence_model = None

    def is_loaded(self) -> bool:
        return self.is_loaded_flag

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills using enhanced keyword matching"""
        text_lower = text.lower()

        # Enhanced skill database with variations
        technical_skills = {
            # Programming Languages
            'python': ['python', 'py', 'python3', 'python programming'],
            'javascript': ['javascript', 'js', 'node.js', 'nodejs', 'ecmascript'],
            'java': ['java', 'openjdk', 'oracle java'],
            'typescript': ['typescript', 'ts'],
            'c++': ['c++', 'cpp', 'c plus plus'],
            'c#': ['c#', 'csharp', 'c sharp'],
            'sql': ['sql', 'mysql', 'postgresql', 'sqlite', 'oracle sql'],
            'r': ['r programming', 'r language'],

            # ML/AI Technologies
            'machine learning': ['machine learning', 'ml', 'artificial intelligence', 'ai', 'predictive modeling'],
            'deep learning': ['deep learning', 'neural networks', 'deep neural networks', 'dnn'],
            'natural language processing': ['nlp', 'natural language processing', 'text mining', 'language models'],
            'computer vision': ['computer vision', 'cv', 'image processing', 'opencv'],
            'data science': ['data science', 'data analysis', 'analytics', 'data mining'],

            # Frameworks
            'react': ['react', 'reactjs', 'react.js'],
            'angular': ['angular', 'angularjs'],
            'vue': ['vue', 'vue.js', 'vuejs'],
            'django': ['django', 'django framework'],
            'flask': ['flask', 'flask framework'],
            'tensorflow': ['tensorflow', 'tf'],
            'pytorch': ['pytorch', 'torch'],
            'keras': ['keras'],

            # Cloud & DevOps
            'aws': ['aws', 'amazon web services', 'amazon cloud'],
            'azure': ['azure', 'microsoft azure'],
            'gcp': ['gcp', 'google cloud', 'google cloud platform'],
            'docker': ['docker', 'containerization'],
            'kubernetes': ['kubernetes', 'k8s'],
            'git': ['git', 'version control', 'github', 'gitlab'],

            # Databases
            'mongodb': ['mongodb', 'mongo'],
            'redis': ['redis'],
            'elasticsearch': ['elasticsearch', 'elastic search'],

            # Other Skills
            'agile': ['agile', 'scrum', 'kanban'],
            'project management': ['project management', 'pmp'],
            'communication': ['communication', 'presentation', 'public speaking'],
            'leadership': ['leadership', 'team lead', 'management']
        }

        found_skills = []

        # Extract skills using enhanced matching
        for skill_name, variations in technical_skills.items():
            for variation in variations:
                if variation in text_lower:
                    found_skills.append(skill_name.title())
                    break  # Don't add duplicates

        # Extract additional patterns
        skill_patterns = [
            r'\b([A-Z][a-z]+(?:\.[a-z]+)+)\b',  # Framework patterns like React.js
            r'\b([A-Z]{2,5})\b',  # Acronyms like API, REST, JSON
        ]

        for pattern in skill_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 2 and match.lower() not in [v for vals in technical_skills.values() for v in vals]:
                    found_skills.append(match)

        return list(set(found_skills))

    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        if self.sentence_model is None:
            return self.calculate_similarity(text1, text2)

        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Semantic similarity error: {e}")
            return self.calculate_similarity(text1, text2)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using TF-IDF (fallback method)"""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0

    def get_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings using sentence transformers or TF-IDF"""
        if self.sentence_model is not None:
            try:
                return self.sentence_model.encode([text])[0]
            except:
                pass

        # Fallback to TF-IDF
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text])
            return tfidf_matrix.toarray()[0]
        except:
            return np.zeros(384)  # Default sentence transformer dimension

    def extract_experience_years(self, text: str) -> int:
        """Extract years of experience from text with enhanced patterns"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'experience\s*[:.]?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?',
            r'(\d+)\+?\s*year\s*(?:of\s*)?(?:experience|exp)',
            r'over\s*(\d+)\s*years?',
            r'more\s*than\s*(\d+)\s*years?'
        ]

        years = []
        text_lower = text.lower()

        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    years.append(int(match))
                except:
                    continue

        return max(years) if years else 0

    def extract_education(self, text: str) -> List[str]:
        """Extract education information with enhanced matching"""
        education_keywords = {
            'bachelor': ['bachelor', 'bachelors', 'b.s.', 'b.a.', 'b.sc.', 'b.tech', 'undergraduate'],
            'master': ['master', 'masters', 'm.s.', 'm.a.', 'm.sc.', 'm.tech', 'graduate'],
            'phd': ['phd', 'ph.d.', 'doctorate', 'doctoral', 'doctor of philosophy'],
            'mba': ['mba', 'master of business administration'],
            'computer science': ['computer science', 'cs', 'computer engineering'],
            'engineering': ['engineering', 'engineer'],
            'mathematics': ['mathematics', 'math', 'statistics', 'applied math'],
            'data science': ['data science', 'data analytics', 'information systems']
        }

        found_education = []
        text_lower = text.lower()

        for edu_name, variations in education_keywords.items():
            for variation in variations:
                if variation in text_lower:
                    found_education.append(edu_name.title())
                    break

        return list(set(found_education))