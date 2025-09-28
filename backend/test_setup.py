#!/usr/bin/env python3

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import fastapi
        print("✓ FastAPI imported successfully")

        import PyPDF2
        print("✓ PyPDF2 imported successfully")

        import pdfplumber
        print("✓ pdfplumber imported successfully")

        import pandas
        print("✓ Pandas imported successfully")

        import sklearn
        print("✓ scikit-learn imported successfully")

        import nltk
        print("✓ NLTK imported successfully")

        print("\n🎉 All core packages imported successfully!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_nlp_models():
    """Test if NLP models are working"""
    try:
        import nltk
        print("✓ NLTK imported successfully")

        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✓ scikit-learn TF-IDF available")

        # Test our simple NLP processor (PyCharm handles imports better)
        from app.services.nlp_processor import NLPProcessor
        nlp = NLPProcessor()

        test_text = "I have 5 years of experience in Python, JavaScript, and React development."
        skills = nlp.extract_skills(test_text)
        print(f"✓ Skill extraction working: {skills[:3]}...")  # Show first 3 skills

        similarity = nlp.calculate_similarity("Python developer", "JavaScript engineer")
        print(f"✓ Text similarity working: {similarity:.2f}")

        return True

    except Exception as e:
        print(f"❌ NLP setup error: {e}")
        return False


if __name__ == "__main__":
    print("Testing project setup...\n")

    imports_ok = test_imports()
    spacy_ok = test_nlp_models()

    if imports_ok and spacy_ok:
        print("\n🚀 Setup is complete! Ready to proceed.")
    else:
        print("\n⚠️ Some issues found. Please fix them before proceeding.")