import requests
import json


# Test the basic endpoint
def test_root_endpoint():
    print("=== Testing Root Endpoint ===")
    response = requests.get("http://127.0.0.1:8000/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


# Test health check
def test_health_endpoint():
    print("\n=== Testing Health Endpoint ===")
    response = requests.get("http://127.0.0.1:8000/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_nlp_directly():
    print("\n=== Testing NLP Components Directly ===")
    try:
        from app.services.nlp_processor import NLPProcessor
        from app.services.matcher import JobMatcher

        nlp = NLPProcessor()
        matcher = JobMatcher(nlp)

        # Test skill extraction
        test_resume = "Software Engineer with 5 years experience in Python, React, and AWS"
        test_job = "Looking for a developer with Python and React skills"

        resume_skills = nlp.extract_skills(test_resume)
        job_skills = nlp.extract_skills(test_job)
        match_score = matcher.calculate_match_score(test_resume, test_job)

        print(f"Resume Skills: {resume_skills}")
        print(f"Job Skills: {job_skills}")
        print(f"Match Score: {match_score}%")

        return match_score > 0

    except Exception as e:
        print(f"Error in NLP test: {e}")
        return False


if __name__ == "__main__":
    print("Testing Resume Job Matcher API...\n")

    # Run tests
    root_ok = test_root_endpoint()
    health_ok = test_health_endpoint()
    nlp_ok = test_nlp_directly()

    print(f"\n=== Test Results ===")
    print(f"Root Endpoint: {'✓' if root_ok else '✗'}")
    print(f"Health Check: {'✓' if health_ok else '✗'}")
    print(f"NLP Processing: {'✓' if nlp_ok else '✗'}")

    if all([root_ok, health_ok, nlp_ok]):
        print("\n🎉 All tests passed! Your API is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")