from typing import Dict, List
import numpy as np


class EnhancedJobMatcher:
    def __init__(self, nlp_processor):
        self.nlp_processor = nlp_processor

    def calculate_match_score(self, resume_text: str, job_description: str) -> float:
        """Enhanced matching with semantic similarity"""

        # Extract skills from both
        resume_skills = set(skill.lower() for skill in self.nlp_processor.extract_skills(resume_text))
        job_skills = set(skill.lower() for skill in self.nlp_processor.extract_skills(job_description))

        # 1. Skill overlap score (exact matches)
        if not job_skills:
            skill_match_score = 0.0
        else:
            matching_skills = resume_skills.intersection(job_skills)
            skill_match_score = len(matching_skills) / len(job_skills)

        # 2. Semantic similarity score (captures related concepts)
        semantic_similarity = self.nlp_processor.get_semantic_similarity(resume_text, job_description)

        # 3. Experience matching
        job_experience_required = self.nlp_processor.extract_experience_years(job_description)
        resume_experience = self.nlp_processor.extract_experience_years(resume_text)

        if job_experience_required == 0:
            experience_score = 1.0
        else:
            experience_score = min(resume_experience / job_experience_required, 1.0)

        # Enhanced weighted scoring
        final_score = (
                0.35 * skill_match_score +  # 35% weight on exact skill matching
                0.40 * semantic_similarity +  # 40% weight on semantic similarity (NEW!)
                0.25 * experience_score  # 25% weight on experience
        )

        return round(final_score * 100, 2)

    def get_detailed_analysis(self, resume_text: str, job_description: str) -> Dict:
        """Enhanced analysis with semantic insights"""

        resume_skills = set(skill.lower() for skill in self.nlp_processor.extract_skills(resume_text))
        job_skills = set(skill.lower() for skill in self.nlp_processor.extract_skills(job_description))

        matching_skills = resume_skills.intersection(job_skills)
        missing_skills = job_skills - resume_skills
        extra_skills = resume_skills - job_skills

        # Calculate different similarity metrics
        semantic_similarity = self.nlp_processor.get_semantic_similarity(resume_text, job_description)
        keyword_similarity = self.nlp_processor.calculate_similarity(resume_text, job_description)

        job_experience_required = self.nlp_processor.extract_experience_years(job_description)
        resume_experience = self.nlp_processor.extract_experience_years(resume_text)

        match_score = self.calculate_match_score(resume_text, job_description)

        return {
            "overall_score": match_score,
            "skill_analysis": {
                "matching_skills": list(matching_skills),
                "missing_skills": list(missing_skills),
                "additional_skills": list(extra_skills),
                "skill_match_percentage": len(matching_skills) / len(job_skills) * 100 if job_skills else 0
            },
            "similarity_analysis": {
                "semantic_similarity": round(semantic_similarity * 100, 2),
                "keyword_similarity": round(keyword_similarity * 100, 2),
                "similarity_explanation": self._explain_similarity_difference(semantic_similarity, keyword_similarity)
            },
            "experience_analysis": {
                "required_years": job_experience_required,
                "candidate_years": resume_experience,
                "meets_requirement": resume_experience >= job_experience_required,
                "experience_gap": max(0, job_experience_required - resume_experience)
            },
            "recommendation": self._get_enhanced_recommendation(match_score, semantic_similarity, matching_skills,
                                                                job_skills),
            "improvement_suggestions": self._get_improvement_suggestions(missing_skills, job_experience_required,
                                                                         resume_experience)
        }

    def _explain_similarity_difference(self, semantic_sim: float, keyword_sim: float) -> str:
        """Explain the difference between semantic and keyword similarity"""
        diff = abs(semantic_sim - keyword_sim)

        if diff < 0.1:
            return "Semantic and keyword similarities are aligned"
        elif semantic_sim > keyword_sim:
            return "Strong conceptual match despite different terminology"
        else:
            return "Keywords match but concepts may differ"

    def _get_enhanced_recommendation(self, score: float, semantic_sim: float, matching_skills: set,
                                     job_skills: set) -> str:
        """Enhanced recommendation considering multiple factors"""
        skill_coverage = len(matching_skills) / len(job_skills) if job_skills else 0

        if score >= 80:
            return "Highly Recommended - Excellent overall match"
        elif score >= 70:
            return "Strongly Recommended - Very good match"
        elif score >= 60:
            return "Recommended - Good match with minor gaps"
        elif score >= 50:
            if semantic_sim > 0.7:
                return "Consider - Strong conceptual alignment despite skill gaps"
            else:
                return "Consider - Moderate match, may need training"
        elif score >= 40:
            if skill_coverage > 0.5:
                return "Weak Match - Has some relevant skills but significant gaps"
            else:
                return "Weak Match - Limited relevant experience"
        else:
            return "Not Recommended - Significant mismatch"

    def _get_improvement_suggestions(self, missing_skills: set, required_exp: int, candidate_exp: int) -> List[str]:
        """Provide suggestions for improving the match"""
        suggestions = []

        if missing_skills:
            top_missing = list(missing_skills)[:3]
            suggestions.append(f"Consider developing skills in: {', '.join(top_missing)}")

        exp_gap = required_exp - candidate_exp
        if exp_gap > 0:
            suggestions.append(f"Role requires {exp_gap} more years of experience")

        if not suggestions:
            suggestions.append("Strong candidate profile for this role")

        return suggestions