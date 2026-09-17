"""
Comprehensive test suite for models/analyzer.py

Tests cover the core NLP/TF-IDF scoring engine, keyword extraction,
section detection, experience-level classification, and the full
analyze_resume() integration path.
"""
import pytest
import math
import sys
import os

# Ensure the project root is on sys.path so 'models' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.analyzer import (
    _clean,
    _tokenize,
    _bigrams,
    _extract_phrases,
    _tfidf_score,
    _keyword_analysis,
    _detect_sections,
    _section_feedback,
    _experience_level,
    _infer_job_title,
    _generate_strengths,
    _generate_improvements,
    _ats_tips,
    analyze_resume,
    STOP_WORDS,
    TECH_BOOST,
    SECTION_PATTERNS,
)


# ---------------------------------------------------------------------------
# _clean
# ---------------------------------------------------------------------------
class TestClean:
    def test_lowercases_text(self):
        assert _clean("Hello WORLD") == "hello world"

    def test_removes_punctuation(self):
        result = _clean("hello, world! foo@bar")
        assert "," not in result
        assert "!" not in result
        assert "@" not in result

    def test_collapses_whitespace(self):
        assert _clean("hello   world\t\nfoo") == "hello world foo"

    def test_strips_edges(self):
        assert _clean("  hello  ") == "hello"

    def test_empty_string(self):
        assert _clean("") == ""


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------
class TestTokenize:
    def test_removes_stop_words(self):
        tokens = _tokenize("the quick brown fox is very fast")
        for sw in ("the", "is", "very"):
            assert sw not in tokens

    def test_removes_short_tokens(self):
        tokens = _tokenize("I am an AI ML expert")
        # Words with len <= 2 should be filtered
        for t in tokens:
            assert len(t) > 2

    def test_returns_list(self):
        result = _tokenize("python developer with flask experience")
        assert isinstance(result, list)

    def test_empty_input(self):
        assert _tokenize("") == []


# ---------------------------------------------------------------------------
# _bigrams
# ---------------------------------------------------------------------------
class TestBigrams:
    def test_basic_bigrams(self):
        tokens = ["python", "flask", "developer"]
        result = _bigrams(tokens)
        assert "python flask" in result
        assert "flask developer" in result
        assert len(result) == 2

    def test_single_token(self):
        assert _bigrams(["python"]) == []

    def test_empty_list(self):
        assert _bigrams([]) == []


# ---------------------------------------------------------------------------
# _extract_phrases
# ---------------------------------------------------------------------------
class TestExtractPhrases:
    def test_finds_tech_boost_terms(self):
        text = "Experienced Python developer with Docker and Kubernetes skills"
        phrases = _extract_phrases(text)
        assert "python" in phrases
        assert "docker" in phrases
        assert "kubernetes" in phrases

    def test_includes_unigrams_and_bigrams(self):
        text = "machine learning engineer"
        phrases = _extract_phrases(text)
        # Should contain the bigram "machine learning" from TECH_BOOST
        assert "machine learning" in phrases

    def test_returns_set(self):
        result = _extract_phrases("python developer")
        assert isinstance(result, set)


# ---------------------------------------------------------------------------
# _tfidf_score
# ---------------------------------------------------------------------------
class TestTfidfScore:
    def test_identical_texts_high_similarity(self):
        text = "python flask developer with docker kubernetes aws experience"
        score = _tfidf_score(text, text)
        # Identical texts should have cosine similarity close to 1.0
        assert score > 0.9

    def test_completely_different_texts_low_similarity(self):
        resume = "expert baker specializing in sourdough bread and pastry arts"
        jd = "quantum physicist researching dark matter and particle acceleration"
        score = _tfidf_score(resume, jd)
        assert score < 0.3

    def test_partial_overlap_mid_similarity(self):
        resume = "python developer with flask and django experience building REST APIs"
        jd = "looking for python developer with react and flask skills for API development"
        score = _tfidf_score(resume, jd)
        assert 0.2 < score < 1.0

    def test_returns_float(self):
        score = _tfidf_score("python developer", "python developer")
        assert isinstance(score, float)

    def test_tech_boost_increases_weight(self):
        # A resume with matching tech terms should score higher than one without
        jd = "python flask docker kubernetes"
        resume_match = "python flask docker kubernetes developer"
        resume_no_match = "experienced professional leader coordinator"
        score_match = _tfidf_score(resume_match, jd)
        score_no_match = _tfidf_score(resume_no_match, jd)
        assert score_match > score_no_match


# ---------------------------------------------------------------------------
# _keyword_analysis
# ---------------------------------------------------------------------------
class TestKeywordAnalysis:
    def test_returns_tuple_of_two_lists(self):
        matched, missing = _keyword_analysis("python flask", "python flask django")
        assert isinstance(matched, list)
        assert isinstance(missing, list)

    def test_finds_matching_tech_keywords(self):
        resume = "I am a Python developer skilled in Flask and Docker"
        jd = "We need a Python developer with Flask and Docker experience"
        matched, _ = _keyword_analysis(resume, jd)
        assert "python" in matched
        assert "flask" in matched
        assert "docker" in matched

    def test_identifies_missing_keywords(self):
        resume = "I am a Python developer"
        jd = "We need a Python developer with Docker and Kubernetes and AWS experience"
        _, missing = _keyword_analysis(resume, jd)
        # Docker, Kubernetes, AWS should appear in missing
        assert any(kw in missing for kw in ["docker", "kubernetes", "aws"])

    def test_matched_capped_at_12(self):
        # Build a resume and JD with many overlapping tech terms
        terms = list(TECH_BOOST)[:20]
        text = " ".join(terms)
        matched, _ = _keyword_analysis(text, text)
        assert len(matched) <= 12

    def test_missing_capped_at_12(self):
        terms = list(TECH_BOOST)[:20]
        jd = " ".join(terms)
        resume = "completely unrelated text about gardening"
        _, missing = _keyword_analysis(resume, jd)
        assert len(missing) <= 12


# ---------------------------------------------------------------------------
# _detect_sections
# ---------------------------------------------------------------------------
class TestDetectSections:
    def test_detects_experience_section(self):
        resume = "John Doe\n\nExperience\nSoftware Engineer at Google\n5 years building APIs"
        sections = _detect_sections(resume)
        assert "experience" in sections
        assert "google" in sections["experience"].lower() or "software" in sections["experience"].lower()

    def test_detects_education_section(self):
        resume = "Name\n\nEducation\nBS Computer Science, MIT 2020"
        sections = _detect_sections(resume)
        assert "education" in sections
        assert "mit" in sections["education"].lower() or "computer" in sections["education"].lower()

    def test_detects_skills_section(self):
        resume = "Name\n\nSkills\nPython, Flask, Docker, AWS"
        sections = _detect_sections(resume)
        assert "skills" in sections
        assert "python" in sections["skills"].lower()

    def test_returns_dict_with_all_section_keys(self):
        sections = _detect_sections("some random text")
        for key in SECTION_PATTERNS:
            assert key in sections


# ---------------------------------------------------------------------------
# _section_feedback
# ---------------------------------------------------------------------------
class TestSectionFeedback:
    def test_missing_summary_feedback(self):
        sections = {k: '' for k in SECTION_PATTERNS}
        feedback = _section_feedback(sections, "any job description here")
        assert "summary" in feedback
        assert "no summary" in feedback["summary"].lower() or "add" in feedback["summary"].lower()

    def test_present_experience_feedback(self):
        sections = {k: '' for k in SECTION_PATTERNS}
        sections["experience"] = "Worked at Google for 5 years. Increased revenue by 30%. Managed team of 8 engineers. Built scalable microservices. Led migration to AWS. " * 3
        feedback = _section_feedback(sections, "software engineer role")
        assert "experience" in feedback


# ---------------------------------------------------------------------------
# _experience_level
# ---------------------------------------------------------------------------
class TestExperienceLevel:
    def test_perfect_fit(self):
        result = _experience_level("5 years experience in Python", "requires 5 years experience")
        assert result == "Perfect Fit"

    def test_overqualified(self):
        result = _experience_level("10 years experience", "requires 3 years experience")
        assert result == "Overqualified"

    def test_slight_stretch(self):
        result = _experience_level("2 years experience", "requires 5 years experience")
        assert result == "Slight Stretch"

    def test_underqualified(self):
        result = _experience_level("1 year experience", "requires 8 years experience")
        assert result == "Underqualified"

    def test_no_jd_years_defaults_perfect(self):
        result = _experience_level("5 years experience", "looking for a developer")
        assert result == "Perfect Fit"


# ---------------------------------------------------------------------------
# _infer_job_title
# ---------------------------------------------------------------------------
class TestInferJobTitle:
    def test_extracts_title_from_first_lines(self):
        jd = "Senior Python Developer\nWe are looking for an experienced developer..."
        result = _infer_job_title(jd)
        assert "python" in result.lower() or "developer" in result.lower()

    def test_defaults_to_target_role(self):
        # All lines are too long or end with period
        jd = "This is a very long line that definitely exceeds the eighty character limit set by the function and should not be picked as a title."
        result = _infer_job_title(jd)
        assert result == "Target Role"


# ---------------------------------------------------------------------------
# _generate_strengths
# ---------------------------------------------------------------------------
class TestGenerateStrengths:
    def test_returns_list_of_max_3(self):
        strengths = _generate_strengths(
            "python developer 10% growth",
            ["python", "flask"],
            {"skills": "python flask", "summary": "a " * 25, "experience": "", "education": "", "projects": "", "certifications": ""}
        )
        assert isinstance(strengths, list)
        assert len(strengths) <= 3

    def test_includes_keyword_alignment_when_matched(self):
        strengths = _generate_strengths(
            "python developer",
            ["python", "flask", "docker"],
            {k: '' for k in SECTION_PATTERNS}
        )
        assert any("keyword" in s.lower() or "alignment" in s.lower() for s in strengths)


# ---------------------------------------------------------------------------
# _generate_improvements
# ---------------------------------------------------------------------------
class TestGenerateImprovements:
    def test_returns_list_of_max_5(self):
        tips = _generate_improvements(
            ["docker", "kubernetes"],
            {k: '' for k in SECTION_PATTERNS},
            score=40
        )
        assert isinstance(tips, list)
        assert len(tips) <= 5

    def test_suggests_missing_keywords(self):
        tips = _generate_improvements(
            ["docker", "kubernetes"],
            {k: '' for k in SECTION_PATTERNS},
            score=40
        )
        assert any("missing" in t.lower() or "docker" in t.lower() for t in tips)


# ---------------------------------------------------------------------------
# _ats_tips
# ---------------------------------------------------------------------------
class TestAtsTips:
    def test_returns_list_of_max_3(self):
        tips = _ats_tips("python developer Jan 2022")
        assert isinstance(tips, list)
        assert len(tips) <= 3

    def test_date_detection(self):
        tips_with_dates = _ats_tips("Worked from Jan 2020 to Dec 2022")
        tips_no_dates = _ats_tips("worked at a company doing things")
        # With dates should mention "Dates detected"
        assert any("dates detected" in t.lower() for t in tips_with_dates)
        # Without dates should suggest adding them
        assert any("add dates" in t.lower() for t in tips_no_dates)


# ---------------------------------------------------------------------------
# analyze_resume (integration test)
# ---------------------------------------------------------------------------
class TestAnalyzeResume:
    SAMPLE_RESUME = """
    John Doe
    Software Engineer

    Summary
    Experienced Python developer with 5 years building scalable web applications
    using Flask, Django, and modern cloud infrastructure on AWS.

    Experience
    Senior Software Engineer at TechCorp (Jan 2020 - Present)
    - Built REST APIs serving 10M+ requests/day using Flask and Python
    - Reduced deployment time by 60% with Docker and Kubernetes
    - Led migration of monolith to microservices architecture

    Skills
    Python, Flask, Django, Docker, Kubernetes, AWS, PostgreSQL, Git, REST, API

    Education
    Bachelor of Science in Computer Science, State University, 2018
    """

    SAMPLE_JD = """
    Senior Python Developer
    We are looking for an experienced Python developer to join our backend team.

    Requirements:
    - 5+ years experience with Python
    - Strong knowledge of Flask or Django
    - Experience with Docker and Kubernetes
    - AWS cloud infrastructure experience
    - RESTful API design
    - PostgreSQL or MySQL database experience
    - Git version control
    - Bachelor's degree in Computer Science or related field
    """

    def test_returns_complete_result_dict(self):
        result = analyze_resume(self.SAMPLE_RESUME, self.SAMPLE_JD)
        required_keys = [
            'score', 'verdict', 'summary', 'job_title_match',
            'experience_match', 'skill_match', 'matched_keywords',
            'missing_keywords', 'strengths', 'improvements',
            'section_feedback', 'ats_tips',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_score_within_bounds(self):
        result = analyze_resume(self.SAMPLE_RESUME, self.SAMPLE_JD)
        assert 10 <= result['score'] <= 98

    def test_verdict_is_valid(self):
        result = analyze_resume(self.SAMPLE_RESUME, self.SAMPLE_JD)
        assert result['verdict'] in ('Strong Match', 'Good Match', 'Weak Match', 'Poor Match')

    def test_skill_match_has_dimensions(self):
        result = analyze_resume(self.SAMPLE_RESUME, self.SAMPLE_JD)
        for dim in ('technical', 'experience', 'education', 'keywords'):
            assert dim in result['skill_match']
            assert 0 <= result['skill_match'][dim] <= 100

    def test_matched_keywords_are_relevant(self):
        result = analyze_resume(self.SAMPLE_RESUME, self.SAMPLE_JD)
        # At least some of the known tech terms should be matched
        all_matched = [kw.lower() for kw in result['matched_keywords']]
        assert any(kw in all_matched for kw in ['python', 'flask', 'docker'])

    def test_strengths_and_improvements_are_lists(self):
        result = analyze_resume(self.SAMPLE_RESUME, self.SAMPLE_JD)
        assert isinstance(result['strengths'], list)
        assert isinstance(result['improvements'], list)
        assert len(result['strengths']) <= 3
        assert len(result['improvements']) <= 5

    def test_section_feedback_is_dict(self):
        result = analyze_resume(self.SAMPLE_RESUME, self.SAMPLE_JD)
        assert isinstance(result['section_feedback'], dict)

    def test_empty_resume_does_not_crash(self):
        """Edge case: empty resume should still return a valid result, not crash."""
        result = analyze_resume("", self.SAMPLE_JD)
        assert isinstance(result, dict)
        assert 'score' in result

    def test_empty_jd_does_not_crash(self):
        """Edge case: empty JD should still return a valid result, not crash."""
        result = analyze_resume(self.SAMPLE_RESUME, "")
        assert isinstance(result, dict)
        assert 'score' in result
