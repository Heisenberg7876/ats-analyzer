import re
import math
import string
from collections import Counter

STOP_WORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'by','from','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might',
    'shall','can','need','dare','ought','used','it','its','this','that',
    'these','those','i','we','you','he','she','they','me','us','him','her',
    'them','my','our','your','his','their','what','which','who','whom',
    'when','where','why','how','all','each','every','both','either','neither',
    'one','two','three','as','if','then','than','so','yet','not','no','nor',
    'just','about','above','after','before','between','into','through',
    'during','including','until','against','among','throughout','despite',
    'towards','upon','whether','while','although','because','since','unless',
    'however','therefore','thus','also','too','very','more','most','other',
    'such','own','same','than','there','here','any','some','well','new',
    'good','great','strong','work','working','works','able','ability',
    'experience','experienced','knowledge','understanding','skills','skill',
}

TECH_BOOST = {
    'python','java','javascript','typescript','react','angular','vue','node',
    'django','flask','fastapi','sql','nosql','mongodb','postgresql','mysql',
    'aws','azure','gcp','docker','kubernetes','git','linux','bash',
    'machine learning','deep learning','nlp','data science','tensorflow',
    'pytorch','pandas','numpy','spark','hadoop','kafka','redis','elasticsearch',
    'rest','api','graphql','microservices','agile','scrum','devops','mlops',
    'tableau','power bi','excel','r','scala','c++','c#','go','rust','swift',
    'html','css','sass','webpack','figma','jira','confluence',
    'selenium','pytest','junit','jenkins','terraform','ansible',
    'blockchain','cybersecurity','networking','cloud','saas','erp','crm',
    'leadership','management','communication','problem solving','teamwork',
    'analytical','strategic','project management','stakeholder','budget',
    'marketing','seo','sem','content','social media','analytics','growth',
    'sales','negotiation','customer','revenue','pipeline','forecasting',
    'research','design','ux','ui','wireframe','prototype','testing','qa',
}

SECTION_PATTERNS = {
    'summary': r'(summary|objective|profile|about|overview)',
    'experience': r'(experience|employment|work history|career|positions?|jobs?)',
    'skills': r'(skills?|technical|competencies|expertise|proficiencies)',
    'education': r'(education|academic|degree|university|college|qualification)',
    'projects': r'(projects?|portfolio|achievements?|accomplishments?)',
    'certifications': r'(certif|license|credential|award)',
}


def _clean(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _tokenize(text):
    return [w for w in _clean(text).split() if w not in STOP_WORDS and len(w) > 2]


def _bigrams(tokens):
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]


def _extract_phrases(text):
    tokens = _tokenize(text)
    bi = _bigrams(tokens)
    all_phrases = set(tokens) | set(bi)
    found = set()
    text_lower = text.lower()
    for term in TECH_BOOST:
        if term in text_lower:
            found.add(term)
    return all_phrases | found


def _tfidf_score(resume_text, jd_text):
    def tf(tokens):
        count = Counter(tokens)
        total = max(len(tokens), 1)
        return {w: c/total for w, c in count.items()}

    def tfidf_vec(tokens, idf_map):
        tf_map = tf(tokens)
        return {w: tf_map[w] * idf_map.get(w, 1.0) for w in tf_map}

    r_tokens = _tokenize(resume_text)
    j_tokens = _tokenize(jd_text)
    all_words = set(r_tokens) | set(j_tokens)

    idf = {}
    for w in all_words:
        in_docs = (w in set(r_tokens)) + (w in set(j_tokens))
        idf[w] = math.log((2 + 1) / (in_docs + 1)) + 1

    for w in all_words:
        if w in TECH_BOOST:
            idf[w] *= 2.5

    rv = tfidf_vec(r_tokens, idf)
    jv = tfidf_vec(j_tokens, idf)

    dot = sum(rv.get(w, 0) * jv.get(w, 0) for w in jv)
    norm_r = math.sqrt(sum(v**2 for v in rv.values())) or 1
    norm_j = math.sqrt(sum(v**2 for v in jv.values())) or 1
    return dot / (norm_r * norm_j)


def _keyword_analysis(resume_text, jd_text):
    jd_phrases = _extract_phrases(jd_text)
    resume_phrases = _extract_phrases(resume_text)
    jd_words = _tokenize(jd_text)
    jd_freq = Counter(jd_words)
    important = {w for w, c in jd_freq.items() if c >= 2 or w in TECH_BOOST}
    important |= {p for p in jd_phrases if p in TECH_BOOST}
    matched = sorted(important & resume_phrases)[:14]
    missing = sorted(important - resume_phrases)[:14]
    missing_tech = [k for k in missing if k in TECH_BOOST]
    missing_other = [k for k in missing if k not in TECH_BOOST]
    missing = (missing_tech + missing_other)[:12]
    matched = matched[:12]
    return matched, missing


def _detect_sections(resume_text):
    lines = resume_text.split('\n')
    sections = {k: '' for k in SECTION_PATTERNS}
    current = None
    for line in lines:
        line_lower = line.lower().strip()
        matched_section = None
        for sec, pat in SECTION_PATTERNS.items():
            if re.search(pat, line_lower) and len(line_lower) < 60:
                matched_section = sec
                break
        if matched_section:
            current = matched_section
        elif current:
            sections[current] += ' ' + line
    return sections


def _section_feedback(sections, jd_text):
    feedback = {}
    s = sections.get('summary', '').strip()
    if not s:
        feedback['summary'] = 'No summary/objective section found. Add a 2-3 sentence professional summary tailored to the role.'
    elif len(s.split()) < 20:
        feedback['summary'] = 'Summary is very brief. Expand it to highlight your key value proposition for this specific role.'
    else:
        feedback['summary'] = 'Summary section present. Make sure it directly references the target role and your top relevant skills.'

    e = sections.get('experience', '').strip()
    if not e:
        feedback['experience'] = 'No clear experience section detected. Ensure your work history has a clear heading.'
    elif len(e.split()) < 50:
        feedback['experience'] = 'Experience section seems thin. Add bullet points with measurable achievements (numbers, percentages, impact).'
    else:
        has_numbers = bool(re.search(r'\d+[%+]?|\$\d+', e))
        if has_numbers:
            feedback['experience'] = 'Experience section looks solid with quantified achievements. Ensure bullets start with strong action verbs.'
        else:
            feedback['experience'] = 'Experience found but lacks quantified results. Add metrics (e.g., "Increased sales by 30%", "Managed team of 8").'

    sk = sections.get('skills', '').strip()
    if not sk:
        feedback['skills'] = 'No dedicated skills section found. Add one with a comma-separated list of your technical and soft skills.'
    else:
        feedback['skills'] = 'Skills section present. Ensure it includes the exact keywords from the job description for ATS matching.'

    ed = sections.get('education', '').strip()
    if not ed:
        feedback['education'] = 'No education section detected. Add your highest qualification even if older.'
    else:
        feedback['education'] = 'Education section found. Include graduation year and GPA if strong (above 3.5/8.0).'

    return feedback


def _experience_level(resume_text, jd_text):
    years_in_resume = re.findall(r'(\d+)\+?\s*years?', resume_text.lower())
    years_in_jd = re.findall(r'(\d+)\+?\s*years?', jd_text.lower())
    resume_exp = max((int(y) for y in years_in_resume), default=0)
    jd_exp = max((int(y) for y in years_in_jd), default=0)
    if jd_exp == 0:
        return 'Perfect Fit'
    diff = resume_exp - jd_exp
    if diff >= 4:
        return 'Overqualified'
    elif diff >= -1:
        return 'Perfect Fit'
    elif diff >= -3:
        return 'Slight Stretch'
    else:
        return 'Underqualified'


def _infer_job_title(jd_text):
    lines = [l.strip() for l in jd_text.split('\n') if l.strip()]
    for line in lines[:8]:
        if 5 < len(line) < 80 and not line.endswith('.'):
            clean = re.sub(r'(job title|position|role)[:\-\s]+', '', line, flags=re.I)
            if clean:
                return clean.title()
    return 'Target Role'


def _generate_strengths(resume_text, matched, sections):
    strengths = []
    r_lower = resume_text.lower()
    if matched:
        top = ', '.join(matched[:3])
        strengths.append(f"Strong keyword alignment — '{top}' all appear in your resume.")
    has_numbers = bool(re.search(r'\d+[%+]|\$[\d,]+|\d+ (people|team|projects?)', r_lower))
    if has_numbers:
        strengths.append("Resume includes quantified achievements which ATS systems and recruiters favor.")
    if sections.get('skills', '').strip():
        strengths.append("Dedicated skills section helps ATS parsers quickly identify your competencies.")
    if sections.get('summary', '').strip() and len(sections['summary'].split()) > 20:
        strengths.append("Professional summary provides strong first impression and context for recruiters.")
    if sections.get('certifications', '').strip():
        strengths.append("Certifications listed — strong differentiators for technical roles.")
    if len(strengths) < 3:
        strengths.append("Resume structure is readable and can be parsed by standard ATS systems.")
    return strengths[:3]


def _generate_improvements(missing, sections, score):
    tips = []
    if missing:
        top_missing = ', '.join(missing[:4])
        tips.append(f"Add these missing keywords to your resume: {top_missing}.")
    if not sections.get('summary', '').strip():
        tips.append("Add a 2-3 sentence professional summary at the top tailored to this specific role.")
    if not sections.get('skills', '').strip():
        tips.append("Create a dedicated 'Skills' section listing both technical tools and soft skills.")
    no_numbers = not bool(re.search(r'\d+[%+]|\$[\d,]+', sections.get('experience', '')))
    if no_numbers:
        tips.append("Quantify achievements — add numbers, percentages, or dollar amounts to bullet points.")
    if score < 60:
        tips.append("Tailor your resume to this job by mirroring its exact language and keywords.")
    tips.append("Use a clean single-column format with standard headings for maximum ATS compatibility.")
    tips.append("Submit as .docx or plain-text PDF — avoid tables, columns, and graphics.")
    return tips[:5]


def _ats_tips(resume_text):
    tips = []
    r_lower = resume_text.lower()
    if not re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4})', r_lower):
        tips.append("Add dates (month + year) to all work experience entries — ATS uses them to calculate tenure.")
    else:
        tips.append("Dates detected. Use a consistent format (e.g., Jan 2022 - Mar 2024) throughout your resume.")
    tips.append("Use simple single-column formatting — ATS systems struggle with tables and multi-column layouts.")
    tips.append("Mirror exact phrases from the job description — ATS matches keywords literally, not by meaning.")
    return tips[:3]


def analyze_resume(resume_text, job_desc):
    similarity = _tfidf_score(resume_text, job_desc)
    matched, missing = _keyword_analysis(resume_text, job_desc)
    total_important = len(matched) + len(missing)
    kw_ratio = len(matched) / max(total_important, 1)
    sections = _detect_sections(resume_text)

    technical = min(100, int(kw_ratio * 100 * 1.1 + similarity * 30))
    experience_score = min(100, int(similarity * 90 + (20 if sections.get('experience') else 0)))
    education_score = min(100, 60 + (30 if sections.get('education') else 0) + (10 if 'bachelor' in resume_text.lower() or 'master' in resume_text.lower() else 0))
    keywords_score = min(100, int(kw_ratio * 100))

    raw_score = (technical * 0.35 + experience_score * 0.30 + keywords_score * 0.25 + education_score * 0.10)
    score = max(10, min(98, int(raw_score)))

    if score >= 75:
        verdict = 'Strong Match'
    elif score >= 55:
        verdict = 'Good Match'
    elif score >= 35:
        verdict = 'Weak Match'
    else:
        verdict = 'Poor Match'

    matched_count = len(matched)
    missing_count = len(missing)
    summary = (
        f"Your resume matches {score}% of the job requirements with {matched_count} key skills aligned. "
        f"{'You are a competitive candidate — refine your summary and quantify achievements to stand out.' if score >= 60 else f'Adding {min(missing_count, 5)} missing keywords and tailoring your experience section could significantly improve your ATS ranking.'}"
    )

    return {
        'score': score,
        'verdict': verdict,
        'summary': summary,
        'job_title_match': _infer_job_title(job_desc),
        'experience_match': _experience_level(resume_text, job_desc),
        'skill_match': {
            'technical': technical,
            'experience': experience_score,
            'education': education_score,
            'keywords': keywords_score,
        },
        'matched_keywords': matched,
        'missing_keywords': missing,
        'strengths': _generate_strengths(resume_text, matched, sections),
        'improvements': _generate_improvements(missing, sections, score),
        'section_feedback': _section_feedback(sections, job_desc),
        'ats_tips': _ats_tips(resume_text),
    }
