# ATS Resume Analyzer + Job Matching System

AI-powered resume scoring using Claude. Upload your resume and a job description to get an ATS score, keyword analysis, strengths, improvements, and section-by-section feedback.

## Features

- ATS score (0–100) with dimension breakdown (Technical, Experience, Education, Keywords)
- Matched and missing keywords
- 3 strengths + 5 actionable improvement suggestions
- Section-by-section resume feedback
- ATS formatting tips
- Experience level match (Perfect Fit / Overqualified / Slight Stretch / Underqualified)
- Supports PDF, DOCX, and TXT resumes

## Live Demo

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/a25bd984-e858-4041-8424-3341b26acde6" />

---

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/fac1a71b-07d2-4859-bbac-cd664da645ba" />

---

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/3fe8cd29-1af6-4567-9845-0c506b940afb" />

---
 

## Project Structure

```
ats_analyzer/
├── app.py                  # Flask app, routes
├── requirements.txt
├── .env.example
├── models/
│   └── analyzer.py         # Claude API analysis logic
├── utils/
│   └── parser.py           # Resume text extraction (PDF/DOCX/TXT)
├── templates/
│   ├── base.html
│   ├── index.html          # Upload form
│   └── dashboard.html      # Results page
└── static/
    ├── css/style.css
    └── js/main.js
```

## Setup

**1. Clone and enter project**
```bash
cd ats_analyzer
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment**
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

**5. Run**
```bash
python app.py
```

Open http://localhost:5000

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (required) | — |
| `FLASK_DEBUG` | Enable debug mode | `True` |
| `SECRET_KEY` | Flask secret key | `dev-secret-key` |
| `MAX_FILE_SIZE_MB` | Max upload size in MB | `10` |

## API Key

Get your key at https://console.anthropic.com
