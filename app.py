import os
from flask import Flask, render_template, request
from utils.parser import extract_text, allowed_file
from models.analyzer import analyze_resume

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        return render_template('index.html', error="No resume file uploaded.")

    file = request.files['resume']
    job_desc = request.form.get('job_desc', '').strip()

    if file.filename == '':
        return render_template('index.html', error="Please select a resume file.")

    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type. Please upload PDF, DOCX, or TXT.")

    if not job_desc or len(job_desc) < 30:
        return render_template('index.html', error="Please paste a job description (at least 30 characters).")

    try:
        resume_text = extract_text(file)
    except ValueError as e:
        return render_template('index.html', error=str(e))

    try:
        result = analyze_resume(resume_text, job_desc)
    except Exception as e:
        return render_template('index.html', error=f"Analysis failed: {str(e)}")

    return render_template('dashboard.html', **result)


@app.errorhandler(413)
def file_too_large(e):
    return render_template('index.html', error="File too large. Maximum size is 10MB."), 413


if __name__ == '__main__':
    app.run(debug=True, port=5000)
