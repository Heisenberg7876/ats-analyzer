import PyPDF2
import docx
import os


ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(file):
    """
    Extract text from PDF, DOCX, or TXT file objects.
    Returns cleaned text string or raises ValueError on failure.
    """
    filename = file.filename.lower()

    if filename.endswith('.pdf'):
        return _extract_pdf(file)
    elif filename.endswith('.docx'):
        return _extract_docx(file)
    elif filename.endswith('.txt'):
        return _extract_txt(file)
    else:
        raise ValueError(f"Unsupported file type. Please upload PDF, DOCX, or TXT.")


def _extract_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        if reader.is_encrypted:
            raise ValueError("PDF is encrypted. Please upload an unencrypted file.")
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        text = text.strip()
        if not text:
            raise ValueError("Could not extract text from PDF. It may be image-based — try a text-based PDF.")
        return text
    except PyPDF2.errors.PdfReadError as e:
        raise ValueError(f"Could not read PDF: {str(e)}")


def _extract_docx(file):
    try:
        doc = docx.Document(file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs).strip()
        if not text:
            raise ValueError("DOCX file appears to be empty.")
        return text
    except Exception as e:
        raise ValueError(f"Could not read DOCX: {str(e)}")


def _extract_txt(file):
    try:
        raw = file.read()
        text = raw.decode('utf-8', errors='replace').strip()
        if not text:
            raise ValueError("Text file appears to be empty.")
        return text
    except Exception as e:
        raise ValueError(f"Could not read TXT file: {str(e)}")
