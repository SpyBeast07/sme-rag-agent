import os
import uuid
import smtplib
from typing import Optional, Tuple
from fpdf import FPDF
from docx import Document
from email.message import EmailMessage
from pathlib import Path

# output folder for generated files
OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _ensure_filename(filename: Optional[str], ext: str) -> str:
    if filename:
        fn = Path(filename)
        if fn.suffix != ext:
            fn = fn.with_suffix(ext)
        return str(OUT_DIR / fn.name)
    # generate unique
    return str(OUT_DIR / f"{uuid.uuid4().hex}{ext}")

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import utils
import re

def generate_filename_from_query(query: str, ext: str = ".pdf") -> str:
    """
    Generate filenames like:
        quiz_salmonella.pdf
        handout_allergens.docx
        summary_food_poisoning.pdf

    Extracts:
        - task (quiz, summary, handout, report, guideline, analysis)
        - topic (salmonella, allergens, food poisoning, etc.)

    Falls back safely if extraction fails.
    """

    q = query.lower()

    # 1️⃣ Extract TASK keyword
    task_keywords = [
        "quiz", "handout", "summary", "report",
        "guideline", "guide", "analysis"
    ]
    task = "document"
    for t in task_keywords:
        if t in q:
            task = t
            break
    if task == "guide":
        task = "guideline"

    # 2️⃣ Remove emails
    q = re.sub(r"\S+@\S+", "", q)

    # 3️⃣ Extract topic
    topic = None
    topic_match = re.search(r"(on|about)\s+([a-zA-Z ]+)", q)
    if topic_match:
        topic = topic_match.group(2)

    # Fallback → last 2–3 words
    if not topic:
        topic = " ".join(q.split()[-3:])

    # 4️⃣ Clean topic
    topic = topic.strip()
    topic = re.sub(r"[^a-zA-Z0-9 ]", " ", topic)
    topic = re.sub(r"\s+", "_", topic)

    # Remove stop words
    stop_words = ["create", "make", "generate", "send", "give", "it", "to", "me", "pdf", "doc", "docx", "mail", "email"]
    topic = "_".join([w for w in topic.split("_") if w not in stop_words])

    if not topic:
        topic = "output"

    # 5️⃣ Clean task
    task = re.sub(r"[^a-zA-Z0-9]", "", task)

    # 6️⃣ Build filename
    filename = f"{task}_{topic}{ext}".lower()
    filename = re.sub(r"_+", "_", filename).strip("_")

    # 7️⃣ Trim long filenames
    if len(filename) > 80:
        filename = filename[:80] + ext

    return filename

def clean_text_for_pdf(text: str) -> str:
    """Prevents ReportLab crashes by sanitizing the content."""
    # Remove XML tokens
    text = re.sub(r"<\/?s>", " ", text)

    # Replace long continuous strings ( >80 chars ) with line breaks
    text = re.sub(r"(\S{80,})", lambda m: m.group(1)[:80] + "\n" + m.group(1)[80:], text)

    # Convert multiple newlines to one
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text

def generate_pdf_report(text: str, filename: Optional[str] = None, query: Optional[str] = None):
    """Safe PDF generation (prevents ReportLab width errors)."""
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4

    text = clean_text_for_pdf(text)
    
    if query and not filename:
        filename = generate_filename_from_query(query, ".pdf")
    elif not filename:
        filename = "report.pdf"

    doc_path = OUT_DIR / filename
    styles = getSampleStyleSheet()
    story = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 12))
            continue
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 6))

    doc = SimpleDocTemplate(str(doc_path), pagesize=A4)
    doc.build(story)

    return str(doc_path)

def generate_docx_report(content: str, filename: Optional[str] = None, query: Optional[str] = None) -> str:
    """
    Generate a DOCX from content and return absolute path.
    """
    if query and not filename:
        filename = generate_filename_from_query(query, ".docx")
        filename = filename.lower()
    elif not filename:
        filename = "report.docx"
    out_path = _ensure_filename(filename, ".docx")
    doc = Document()
    for para in content.split("\n"):
        doc.add_paragraph(para)
    doc.save(out_path)
    return out_path

def send_email(
    to_email: str,
    subject: str,
    body: str,
    attachments: Optional[list] = None,
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
    smtp_user: Optional[str] = None,
    smtp_pass: Optional[str] = None,
    use_tls: bool = True
) -> Tuple[bool, str]:
    """
    Send email with optional attachments.
    Returns (success, message).
    IMPORTANT: For Gmail, use an app password, not your normal password.
    Environment variables fallback: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS
    """
    smtp_host = smtp_host or os.getenv("SMTP_HOST")
    smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
    smtp_user = smtp_user or os.getenv("SMTP_USER")
    smtp_pass = smtp_pass or os.getenv("SMTP_PASS")

    if not (smtp_host and smtp_user and smtp_pass):
        return False, "SMTP credentials not configured (SMTP_HOST/SMTP_USER/SMTP_PASS)."
    
    if not to_email:
        return False, "No email address provided"

    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    # attach files
    attachments = attachments or []
    for path in attachments:
        try:
            with open(path, "rb") as f:
                data = f.read()
            maintype = "application"
            subtype = "octet-stream"
            filename = Path(path).name
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
        except Exception as e:
            return False, f"Failed to attach {path}: {e}"

    try:
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=90)
        if use_tls:
            server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully."
    except Exception as e:
        return False, f"SMTP error: {e}"