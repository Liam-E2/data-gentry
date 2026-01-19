from io import BytesIO

from pytest import fixture
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from src.data_gentry.preprocessors import preprocess_pdf


@fixture
def minimal_pdf_bytes():
    """Create a minimal PDF with test content."""
    buffer = BytesIO()

    # Create a simple PDF with reportlab
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Hello World")
    c.drawString(100, 730, "This is a test PDF document.")
    c.showPage()
    c.save()

    # Reset buffer position to beginning
    buffer.seek(0)
    return buffer


def test_preprocess_pdf(minimal_pdf_bytes):
    """Test that PDF preprocessing converts PDF to markdown correctly."""
    markdown = preprocess_pdf(minimal_pdf_bytes)

    # Verify it returns a string
    assert isinstance(markdown, str)

    # Verify the markdown contains the expected text
    assert "Hello World" in markdown
    assert "This is a test PDF document." in markdown


def test_preprocess_pdf_empty():
    """Test that preprocessing an empty PDF still returns a string."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.showPage()  # Empty page
    c.save()
    buffer.seek(0)

    markdown = preprocess_pdf(buffer)

    assert isinstance(markdown, str)
    assert len(markdown) == 0
