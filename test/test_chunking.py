from src.data_gent.chunking import ParagraphChunker


def test_paragraph_chunker_basic():
    """Test basic paragraph splitting."""
    chunker = ParagraphChunker()
    text = """First paragraph here.

Second paragraph here.

Third paragraph."""

    chunks = chunker.chunk(text)

    assert len(chunks) == 3
    assert chunks[0] == "First paragraph here."
    assert chunks[1] == "Second paragraph here."
    assert chunks[2] == "Third paragraph."


def test_paragraph_chunker_multiline_paragraphs():
    """Test paragraphs with multiple lines."""
    chunker = ParagraphChunker()
    text = """First paragraph line 1.
First paragraph line 2.
First paragraph line 3.

Second paragraph line 1.
Second paragraph line 2."""

    chunks = chunker.chunk(text)

    assert len(chunks) == 2
    assert chunks[0] == "First paragraph line 1.\nFirst paragraph line 2.\nFirst paragraph line 3."
    assert chunks[1] == "Second paragraph line 1.\nSecond paragraph line 2."


def test_paragraph_chunker_multiple_blank_lines():
    """Test that multiple blank lines are treated the same as one."""
    chunker = ParagraphChunker()
    text = """First paragraph.


Second paragraph.



Third paragraph."""

    chunks = chunker.chunk(text)

    assert len(chunks) == 3
    assert chunks[0] == "First paragraph."
    assert chunks[1] == "Second paragraph."
    assert chunks[2] == "Third paragraph."


def test_paragraph_chunker_whitespace_handling():
    """Test that leading/trailing whitespace is stripped."""
    chunker = ParagraphChunker()
    text = """  First paragraph with leading space.

    Second paragraph with lots of whitespace.

Third paragraph."""

    chunks = chunker.chunk(text)

    assert len(chunks) == 3
    assert chunks[0] == "First paragraph with leading space."
    assert chunks[1] == "Second paragraph with lots of whitespace."
    assert chunks[2] == "Third paragraph."


def test_paragraph_chunker_empty_input():
    """Test that empty input returns empty list."""
    chunker = ParagraphChunker()
    chunks = chunker.chunk("")

    assert chunks == []


def test_paragraph_chunker_single_paragraph():
    """Test that a single paragraph returns a list with one item."""
    chunker = ParagraphChunker()
    text = "Just one paragraph here."

    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert chunks[0] == "Just one paragraph here."


def test_paragraph_chunker_data_dictionary_example():
    """Test a realistic data dictionary example."""
    chunker = ParagraphChunker()
    text = """user_id: Integer
Primary key for the users table. Auto-incrementing identifier.

email: String
User's email address. Must be unique across the system.

created_at: Timestamp
When the user account was created. Defaults to current timestamp."""

    chunks = chunker.chunk(text)

    assert len(chunks) == 3
    assert "user_id" in chunks[0]
    assert "Primary key" in chunks[0]
    assert "email" in chunks[1]
    assert "created_at" in chunks[2]


def test_paragraph_chunker_only_whitespace():
    """Test that input with only whitespace returns empty list."""
    chunker = ParagraphChunker()
    text = "   \n\n   \n   "

    chunks = chunker.chunk(text)

    assert chunks == []
