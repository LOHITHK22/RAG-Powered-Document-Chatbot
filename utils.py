import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_tokens=300):
    # Naive chunking by paragraph, fallback to splitting every N characters
    paragraphs = text.split('\n\n')
    chunks = []

    for para in paragraphs:
        if len(para.strip()) > 0:
            chunks.append(para.strip())

    # If still too long, break up into fixed-size chunks
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_tokens * 4:  # Roughly 4 chars/token
            for i in range(0, len(chunk), max_tokens * 4):
                final_chunks.append(chunk[i:i + max_tokens * 4])
        else:
            final_chunks.append(chunk)

    return final_chunks
