import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    """
    Extract all text from a PDF file.

    Parameters:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Combined text from all pages
    """
    document = fitz.open(pdf_path)
    full_text = ""

    for page_number in range(len(document)):
        page = document[page_number]
        text = page.get_text()
        full_text += text + "\n"

    document.close()

    return full_text