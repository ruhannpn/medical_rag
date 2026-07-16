import os
import fitz  # PyMuPDF
from PIL import Image

def extract_text_from_pdf(pdf_path):
    """
    Extract raw text from a PDF file.
    """
    document = fitz.open(pdf_path)
    full_text = ""
    for page_number in range(len(document)):
        page = document[page_number]
        text = page.get_text()
        full_text += text + "\n"
    document.close()
    return full_text

def extract_multimodal(file_path):
    """
    Extract text from documents. If it's an image or a scanned PDF (where text extraction yields little/nothing),
    it falls back to using Gemini 1.5 Flash Vision capabilities.
    """
    ext = file_path.lower().split('.')[-1]
    
    if ext == 'pdf':
        text = extract_text_from_pdf(file_path)
        # If the PDF contains native text, return it.
        if len(text.strip()) > 50:
            return text
            
        # Otherwise, assume it's a scanned PDF and fall back to vision.
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set. Cannot transcribe scanned PDF.")
            
        from google import genai
        client = genai.Client(api_key=api_key)
        
        full_text = ""
        doc = fitz.open(file_path)
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=[
                    img, 
                    "You are an expert medical scribe. Transcribe all printed text and handwriting in this scan exactly as written. Do not add any conversational filler. Just the transcribed text."
                ]
            )
            full_text += response.text + "\n"
        doc.close()
        return full_text
        
    elif ext in ['png', 'jpg', 'jpeg']:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set. Cannot transcribe image.")
            
        from google import genai
        client = genai.Client(api_key=api_key)
        
        img = Image.open(file_path)
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=[
                img, 
                "You are an expert medical scribe. Transcribe all printed text and handwriting in this scan exactly as written. Do not add any conversational filler. Just the transcribed text."
            ]
        )
        return response.text
        
    else:
        raise ValueError(f"Unsupported file type: {ext}")