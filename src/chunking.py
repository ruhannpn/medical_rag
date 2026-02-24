def chunk_text_by_sections(text):
    """
    Split text into structured chunks based on known medical section headers.
    Returns list of dictionaries:
    [
        {"section": "Prescription", "content": "..."},
        ...
    ]
    """

    section_headers = [
        "Patient Information",
        "Medical History",
        "Examination Findings",
        "Laboratory Results",
        "Diagnosis",
        "Prescription",
        "Recommendations"
    ]

    chunks = []
    current_section = None
    current_content = []

    lines = text.split("\n")

    for line in lines:
        clean_line = line.strip()

        # If line matches a known section header
        if clean_line in section_headers:
            # Save previous section
            if current_section is not None:
                chunks.append({
                    "section": current_section,
                    "content": "\n".join(current_content).strip()
                })

            # Start new section
            current_section = clean_line
            current_content = []

        else:
            if current_section is not None:
                current_content.append(clean_line)

    # Add last section
    if current_section is not None:
        chunks.append({
            "section": current_section,
            "content": "\n".join(current_content).strip()
        })

    return chunks