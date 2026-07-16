from __future__ import annotations
import re
import json
from groq import Groq

# Reuse the medication list pattern from the original implementation
_MED_PATTERN = re.compile(
    r"\b(metformin|lisinopril|amlodipine|atorvastatin|aspirin|salbutamol|albuterol|"
    r"montelukast|fluticasone|budesonide|salmeterol|omeprazole|losartan|"
    r"hydrochlorothiazide|glipizide|sitagliptin|insulin|prednisone|ibuprofen|"
    r"paracetamol|acetaminophen|cetirizine|loratadine|amoxicillin|azithromycin)\b",
    re.IGNORECASE,
)

class ClinicalExtractor:
    @staticmethod
    def _find(pattern: str, text: str, group: int = 1) -> str | None:
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(group).strip() if m else None

    @classmethod
    def extract_name(cls, text: str) -> str | None:
        return cls._find(r"Name:\s*(.+)", text)

    @classmethod
    def extract_age(cls, text: str) -> int | None:
        val = cls._find(r"Age:\s*(\d+)", text)
        return int(val) if val and val.isdigit() else None

    @classmethod
    def extract_diagnosis(cls, text: str) -> str | None:
        return cls._find(r"Diagnosis[:\s]+(.+)", text)

    @classmethod
    def extract_gender(cls, text: str) -> str | None:
        return cls._find(r"(?:Gender|Sex)[:\s]+(.+)", text)

    @classmethod
    def extract_dob(cls, text: str) -> str | None:
        return cls._find(r"(?:DOB|Date of Birth)[:\s]+(.+)", text)

    @classmethod
    def extract_visit_date(cls, text: str) -> str | None:
        return cls._find(r"(?:Visit Date|Date of Visit|Appointment Date)[:\s]+(.+)", text)

    @classmethod
    def extract_medications(cls, text: str) -> list[str]:
        candidates: list[str] = []

        block = re.search(
            r"(?:Medications?|Prescribed Medications?|Current Medications?)[:\s]*\n(.*?)"
            r"(?=\n[A-Z][^\n]{0,50}:|\Z)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if block:
            for line in block.group(1).splitlines():
                line = re.sub(r"^[-•*\d.\s]+", "", line).strip()
                if line:
                    candidates.append(line)

        if not candidates:
            inline = cls._find(r"(?:Medications?|Prescribed)[:\s]+(.+)", text)
            if inline:
                candidates = [s.strip() for s in re.split(r"[,;]", inline) if s.strip()]

        if not candidates:
            found = _MED_PATTERN.findall(text)
            candidates = list(dict.fromkeys(m.capitalize() for m in found))

        return [
            c for c in candidates
            if 1 <= len(c.split()) <= 5
            and re.match(r"[A-Z]", c)
            and not re.search(r"\b(after|follow|month|week|day|return|visit|per|as needed)\b", c, re.I)
        ]

    @classmethod
    def extract_symptoms(cls, text: str) -> list[str]:
        block = re.search(
            r"(?:Symptoms?|Chief Complaint|Presenting Complaints?)[:\s]*\n(.*?)"
            r"(?=\n[A-Z][^\n]{0,50}:|\Z)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if block:
            items = re.findall(r"[-•*\d.]*\s*([A-Za-z][^\n,;]{2,})", block.group(1))
            if items:
                return [i.strip() for i in items]

        inline = cls._find(r"(?:Symptoms?|Chief Complaint)[:\s]+(.+)", text)
        if inline:
            return [s.strip() for s in re.split(r"[,;]", inline) if s.strip()]
        return []

    @classmethod
    def extract_allergies(cls, text: str) -> list[str]:
        inline = cls._find(r"Allergies?[:\s]+(.+)", text)
        if inline and inline.lower() not in ("none", "nkda", "none known", "n/a"):
            return [a.strip() for a in re.split(r"[,;]", inline) if a.strip()]
        return []

    @classmethod
    def extract_with_regex(cls, text: str) -> dict:
        """
        Runs the legacy fast regex extraction on the text.
        """
        return {
            "name": cls.extract_name(text),
            "age": cls.extract_age(text),
            "gender": cls.extract_gender(text),
            "dob": cls.extract_dob(text),
            "visit_date": cls.extract_visit_date(text),
            "diagnosis": cls.extract_diagnosis(text),
            "medications": cls.extract_medications(text),
            "symptoms": cls.extract_symptoms(text),
            "allergies": cls.extract_allergies(text),
        }

    @classmethod
    def extract_with_llm(cls, client: Groq, text: str) -> dict:
        """
        Uses LLM (JSON mode) to extract clinical entities.
        """
        try:
            prompt = (
                "You are an expert clinical records extractor. "
                "Analyze the medical report below and extract the following patient details:\n"
                "1. Name (patient's full name)\n"
                "2. Age (integer or null)\n"
                "3. Gender (string or null)\n"
                "4. DOB (string or null)\n"
                "5. Visit Date (string or null)\n"
                "6. Diagnosis (primary clinical diagnosis string or null)\n"
                "7. Medications (array of active prescribed medications, including dosage/frequency if mentioned)\n"
                "8. Symptoms (array of symptoms or chief complaints)\n"
                "9. Allergies (array of known drug/food allergies, return empty array if none/NKDA)\n\n"
                "Respond ONLY with a valid JSON object matching the schema below:\n"
                "{\n"
                '  "name": "string or null",\n'
                '  "age": integer or null,\n'
                '  "gender": "string or null",\n'
                '  "dob": "string or null",\n'
                '  "visit_date": "string or null",\n'
                '  "diagnosis": "string or null",\n'
                '  "medications": ["string"],\n'
                '  "symptoms": ["string"],\n'
                '  "allergies": ["string"]\n'
                "}\n\n"
                f"Medical Report:\n{text}"
            )

            # Attempt JSON Mode
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a precise JSON clinical data extractor. Output ONLY JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=1000
            )

            data = json.loads(response.choices[0].message.content.strip())
            
            # Type correction and validation
            if not isinstance(data.get("medications"), list):
                data["medications"] = []
            if not isinstance(data.get("symptoms"), list):
                data["symptoms"] = []
            if not isinstance(data.get("allergies"), list):
                data["allergies"] = []
            if data.get("age") is not None:
                try:
                    data["age"] = int(data["age"])
                except (ValueError, TypeError):
                    data["age"] = None
                    
            return data
        except Exception as e:
            # If API fails or parsing fails, return None (calling code can fallback to regex)
            print(f"[Warning] LLM clinical extraction failed: {e}. Falling back to regex.")
            return None

    @classmethod
    def extract(cls, text: str, client: Groq = None) -> dict:
        """
        Extraction entry point: uses LLM if client is available, otherwise regex.
        If LLM fails or is incomplete, uses regex values as fallback.
        """
        regex_data = cls.extract_with_regex(text)
        
        if client:
            llm_data = cls.extract_with_llm(client, text)
            if llm_data:
                # Merge: prefer LLM data, but fallback to regex if LLM field is missing or null
                merged = {}
                for key in regex_data.keys():
                    llm_val = llm_data.get(key)
                    # For lists, prefer LLM if it's not empty, otherwise fallback to regex list if it has items
                    if isinstance(regex_data[key], list):
                        merged[key] = llm_val if llm_val else regex_data[key]
                    else:
                        merged[key] = llm_val if llm_val is not None else regex_data[key]
                return merged
                
        return regex_data
