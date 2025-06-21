import re
from typing import Optional, Tuple
import PyPDF2
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import io
import aiohttp
from pdfminer.pdfpage import PDFPage
import google.generativeai as genai
from django.conf import settings

from django.core.files.uploadedfile import InMemoryUploadedFile
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

from pdfminer.converter import PDFPageAggregator


class EnhancedLinkedInExtractor:
    def __init__(self):
        self.scrapingdog_api_key = "685677aff97326753bf58e05"
        self.api_url = "https://api.scrapingdog.com/linkedin"

    def extract_text_from_pdf(self, pdf_file) -> str:
        text = ""
        hyperlinks = []

        try:
            if isinstance(pdf_file, str):
                with open(pdf_file, "rb") as f:
                    pdf_content = f.read()
            elif isinstance(pdf_file, InMemoryUploadedFile):
                pdf_content = pdf_file.read()
            elif hasattr(pdf_file, "read"):
                pdf_content = pdf_file.read()
                if isinstance(pdf_content, str):
                    pdf_content = pdf_content.encode("utf-8")
            else:
                raise ValueError("Unsupported file type")

            pdf_file_text = io.BytesIO(pdf_content)
            pdf_file_links = io.BytesIO(pdf_content)

            # Extract text using pdfminer
            laparams = LAParams()
            text = extract_text(pdf_file_text, laparams=laparams)

            # Extract hyperlinks using PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file_links)
            for page in pdf_reader.pages:
                if "/Annots" in page:
                    for annot in page["/Annots"]:
                        obj = annot.get_object()
                        if "/A" in obj and "/URI" in obj["/A"]:
                            hyperlinks.append(obj["/A"]["/URI"])

            if hyperlinks:
                text += "\n=== EMBEDDED LINKS ===\n" + "\n".join(list(set(hyperlinks)))

            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)

            return self._clean_text(text)

        except Exception as e:
            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)
            raise e

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        # Remove extra whitespace while preserving newlines
        text = "\n".join(line.strip() for line in text.splitlines())

        # Remove duplicate newlines while preserving structured spacing
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Fix common PDF extraction issues
        text = text.replace("•", "\n•")  # Fix bullet points
        text = text.replace("|", " | ")  # Fix separator pipes

        # Remove non-printable characters except newlines and tabs
        text = "".join(char for char in text if char.isprintable() or char in "\n\t")

        return text.strip()

    async def extract_linkedin_info(self, text: str) -> Optional[Tuple[str, str]]:
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""Find the LinkedIn profile URL in this resume text. Return ONLY the username (what comes after linkedin.com/in/) and full URL in this format:
                USERNAME: <username>
                URL: <full url>
                If no valid LinkedIn URL is found, return "None".

                Resume text:
                {text}"""

            response = model.generate_content(prompt)
            response_text = response.text.strip()

            if "None" in response_text:
                return None

            username_match = re.search(r"USERNAME:\s*(\S+)", response_text)
            url_match = re.search(r"URL:\s*(https://[^\s]+)", response_text)

            if username_match and url_match:
                username = username_match.group(1).strip()
                full_url = url_match.group(1).strip()
                return username, full_url

            return None

        except Exception as e:
            print(f"Error extracting LinkedIn info: {e}")
            return None

    def _is_valid_username(self, username: str) -> bool:
        """
        Validate LinkedIn username format.
        """
        if not username:
            return False

        min_length = 3
        max_length = 100
        valid_chars = re.compile(r"^[\w\-]+$")

        return (
            min_length <= len(username) <= max_length
            and valid_chars.match(username)
            and not username.startswith("-")
            and not username.endswith("-")
        )

    async def fetch_linkedin_data(self, username: str) -> dict:
        """
        Fetch LinkedIn profile data using the username.
        """
        if not username:
            raise ValueError("LinkedIn username not found in resume")

        url = f"{self.api_url}?api_key={self.scrapingdog_api_key}&type=profile&linkId={username}&private=false"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                
                # ScrapingDog returns an array, get the first item
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                else:
                    raise Exception("No profile data found in response")
