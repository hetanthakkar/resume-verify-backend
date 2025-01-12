import re
from typing import Optional, Tuple
import PyPDF2
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import io
import aiohttp
from pdfminer.pdfpage import PDFPage

from django.core.files.uploadedfile import InMemoryUploadedFile
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

from pdfminer.converter import PDFPageAggregator


class EnhancedLinkedInExtractor:
    def __init__(self):
        self.auth_token = "ec2fc77d-4dc8-4939-8d11-96854f75a6c3"
        self.api_url = "https://gateway.getapihub.cloud/api/profile"

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

    def extract_linkedin_info(self, text: str) -> Optional[Tuple[str, str]]:
        print("text is", text)
        """
        Extract LinkedIn username or full URL using multiple patterns.
        Returns tuple of (username, full_url) or None if not found.
        """
        patterns = [
            r"linkedin\.com/in/([\w\-%.]+)/?",  # Standard profile URL
            r"linkedin\.com/pub/([\w\-%.]+)/?",  # Public profile URL
            r"linkedin\.com/profile/view\?id=([\w\-%.]+)",  # Profile view URL
            r"linkedin:?\s*(?:profile)?:?\s*([\w\-%.]+)",  # LinkedIn username
            r"(?:profile|connect).*linkedin\.com/(?:in|pub)/([\w\-%.]+)",  # Profile with context
            r"(?i)linkedin:?\s*(?:profile)?:?\s*([\w\-%.]+)",  # Case-insensitive username
            r"(?i)/?(?:in|pub)/([\w\-%.]+)",  # Short format
            r"@([\w\-%.]+)\s+(?:on\s+)?linkedin",  # @ format
            r"linkedin(?:.com)?[/\s]+(?:profile|in|pub)?[/\s]*([\w\-%.]+)",  # Generic format
        ]

        # Split text into lines to process embedded links section separately
        lines = text.split("\n")
        for line in lines:
            for pattern in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    username = match.group(1)
                    # Clean up the username
                    username = re.sub(r"[^\w\-]", "", username)
                    username = username.rstrip("/")
                    username = username.split("?")[0]  # Remove URL parameters

                    if self._is_valid_username(username):
                        full_url = f"https://www.linkedin.com/in/{username}/"
                        return username, full_url

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

        profile_url = f"https://www.linkedin.com/in/{username}/"
        headers = {"x-api-key": f"{self.auth_token}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.api_url, headers=headers, params={"li_profile_url": profile_url}
            ) as response:
                response.raise_for_status()
                return await response.json()
