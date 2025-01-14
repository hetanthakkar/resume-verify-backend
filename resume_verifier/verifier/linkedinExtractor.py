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

    async def extract_linkedin_info(self, text: str) -> Optional[Tuple[str, str]]:
        try:
            headers = {
                "x-api-key": "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",
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            }

            prompt = f"""Find the LinkedIn profile URL in this resume text. Return ONLY the username (what comes after linkedin.com/in/) and full URL in this format:
                USERNAME: <username>
                URL: <full url>
                If no valid LinkedIn URL is found, return "None".

                Resume text:
                {text}"""

            payload = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1024,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        return None

                    result = await response.json()
                    response_text = result["content"][0]["text"].strip()

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

        profile_url = f"https://www.linkedin.com/in/{username}/"
        headers = {"x-api-key": f"{self.auth_token}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.api_url, headers=headers, params={"li_profile_url": profile_url}
            ) as response:
                response.raise_for_status()
                return await response.json()
