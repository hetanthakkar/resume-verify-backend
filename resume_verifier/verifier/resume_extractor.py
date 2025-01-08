import json
import openai
import re
from typing import Dict, List, Tuple
import asyncio
import PyPDF2
import requests
from .projectverifier import ProjectVerifier
from django.core.files.uploadedfile import InMemoryUploadedFile
from urllib.parse import urlparse
from pathlib import Path


class ResumeProjectExtractor:
    def __init__(self, openai_key: str):
        self.openai_key = openai_key
        openai.api_key = openai_key

    def is_valid_url(self, url: str) -> bool:
        """Check if the given string is a valid URL."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def get_pdf_file(self, pdf_path) -> str:
        """
        Handle URL, local file paths, and Django's InMemoryUploadedFile.
        Returns the path to a local PDF file.
        """
        # Handle Django's InMemoryUploadedFile
        if isinstance(pdf_path, InMemoryUploadedFile):
            try:
                # Create temp directory if it doesn't exist
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)

                # Save temporarily
                temp_pdf = temp_dir / pdf_path.name
                with open(temp_pdf, "wb") as file:
                    file.write(pdf_path.read())
                return str(temp_pdf)
            except Exception as e:
                raise Exception(f"Error saving uploaded file: {e}")

        # Handle URL and local paths as before
        return super().get_pdf_file(pdf_path)

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """Extract text and links from PDF, handling both local files and URLs."""
        try:
            local_pdf_path = self.get_pdf_file(pdf_path)
            text = ""
            links_with_context = []
            current_position = 0

            with open(local_pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    # Extract text
                    page_text = page.extract_text() or ""
                    text += page_text

                    # Extract links from annotations with their positions
                    if "/Annots" in page:
                        annotations = page["/Annots"]
                        for annotation in annotations:
                            annotation_object = annotation.get_object()
                            if annotation_object.get("/Subtype") == "/Link":
                                if (
                                    "/A" in annotation_object
                                    and "/URI" in annotation_object["/A"]
                                ):
                                    url = annotation_object["/A"]["/URI"]
                                    # Get surrounding text (100 characters before and after)
                                    start_pos = max(0, current_position - 100)
                                    end_pos = min(len(text), current_position + 100)
                                    context = text[start_pos:end_pos]
                                    links_with_context.append(
                                        {
                                            "url": url,
                                            "position": current_position,
                                            "context": context,
                                            "page": page_num + 1,
                                        }
                                    )

                    current_position += len(page_text)

            # Extract URLs from text using regex with context
            url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            for match in re.finditer(url_pattern, text):
                start_pos = max(0, match.start() - 100)
                end_pos = min(len(text), match.end() + 100)
                context = text[start_pos:end_pos]
                links_with_context.append(
                    {
                        "url": match.group(),
                        "position": match.start(),
                        "context": context,
                        "page": None,
                    }
                )

            # Clean up temporary file if it was downloaded
            if self.is_valid_url(pdf_path):
                Path(local_pdf_path).unlink()
                Path("temp").rmdir()

            return text, links_with_context
        except Exception as e:
            # Clean up temporary files in case of error
            if self.is_valid_url(pdf_path) and Path("temp/temp_resume.pdf").exists():
                Path("temp/temp_resume.pdf").unlink()
                Path("temp").rmdir()
            print(f"Error extracting PDF text: {e}")
            return None, []

    def extract_projects(self, text: str, links_with_context: List[Dict]) -> List[Dict]:
        """Extract projects and their URLs from resume text using Claude."""
        prompt = """Extract all projects from the resume text. Pay special attention to maintaining the correct relationship between projects and their URLs.

        For each project found in the text:
        1. Look for URLs that appear in close proximity to the project description
        2. Only associate URLs that are explicitly linked to that specific project
        3. Consider the context where each URL appears

        Here are the links found in the PDF with their surrounding context:
        {links_context}

        Return as valid JSON array with this structure:
        [{
            "name": "Project Name",
            "description": "Full project description",
            "technologies": ["tech1", "tech2"],
            "url": "URL that appears next to this specific project",
            "context": "The text surrounding where this project appears",
            "confidence": "HIGH if URL appears right next to project, LOW if uncertain"
        }]"""

        try:
            # Prepare links context for the prompt
            links_context = "\n".join(
                [
                    f"URL: {link['url']}\nContext: {link['context']}\n"
                    for link in links_with_context
                ]
            )

            # Format the full prompt
            full_prompt = f"\n\nHuman: {prompt}\n\nText: {text}\n\nLinks Context:\n{links_context}\n\nAssistant:"

            # Call Anthropic's Claude API
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": "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",
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={  # Changed from data to json
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "user", "content": full_prompt}
                    ],  # Removed extra curly braces
                },
            )

            if response.status_code != 200:
                raise Exception(f"Claude API error: {response.text}")

            # Parse the response properly
            response_data = response.json()
            content = response_data["content"][0][
                "text"
            ]  # Updated to match current API response structure

            json_start = content.find("[")
            json_end = content.rfind("]") + 1

            if json_start >= 0 and json_end > json_start:
                projects = json.loads(content[json_start:json_end])

                # Only keep URLs with high confidence matches
                for project in projects:
                    if project.get("confidence", "").upper() != "HIGH":
                        project["url"] = None

                print(f"Extracted {len(projects)} projects")
                return projects
            return []
        except Exception as e:
            print(f"Error extracting projects: {e}")
            return []


class ProjectVerificationSystem:
    def __init__(self, openai_key: str, github_token: str = None):
        self.extractor = ResumeProjectExtractor(openai_key)
        self.verifier = ProjectVerifier(openai_key, github_token)

    async def verify_all_projects(self, pdf_path: str) -> Dict:
        """Extract and verify all projects from a resume PDF."""
        # Extract text and links from PDF
        resume_text, links = self.extractor.extract_text_from_pdf(pdf_path)
        if not resume_text:
            return {"error": "Failed to extract text from PDF"}

        print(f"Found {len(links)} links in PDF")

        # Extract projects with corresponding links
        projects = self.extractor.extract_projects(resume_text, links)
        print(f"Extracted projects: {json.dumps(projects, indent=2)}")

        # Create a class for null verification
        class NullVerification:
            similarity_score = 0

            def create_structured_output(self):
                return {
                    "repository_exists": False,
                    "description_match": "No URL provided",
                    "verification_notes": "Project URL not available for verification",
                }

        # Verify each project that has a URL
        verification_results = []
        for project in projects:
            if project.get("url"):
                result = await self.verifier.verify_project(
                    project["url"], project["description"]
                )
                verification_results.append(
                    {"project": project, "verification": result}
                )
            else:
                # Add projects without URLs to results with null verification
                verification_results.append(
                    {"project": project, "verification": NullVerification()}
                )
                print(f"No URL provided for project: {project['name']}")

        return self.create_verification_report(verification_results)

    def create_verification_report(self, results: List[Dict]) -> Dict:
        """Create a structured report of verification results."""
        total_projects = len(results)  # Count all projects
        projects_with_urls = sum(1 for r in results if r["project"].get("url"))
        verified_projects = sum(
            1 for r in results if r["verification"].similarity_score >= 0.7
        )

        return {
            "summary": {
                "total_projects": total_projects,
                "projects_with_urls": projects_with_urls,
                "verified_projects": verified_projects,
                "verification_rate": (
                    f"{(verified_projects/projects_with_urls*100):.1f}%"
                    if projects_with_urls
                    else "0%"
                ),
            },
            "projects": [
                {
                    "name": r["project"]["name"],
                    "url": r["project"].get("url"),
                    "verification_score": round(
                        r["verification"].similarity_score * 10, 2
                    ),
                    "status": (
                        "Verified"
                        if r["verification"].similarity_score >= 0.7
                        else "No URL" if not r["project"].get("url") else "Needs Review"
                    ),
                    "details": r["verification"].create_structured_output(),
                }
                for r in results
            ],
        }


async def main():
    openai_api_key = "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"
    github_token = "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"
    system = ProjectVerificationSystem(
        openai_key=openai_api_key, github_token=github_token
    )

    # Example usage with both URL and local file
    # pdf_url = "https://adinarayanb.github.io/resume.pdf"
    local_pdf = "/Users/hetanthakkar/Downloads/CVJ.pdf"

    # # Verify projects from URL
    # results_url = await system.verify_all_projects(pdf_url)
    # print("Results from URL:")
    # print(json.dumps(results_url, indent=2))

    # Verify projects from local file
    results_local = await system.verify_all_projects(local_pdf)
    print("\nResults from local file:")
    print(json.dumps(results_local, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
