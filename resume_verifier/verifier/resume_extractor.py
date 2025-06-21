# Add imports at the top
from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile

import json
import google.generativeai as genai
import re
from typing import Dict, List, Tuple
import asyncio
import PyPDF2
import requests
from .projectverifier import ProjectVerifier
from django.core.files.uploadedfile import InMemoryUploadedFile
from urllib.parse import urlparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


import aiohttp


class PDFProcessor(multiprocessing.Process):
    def __init__(self, pdf_path, result_queue):
        super().__init__()
        self.pdf_path = pdf_path
        self.result_queue = result_queue

    def run(self):
        try:
            text, links = self._extract_from_pdf()
            self.result_queue.put((text, links))
        except Exception as e:
            self.result_queue.put((str(e), None))

    def _extract_from_pdf(self):
        if hasattr(self.pdf_path, "read"):
            # Handle InMemoryUploadedFile
            content = self.pdf_path.read()
            reader = PyPDF2.PdfReader(BytesIO(content))
        else:
            # Handle regular file path
            with open(self.pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)

        text = ""
        links = []
        current_pos = 0

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if "/Annots" in page:
                for annot in page["/Annots"]:
                    obj = annot.get_object()
                    if obj.get("/Subtype") == "/Link" and "/A" in obj:
                        if "/URI" in obj["/A"]:
                            url = obj["/A"]["/URI"]
                            context_start = max(0, current_pos - 50)
                            context_end = min(
                                len(text) + len(page_text),
                                current_pos + len(page_text) + 50,
                            )
                            context = (text + page_text)[context_start:context_end]
                            links.append(
                                {
                                    "url": url,
                                    "position": current_pos,
                                    "context": context,
                                    "page": page_num + 1,
                                }
                            )
            text += page_text
            current_pos += len(page_text)

        return text, links


class ResumeProjectExtractor:
    def __init__(self, gemini_key: str):
        self.api_key = gemini_key
        genai.configure(api_key=gemini_key)
        self.chunk_size = 2000

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        result_queue = multiprocessing.Queue()
        processor = PDFProcessor(pdf_path, result_queue)
        processor.start()
        processor.join()

        text, links = result_queue.get()
        if links is None:
            raise Exception(f"Error extracting PDF text: {text}")
        return text, links

    async def _extract_initial_projects(self, text: str) -> List[Dict]:
        """Extract initial project data including descriptions"""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""Extract projects from the resume text. DO NOT extract Experience entry as projects. For each project include:
            - name: Project name
            - technologies: List of technologies used
            - description: Full project description from the resume
            Format as JSON array. Maintain original descriptions from resume.
            
            Text: {text}"""

            response = model.generate_content(prompt)
            content = response.text
            try:
                return json.loads(
                    content[content.find("[") : content.rfind("]") + 1]
                )
            except:
                return []
        except Exception as e:
            print(f"Error in _extract_initial_projects: {e}")
            return []

    async def _match_single_project(self, project: Dict, links: List[Dict]) -> Dict:
        try:
            project_keywords = set(
                [
                    *project["name"].lower().split(),
                    *[tech.lower() for tech in project.get("technologies", [])],
                    *project.get("description", "")
                    .lower()
                    .split()[:10],  # Include first 10 words of description
                ]
            )

            relevant_links = []
            for link in links:
                link_text = (link["url"] + " " + link["context"]).lower()
                keyword_matches = sum(
                    1 for keyword in project_keywords if keyword in link_text
                )
                if keyword_matches >= 2:
                    relevant_links.append(
                        {
                            "url": link["url"],
                            "context": link["context"][:50],
                            "page": link["page"],
                        }
                    )

            if not relevant_links:
                project["url"] = None
                project["confidence"] = 0
                return project

            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""Find best matching link for project. Return single JSON object with url and confidence (0-1). Focus on Accuracy.

Project: {json.dumps({"name": project["name"], "technologies": project.get("technologies", []), "description": project.get("description", "")})}

Links: {json.dumps(relevant_links)}"""

            response = model.generate_content(prompt)
            content = response.text
            try:
                match = json.loads(
                    content[content.find("{") : content.rfind("}") + 1]
                )
                project["url"] = match.get("url")
                project["confidence"] = match.get("confidence", 0)
            except:
                project["url"] = None
                project["confidence"] = 0

            return project
        except Exception as e:
            print(f"Error in _match_single_project: {e}")
            project["url"] = None
            project["confidence"] = 0
            return project

    async def extract_projects(
        self, text: str, links_with_context: List[Dict]
    ) -> List[Dict]:
        """Extract projects using async processing and batching"""
        projects = await self._extract_initial_projects(text)
        if not projects:
            return []

        batch_size = 5
        matched_projects = []
        for i in range(0, len(projects), batch_size):
            batch = projects[i : i + batch_size]
            tasks = [
                self._match_single_project(project, links_with_context)
                for project in batch
            ]
            results = await asyncio.gather(*tasks)
            matched_projects.extend(results)

        return matched_projects


class ProjectVerificationSystem:
    def __init__(self, gemini_key: str, github_token: str = None):
        self.extractor = ResumeProjectExtractor(gemini_key)
        self.verifier = ProjectVerifier(gemini_key, github_token)

    async def verify_all_projects(self, pdf_path: str) -> Dict:
        try:
            resume_text, links = self.extractor.extract_text_from_pdf(pdf_path)
            if not resume_text:
                return {"error": "Failed to extract text from PDF"}

            print(f"Found {len(links)} links in PDF")

            projects = await self.extractor.extract_projects(resume_text, links)
            if not isinstance(projects, list):
                return {"error": "Failed to extract projects"}

            print(f"Extracted projects: {json.dumps(projects, indent=2)}")

            class NullVerification:
                similarity_score = 0

                def create_structured_output(self):
                    return {
                        "repository_exists": False,
                        "description_match": "No URL provided",
                        "verification_notes": "Project URL not available for verification",
                    }

            verification_results = []
            for project in projects:
                if project.get("url"):
                    result = await self.verifier.verify_project(
                        project["url"], project.get("description", "")
                    )
                    verification_results.append(
                        {"project": project, "verification": result}
                    )
                else:
                    verification_results.append(
                        {"project": project, "verification": NullVerification()}
                    )
                    print(f"No URL provided for project: {project['name']}")

            return self.create_verification_report(verification_results)

        except Exception as e:
            return {"error": str(e)}

    def create_verification_report(self, results: List[Dict]) -> Dict:
        total_projects = len(results)
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
                    "description": r["project"].get("description", ""),
                    "technologies": r["project"].get("technologies", []),
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
    # Example usage
    gemini_api_key = "YOUR_GEMINI_API_KEY_HERE"
    github_token = "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"
    system = ProjectVerificationSystem(
        gemini_key=gemini_api_key, github_token=github_token
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
