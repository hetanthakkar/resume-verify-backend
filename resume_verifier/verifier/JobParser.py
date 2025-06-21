import logging
import json
import aiohttp
import asyncio
from typing import Dict, Optional, List
from urllib.parse import urlparse, parse_qs
import re
from django.conf import settings
from asgiref.sync import sync_to_async
import google.generativeai as genai
from .models import Job

logger = logging.getLogger(__name__)


class JobParser:
    def __init__(self, user=None):
        self.user = user
        self.scrapingdog_api_key = "685677aff97326753bf58e05"
        self.gemini_api_key = settings.GEMINI_API_KEY
        genai.configure(api_key=self.gemini_api_key)
        self.base_url = "https://api.scrapingdog.com/linkedinjobs"

    async def _extract_skills_from_description(self, description: str) -> Dict:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""Please analyze this job description and extract the following information in JSON format:
                1. Required technical skills (as an array of strings)(only Coding technologies, no soft skills or subjects/topics)
                2. Preferred/optional technical skills (as an array of strings)(only coding technologies, no soft skills or subjects/topics)
                3. Years of experience required (as a number, extract only the largest number mentioned for general experience)
                4. Education requirements (as a string)

                Job Description:
                {description}

                Return only the JSON with these keys: required_skills, preferred_skills, years_of_experience, education
                If any field is not found, use null for that field."""

            response = model.generate_content(prompt)
            content = response.text

            if not content.strip().startswith("{"):
                # Extract JSON from the response if it's wrapped in text
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)

            try:
                parsed = json.loads(content)
                # Ensure empty lists instead of null for skills
                parsed["required_skills"] = parsed.get("required_skills") or []
                parsed["preferred_skills"] = (
                    parsed.get("preferred_skills") or []
                )
                return parsed
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse Gemini response as JSON: {content}"
                )
                return {
                    "required_skills": [],
                    "preferred_skills": [],
                    "years_of_experience": None,
                    "education": None,
                }

        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return {
                "required_skills": [],
                "preferred_skills": [],
                "years_of_experience": None,
                "education": None,
            }

    def _extract_job_id(self, url: str) -> Optional[str]:
        """
        Extract LinkedIn job ID from URL or return None if not found.
        Handles both /view/ and /jobs/view/ URL formats.
        """
        if not url:
            return None

        parsed_url = urlparse(url)

        # Handle URL parameters
        query_params = parse_qs(parsed_url.query)
        if "currentJobId" in query_params:
            return query_params["currentJobId"][0]

        # Handle path-based job IDs
        path_match = re.search(r"/(?:jobs/)?view/(\d+)", parsed_url.path)
        if path_match:
            return path_match.group(1)

        return None

    async def _analyze_single_job(self, input_data: str) -> Dict:
        try:
            job_id = (
                self._extract_job_id(input_data)
                if "linkedin.com" in input_data
                else input_data
            )

            if not job_id:
                raise ValueError("Could not extract job ID from URL")

            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}?api_key={self.scrapingdog_api_key}&job_id={job_id}"

                async with session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"API request failed with status {response.status}: {error_text}"
                        )

                    data = await response.json()
                    
                    # ScrapingDog returns an array, get the first item
                    if isinstance(data, list) and len(data) > 0:
                        job_data = data[0]
                    else:
                        raise Exception("No job data found in response")

                    description = job_data.get("job_description", "")

                    # Extract skills and requirements using Gemini
                    gemini_analysis = await self._extract_skills_from_description(
                        description
                    )

                    # Merge the analysis with the original data
                    analysis = self._transform_response(job_data)
                    analysis.update(
                        {
                            "required_skills": gemini_analysis.get(
                                "required_skills", []
                            ),
                            "preferred_skills": gemini_analysis.get(
                                "preferred_skills", []
                            ),
                            "years_of_experience": gemini_analysis.get(
                                "years_of_experience"
                            ),
                            "education": gemini_analysis.get("education"),
                        }
                    )

                    # Use the async version of save_job_analysis
                    await self._save_job_analysis_async(job_id, input_data, analysis)

                    return {"success": True, "data": analysis, "error": None}

        except Exception as e:
            logger.error(f"Error analyzing job posting: {str(e)}")
            return {"success": False, "data": None, "error": str(e)}

    def _transform_response(self, api_response: Dict) -> Dict:
        return {
            "title": api_response.get("job_position"),
            "company": {
                "name": api_response.get("company_name"),
                "description": "",
                "industry": [api_response.get("Industries", "")] if api_response.get("Industries") else [],
                "size": None,
                "headquarters": None,
            },
            "location": api_response.get("job_location"),
            "employment_type": api_response.get("Employment_type"),
            "job_description": api_response.get("job_description"),
            "work_remote_allowed": None,
            "industries": [api_response.get("Industries", "")] if api_response.get("Industries") else [],
            "job_functions": [api_response.get("Job_function", "")] if api_response.get("Job_function") else [],
            "application_info": {
                "company_apply_url": api_response.get("job_apply_link"),
                "easy_apply_url": None,
            },
            "metadata": {
                "views": None,
                "listed_at": api_response.get("job_posting_time"),
                "expires_at": None,
                "state": None,
                "closed": None,
            },
        }

    @sync_to_async
    def _save_job_analysis_async(
        self, job_id: str, input_data: str, analysis: Dict
    ) -> None:
        """Async wrapper for save_job_analysis"""
        return self.save_job_analysis(job_id, input_data, analysis)

    def save_job_analysis(self, job_id: str, input_data: str, analysis: Dict) -> Job:
        """Synchronous method to save job analysis"""
        # Extract and process skills lists
        required_skills = analysis.get("required_skills", [])
        if required_skills is None:
            required_skills = []

        preferred_skills = analysis.get("preferred_skills", [])
        if preferred_skills is None:
            preferred_skills = []

        # Process years of experience
        years_exp = analysis.get("years_of_experience")
        if isinstance(years_exp, str):
            try:
                years_exp = int(years_exp)
            except (ValueError, TypeError):
                years_exp = None
        print(preferred_skills, "preferred_skills")
        # Create the job with processed data
        job = {
            "title": analysis.get("title", ""),
            "company_name": analysis.get("company", {}).get("name", ""),
            "description": analysis.get("job_description", ""),
            "location": analysis.get("location", ""),
            "employment_type": analysis.get("employment_type", ""),
            "source_url": input_data,
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "years_of_experience": years_exp,
            "education": analysis.get("education"),
            "created_by": self.user,
        }

        return job

    async def analyze_jobs(
        self, urls: List[str] = None, job_ids: List[str] = None
    ) -> List[Dict]:
        inputs = urls or job_ids or []
        tasks = [self._analyze_single_job(input_data) for input_data in inputs]

        return await asyncio.gather(*tasks)
