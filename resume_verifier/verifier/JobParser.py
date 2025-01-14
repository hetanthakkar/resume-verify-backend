import logging
import json
import aiohttp
import asyncio
from typing import Dict, Optional, List
from urllib.parse import urlparse, parse_qs
import re
from django.conf import settings
from asgiref.sync import sync_to_async
from .models import Job

logger = logging.getLogger(__name__)


class JobParser:
    def __init__(self, user=None):
        self.user = user
        self.api_hub_key = settings.API_HUB_KEY
        self.claude_api_key = settings.CLAUDE_API_KEY
        self.base_url = "https://gateway.getapihub.cloud/api/v2/jobs/details"
        self.claude_url = "https://api.anthropic.com/v1/messages"

    async def _extract_skills_from_description(self, description: str) -> Dict:
        try:
            messages = [
                {
                    "role": "user",
                    "content": f"""Please analyze this job description and extract the following information in JSON format:
                1. Required technical skills (as an array of strings)(only Coding technologies, no soft skills or subjects/topics)
                2. Preferred/optional technical skills (as an array of strings)(only coding technologies, no soft skills or subjects/topics)
                3. Years of experience required (as a number, extract only the largest number mentioned for general experience)
                4. Education requirements (as a string)

                Job Description:
                {description}

                Return only the JSON with these keys: required_skills, preferred_skills, years_of_experience, education
                If any field is not found, use null for that field.""",
                }
            ]

            headers = {
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": self.claude_api_key,
            }

            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 4096,
                "messages": messages,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.claude_url, headers=headers, json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Claude API request failed: {error_text}")
                        return {}

                    data = await response.json()
                    content = data.get("content", [{}])[0].get("text", "{}")

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
                            f"Failed to parse Claude response as JSON: {content}"
                        )
                        return {
                            "required_skills": [],
                            "preferred_skills": [],
                            "years_of_experience": None,
                            "education": None,
                        }

        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}")
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
                headers = {"x-api-key": self.api_hub_key}
                url = f"{self.base_url}?job_id={job_id}"

                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"API request failed with status {response.status}: {error_text}"
                        )

                    data = await response.json()
                    description = data.get("description", "")

                    # Extract skills and requirements using Claude
                    claude_analysis = await self._extract_skills_from_description(
                        description
                    )

                    # Merge the analysis with the original data
                    analysis = self._transform_response(data)
                    analysis.update(
                        {
                            "required_skills": claude_analysis.get(
                                "required_skills", []
                            ),
                            "preferred_skills": claude_analysis.get(
                                "preferred_skills", []
                            ),
                            "years_of_experience": claude_analysis.get(
                                "years_of_experience"
                            ),
                            "education": claude_analysis.get("education"),
                        }
                    )
                    # print(analysis, "analysis")

                    # Use the async version of save_job_analysis
                    await self._save_job_analysis_async(job_id, input_data, analysis)

                    return {"success": True, "data": analysis, "error": None}

        except Exception as e:
            logger.error(f"Error analyzing job posting: {str(e)}")
            return {"success": False, "data": None, "error": str(e)}

    def _transform_response(self, api_response: Dict) -> Dict:
        company_info = api_response.get("company", {})

        return {
            "title": api_response.get("title"),
            "company": {
                "name": company_info.get("name"),
                "description": company_info.get("description"),
                "industry": company_info.get("industries", []),
                "size": company_info.get("staff_count"),
                "headquarters": company_info.get("headquarter", {}),
            },
            "location": api_response.get("location"),
            "employment_type": api_response.get("type"),
            "job_description": api_response.get("description"),
            "work_remote_allowed": api_response.get("work_remote_allowed"),
            "industries": api_response.get("formatted_industries", []),
            "job_functions": api_response.get("formatted_job_functions", []),
            "application_info": {
                "company_apply_url": api_response.get("apply_method", {}).get(
                    "company_apply_url"
                ),
                "easy_apply_url": api_response.get("apply_method", {}).get(
                    "easy_apply_url"
                ),
            },
            "metadata": {
                "views": api_response.get("views"),
                "listed_at": api_response.get("listed_at_date"),
                "expires_at": api_response.get("expireAt"),
                "state": api_response.get("state"),
                "closed": api_response.get("closed"),
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
