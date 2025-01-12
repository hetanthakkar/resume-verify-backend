import aiohttp
import json


class JobMatcher:
    def __init__(self, api_key):
        self.api_key = api_key

    async def match_job_requirements(
        self, resume_text: str, job_requirements: dict
    ) -> dict:
        headers = {
            "anthropic-version": "2023-06-01",
            "x-api-key": self.api_key,
            "content-type": "application/json",
        }

        # Prepare the prompt for skills and requirements analysis
        prompt = f"""Based on the resume text and job requirements, analyze:
1. Which required skills are demonstrated through projects (not just listed skills)
2. Which preferred skills are demonstrated through projects
3. Does the candidate meet[greater than equal to] the required years of experience[only include professional experience]
4. Does the candidate meet the education requirements
Be strict about matching the requirements
Resume Text:
{resume_text}

Job Requirements:
Required Skills: {job_requirements['required_skills']}
Preferred Skills: {job_requirements['preferred_skills']}
Required Experience: {job_requirements['years_of_experience']}
Required Education: {job_requirements['education']}

Return response as JSON with this structure:
{{
    "required_skills_matched": [
        {{"skill": "skill_name", "project": "project_name", "description": "how it was used"}}
    ],
    "preferred_skills_matched": [
        {{"skill": "skill_name", "project": "project_name", "description": "how it was used"}}
    ],
    "experience_match": {{
        "meets_requirement": true/false,
        "years": "number",
        "justification": "explanation"
    }},
    "education_match": {{
        "meets_requirement": true/false,
        "justification": "explanation"
    }}
}}"""

        payload = {
            "model": "claude-3-5-sonnet-latest",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=payload
            ) as response:
                if response.status != 200:
                    error_body = await response.text()
                    raise Exception(
                        f"API request failed with status code: {response.status}. Response: {error_body}"
                    )

                result = await response.json()
                content = result["content"][0]["text"]

                # Extract JSON from response
                try:
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        return json.loads(content[json_start:json_end])
                    return {}
                except json.JSONDecodeError as e:
                    raise Exception(f"Failed to parse JSON response: {e}")
