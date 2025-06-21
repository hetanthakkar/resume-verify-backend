import aiohttp
import json
import google.generativeai as genai


class JobMatcher:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)

    async def match_job_requirements(
        self, resume_text: str, job_requirements: dict
    ) -> dict:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
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

            response = model.generate_content(prompt)
            content = response.text

            # Extract JSON from response
            try:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    return json.loads(content[json_start:json_end])
                return {}
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse JSON response: {e}")
        except Exception as e:
            raise Exception(f"Gemini API request failed: {str(e)}")
