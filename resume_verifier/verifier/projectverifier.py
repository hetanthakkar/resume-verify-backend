from typing import Dict, Optional
import aiohttp
import logging
from urllib.parse import urlparse
import re
from dataclasses import dataclass
from enum import Enum
import tiktoken
from functools import lru_cache
import google.generativeai as genai
from .Scrapping import Scrapper


class UrlType(Enum):
    GITHUB = "github"
    APPSTORE = "appstore"
    PLAYSTORE = "playstore"
    GENERAL = "general"
    INVALID = "invalid"


@dataclass
class ProjectVerificationResult:
    similarity_score: float
    content_matched: str
    url_type: UrlType
    error: Optional[str] = None
    repository_stats: Optional[Dict] = None
    app_info: Optional[Dict] = None
    match_justification: Optional[str] = None  # New field for justification

    def create_structured_output(self) -> Dict[str, Dict]:
        """Create structured output with summary and detailed views."""
        project_summary = {
            "type": self.url_type.value,
            "match_score": round(float(self.similarity_score) * 10, 2),
            "status": "Verified" if self.similarity_score >= 0.7 else "Needs Review",
            "match_justification": (
                str(self.match_justification)
                if self.match_justification
                else "No justification provided"
            ),
        }

        project_detailed = {
            "type": self.url_type.value,
            "match_score": round(float(self.similarity_score) * 10, 2),
            "content_preview": (
                str(self.content_matched)[:500] + "..."
                if len(str(self.content_matched)) > 500
                else str(self.content_matched)
            ),
            "error": self.error,
            "match_justification": (
                str(self.match_justification)
                if self.match_justification
                else "No justification provided"
            ),
        }

        if self.repository_stats:
            github_stats = {
                "languages": self.repository_stats.get("languages", {}),
                "total_commits": self.repository_stats.get("total_commits", 0),
                "stars": self.repository_stats.get("stars", 0),
                "forks": self.repository_stats.get("forks", 0),
            }
            project_detailed["repository_statistics"] = github_stats
            project_summary["repository_activity"] = (
                "Active" if github_stats["total_commits"] > 0 else "Inactive"
            )
            project_summary["github_highlights"] = {
                "primary_language": next(iter(github_stats["languages"].keys()), "N/A"),
                "total_commits": github_stats["total_commits"],
                "stars": github_stats["stars"],
            }

        if self.app_info:
            project_detailed["app_information"] = self.app_info
            if isinstance(self.app_info, dict):
                project_summary["app_name"] = self.app_info.get(
                    "name", ""
                ) or self.app_info.get("title", "")
                project_summary["developer"] = self.app_info.get(
                    "developer", ""
                ) or self.app_info.get("seller", "")

        return {"summary": project_summary, "detailed": project_detailed}


class ProjectVerifier:
    def __init__(
        self,
        gemini_api_key: str,
        github_token: Optional[str] = None,
        max_retries: int = 3,
        delay: float = 2.0,
        headless: bool = True,
    ):
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.github_token = github_token
        self.max_retries = max_retries
        self.delay = delay
        self.headless = headless
        self.session = None
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "Content-Type": "application/json",
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @lru_cache(maxsize=1000)
    def num_tokens_from_string(self, string: str) -> int:
        return len(self.encoding.encode(string))

    def truncate_content(self, content: str, max_tokens: int = 6000) -> str:
        if self.num_tokens_from_string(content) <= max_tokens:
            return content

        words = content.split()
        result = []
        current_tokens = 0

        for word in words:
            word_tokens = self.num_tokens_from_string(word)
            if current_tokens + word_tokens > max_tokens:
                break
            result.append(word)
            current_tokens += word_tokens

        return " ".join(result)

    @staticmethod
    def determine_url_type(url: str) -> UrlType:
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return UrlType.INVALID

            domain = parsed.netloc.lower()
            path = parsed.path.lower()

            if "github.com" in domain:
                return UrlType.GITHUB
            elif any(x in domain for x in ("apps.apple.com", "itunes.apple.com")):
                return UrlType.APPSTORE
            elif "play.google.com" in domain and "/store/apps" in path:
                return UrlType.PLAYSTORE
            return UrlType.GENERAL
        except Exception:
            return UrlType.INVALID

    async def check_similarity(
        self,
        resume_description: str,
        web_content: str,
        url_type: UrlType,
        repo_data: Optional[Dict] = None,
    ) -> tuple[float, str]:  # Modified to return both score and justification
        try:
            web_content = self.truncate_content(web_content)

            prompt = self._generate_prompt(
                resume_description, web_content, url_type, repo_data
            )

            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            content = response.text

            # Extract score and justification
            score_match = re.search(r"Score: (\d+(\.\d+)?)", content)
            justification_match = re.search(r"Justification: (.+)(?:\n|$)", content)

            score = float(score_match.group(1)) if score_match else 0.0
            justification = (
                justification_match.group(1)
                if justification_match
                else "No justification provided"
            )

            return score, justification

        except Exception as e:
            self.logger.error(f"Error in similarity check: {str(e)}")
            return 0.0, "Error occurred during similarity check"

    def _generate_prompt(
        self,
        resume_description: str,
        web_content: str,
        url_type: UrlType,
        repo_data: Optional[Dict],
    ) -> str:

        if url_type == UrlType.GITHUB and repo_data:
            tech_stack = repo_data.get("tech_stack", "")
            return f"""Compare the following project description with the GitHub repository details.
                Focus on matching technologies and implementation details.

                Resume Description: {resume_description}
                Repository Technologies: {tech_stack}
                README Content: {web_content}

                Consider:
                1. Do the technologies mentioned/used match the description?
                2. Does the README describe similar functionality?

                Return in this exact format:
                Score: [0-1]
                Justification: [One line explaining what matched/didn't match]

                Example:
                Score: 0.8
                Justification: Strong tech stack match (React/Node.js) but missing some described ML features"""

        return f"""Compare the following project description with the content and determine authenticity.
            Description: {resume_description}
            Content: {web_content}

            Return in this exact format:
            Score: [0-1]
            Justification: [One line explaining what matched/didn't match]

            Example:
            Score: 0.7
            Justification: Core functionality matches but deployment details differ"""

    async def verify_project(
        self, url: str, resume_description: str
    ) -> ProjectVerificationResult:
        url_type = self.determine_url_type(url)

        if url_type == UrlType.INVALID:
            return ProjectVerificationResult(
                similarity_score=0.0,
                content_matched="",
                url_type=UrlType.INVALID,
                error="Invalid URL format",
                match_justification="Invalid URL provided",
            )

        try:
            scraper = Scrapper()
            content = ""
            repo_data = None
            app_info = None

            if self.session is None:
                self.session = aiohttp.ClientSession(
                    headers={
                        "Content-Type": "application/json",
                    }
                )

            # Fetch content based on URL type
            if url_type == UrlType.GITHUB:
                repo_data = await scraper.fetch_github_content(url)
                content = repo_data["content"]

                similarity_score, justification = (
                    await self.check_similarity(  # Unpack both values
                        resume_description, content, url_type, repo_data
                    )
                )
                return ProjectVerificationResult(
                    similarity_score=similarity_score,
                    content_matched=content[:1000],
                    url_type=url_type,
                    repository_stats={"languages": repo_data["languages"]},
                    match_justification=justification,  # Pass the justification
                )

            # Handle app stores and general web content
            result = await self._fetch_content(url_type, url, scraper)
            content, app_info = result["content"], result.get("app_info")

            if not content:
                return ProjectVerificationResult(
                    similarity_score=0.0,
                    content_matched="",
                    url_type=url_type,
                    error="No content found",
                    app_info=app_info,
                    match_justification="No content could be extracted from URL",
                )

            similarity_score, justification = (
                await self.check_similarity(  # Unpack both values
                    resume_description, content, url_type, repo_data
                )
            )

            return ProjectVerificationResult(
                similarity_score=similarity_score,
                content_matched=content[:1000],
                url_type=url_type,
                app_info=app_info,
                match_justification=justification,  # Pass the justification
            )

        except Exception as e:
            self.logger.error(f"Error verifying project: {str(e)}")
            return ProjectVerificationResult(
                similarity_score=0.0,
                content_matched="",
                url_type=url_type,
                error=str(e),
                app_info=app_info,
                match_justification=f"Error occurred: {str(e)}",
            )

    async def _fetch_content(
        self, url_type: UrlType, url: str, scraper: Scrapper
    ) -> Dict:
        if url_type == UrlType.APPSTORE:
            content = await scraper.fetch_appstore_content(url)

            return content
        elif url_type == UrlType.PLAYSTORE:
            content = await scraper.fetch_playstore_content(url)

            return content

        else:
            content = await scraper.fetch_web_content(url)

            return {"content": content}
