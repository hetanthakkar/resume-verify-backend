from typing import Dict, Optional
import aiohttp
import logging
from urllib.parse import urlparse
import re
from dataclasses import dataclass
from enum import Enum
import tiktoken
from functools import lru_cache
from .Scrapping import Scrapper
from .ContentProcessor import ContentProcessor


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

    def create_structured_output(self) -> Dict[str, Dict]:
        project_summary = {
            "type": self.url_type.value,
            "match_score": round(self.similarity_score * 10, 2),
            "status": "Verified" if self.similarity_score >= 0.7 else "Needs Review",
        }

        project_detailed = {
            "type": self.url_type.value,
            "match_score": round(self.similarity_score * 10, 2),
            "content_preview": (
                self.content_matched[:500] + "..."
                if len(self.content_matched) > 500
                else self.content_matched
            ),
            "error": self.error,
        }

        if self.repository_stats:
            project_detailed["repository_statistics"] = self.repository_stats
            project_summary["repository_activity"] = (
                "Active" if self.repository_stats.get("commits", 0) > 0 else "Inactive"
            )

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
        openai_api_key: str,
        github_token: Optional[str] = None,
        max_retries: int = 3,
        delay: float = 2.0,
        headless: bool = True,
    ):
        self.openai_api_key = openai_api_key
        self.github_token = github_token
        self.max_retries = max_retries
        self.delay = delay
        self.headless = headless
        self.session = None
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.setup_logging()
        self.processor = ContentProcessor()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "x-api-key": "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",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
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
    ) -> float:
        try:
            web_content = self.truncate_content(web_content)

            prompt = self._generate_prompt(
                resume_description, web_content, url_type, repo_data
            )

            request_payload = {
                "model": "claude-3-5-haiku-latest",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }

            async with self.session.post(
                "https://api.anthropic.com/v1/messages",
                json=request_payload,
            ) as response:
                if response.status != 200:
                    raise Exception(f"Claude API error: {await response.text()}")

                result = await response.json()
                content = result["content"][0]["text"]
                match = re.search(r"\d+(\.\d+)?", content)
                return float(match.group()) if match else 0.0

        except Exception as e:
            self.logger.error(f"Error in similarity check: {str(e)}")
            return 0.0

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

                Return only a similarity score 0-1:
                0 = No technology/implementation match
                0.5 = Some technology overlap but different implementation
                1 = Strong technology and implementation match"""

        return f"""Compare the following project description with the content and determine authenticity.
            Description: {resume_description}
            Content: {web_content}

            Return only a similarity score between 0 and 1:
            0 = No match or likely false claim
            0.3 = Some general similarities but lacks specific details
            0.7 = Strong match with specific technical details
            1 = Perfect match with verifiable implementation details"""

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
            )

        try:
            scraper = Scrapper()
            content = ""
            repo_data = None
            app_info = None

            if self.session is None:
                self.session = aiohttp.ClientSession(
                    headers={
                        "x-api-key": "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",
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                    }
                )

            # Fetch content based on URL type
            if url_type == UrlType.GITHUB:
                repo_data = await scraper.fetch_github_content(url)
                content = repo_data["content"]

                similarity_score = await self.check_similarity(
                    resume_description, content, url_type, repo_data
                )
                return ProjectVerificationResult(
                    similarity_score=similarity_score,
                    content_matched=content[:1000],
                    url_type=url_type,
                    repository_stats={"languages": repo_data["languages"]},
                )

            # Handle app stores and general web content
            result = await self._fetch_content(url_type, url, scraper)
            content, app_info = result["content"], result.get("app_info")

            print(content, "content is here")
            if not content:
                return ProjectVerificationResult(
                    similarity_score=0.0,
                    content_matched="",
                    url_type=url_type,
                    error="No content found",
                    app_info=app_info,
                )

            similarity_score = await self.check_similarity(
                resume_description, content, url_type, repo_data
            )

            return ProjectVerificationResult(
                similarity_score=similarity_score,
                content_matched=content[:1000],
                url_type=url_type,
                app_info=app_info,
            )

        except Exception as e:
            self.logger.error(f"Error verifying project: {str(e)}")
            return ProjectVerificationResult(
                similarity_score=0.0,
                content_matched="",
                url_type=url_type,
                error=str(e),
                app_info=app_info,
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
