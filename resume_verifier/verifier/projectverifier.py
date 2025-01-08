import asyncio
from typing import Dict, List, Optional, Tuple, Union
import aiohttp
import base64
import json
import logging
from urllib.parse import urlparse
from playwright.async_api import async_playwright
import re
from dataclasses import dataclass
from enum import Enum
import tiktoken
import time


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
        """Create structured output with summary and detailed views."""

        # Create basic project summary
        project_summary = {
            "type": self.url_type.value,
            "match_score": round(self.similarity_score * 10, 2),
            "status": "Verified" if self.similarity_score >= 0.7 else "Needs Review",
        }

        # Create detailed project information
        project_detailed = {
            "type": self.url_type.value,
            "match_score": round(self.similarity_score * 10, 2),
            "content_preview": (
                self.content_matched[:500] + "..."
                if len(self.content_matched) > 500
                else self.content_matched
            ),
            "error": self.error if self.error else None,
        }

        # Add repository stats if available
        if self.repository_stats:
            project_detailed["repository_statistics"] = self.repository_stats
            project_summary["repository_activity"] = (
                "Active" if self.repository_stats.get("commits", 0) > 0 else "Inactive"
            )

        # Add app information if available
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
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    async def check_similarity(
        self,
        resume_description: str,
        web_content: str,
        url_type: UrlType,
        repo_data: Optional[Dict] = None,
    ) -> float:
        """Check similarity between resume description and web content using Claude."""
        try:
            web_content = self.truncate_content(web_content)

            if url_type == UrlType.GITHUB and repo_data:
                tech_stack = repo_data.get("tech_stack", "")
                prompt = f"""Compare the following project description with the GitHub repository details.
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
            else:
                prompt = f"""Compare the following project description with the content and determine authenticity.
                    Description: {resume_description}
                    Content: {web_content}

                    Return only a similarity score between 0 and 1:
                    0 = No match or likely false claim
                    0.3 = Some general similarities but lacks specific details
                    0.7 = Strong match with specific technical details
                    1 = Perfect match with verifiable implementation details"""

            request_payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }
            start_time = time.time()  # Record the start time
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": "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",
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json=request_payload,  # Changed from data to json
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Claude API error: {await response.text()}")
                    elapsed_time = (
                        time.time() - start_time
                    )  # Calculate the elapsed time
                    print(
                        f"API call elapsed time: {elapsed_time:.2f} seconds"
                    )  # Print the elapsed time
                    result = await response.json()
                    # Updated to match current API response structure
                    content = result["content"][0]["text"]
                    match = re.search(r"\d+(\.\d+)?", content)
                    return float(match.group()) if match else 0.0

        except Exception as e:
            self.logger.error(f"Error in similarity check: {str(e)}")
            return 0.0

    def truncate_content(self, content: str, max_tokens: int = 6000) -> str:
        """Truncate content to fit within token limit."""
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

    def num_tokens_from_string(self, string: str, model: str = "gpt-4") -> int:
        """Calculate the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(string))

    def determine_url_type(self, url: str) -> UrlType:
        """Determine URL type including app stores."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return UrlType.INVALID

            domain = parsed.netloc.lower()
            path = parsed.path.lower()

            if domain == "github.com":
                return UrlType.GITHUB
            elif "apps.apple.com" in domain or "itunes.apple.com" in domain:
                return UrlType.APPSTORE
            elif "play.google.com" in domain and "/store/apps" in path:
                return UrlType.PLAYSTORE
            return UrlType.GENERAL
        except Exception:
            return UrlType.INVALID

    async def fetch_appstore_content(self, url: str) -> Dict:
        """Fetch content from Apple App Store using improved web scraping."""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self.headless)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                )

                page = await context.new_page()
                await page.goto(url, wait_until="networkidle")

                # Wait for content to load
                await page.wait_for_selector("main", timeout=10000)

                # Define selectors matching the JS version
                selectors = {
                    "title": ".product-header__title",
                    "subtitle": ".product-header__subtitle",
                    "seller": ".product-header__identity a",
                    "stars_text": ".we-rating-count.star-rating__count",
                    "price": ".app-header__list__item--price",
                    "description": ".section__description .we-truncate",
                    "image": ".we-artwork__image",
                    "size": '.information-list__item__term:has-text("Size") + dd',
                    "category": '.information-list__item__term:has-text("Category") + dd a',
                    "age_rating": '.information-list__item__term:has-text("Age Rating") + dd',
                    "languages": '.information-list__item__term:has-text("Languages") + dd p',
                }

                app_info = {}

                try:
                    # Get basic information using evaluate
                    for key, selector in selectors.items():
                        if key == "image":
                            result = await page.evaluate(
                                f"""() => {{
                                const el = document.querySelector('{selector}');
                                return el ? el.getAttribute('src') : '';
                            }}"""
                            )
                        else:
                            result = await page.evaluate(
                                f"""() => {{
                                const el = document.querySelector('{selector}');
                                return el ? el.textContent.replace(/\\n\\n/g, '\\n').replace(/\\s+/g, ' ').trim() : '';
                            }}"""
                            )
                        app_info[key] = result

                    # Split stars and rating
                    if app_info.get("stars_text"):
                        stars_parts = app_info["stars_text"].split("•")
                        app_info["stars"] = (
                            stars_parts[0].strip() if len(stars_parts) > 0 else ""
                        )
                        app_info["rating"] = (
                            stars_parts[1].strip() if len(stars_parts) > 1 else ""
                        )
                        del app_info["stars_text"]

                    # Get reviews
                    reviews = await page.evaluate(
                        """() => {
                        const reviews = [];
                        document.querySelectorAll('.we-customer-review').forEach(review => {
                            reviews.push({
                                user: review.querySelector('.we-customer-review__user')?.textContent.trim() || '',
                                date: review.querySelector('.we-customer-review__date')?.textContent.trim() || '',
                                title: review.querySelector('.we-customer-review__title')?.textContent.trim() || '',
                                review: review.querySelector('.we-customer-review__body')?.textContent
                                    .replace(/\\n\\n/g, '\\n').replace(/\\s+/g, ' ').trim() || ''
                            });
                        });
                        return reviews;
                    }"""
                    )
                    app_info["reviews"] = reviews

                    # Get compatibility information
                    compatibility = await page.evaluate(
                        """() => {
                        return Array.from(document.querySelectorAll(
                            '.information-list__item.l-column.small-12.medium-6.large-4.small-valign-top dl.information-list__item__definition__item dt.information-list__item__definition__item__term'
                        )).map(el => el.textContent.trim());
                    }"""
                    )
                    app_info["compatibility"] = compatibility

                except Exception as e:
                    self.logger.error(f"Error extracting specific content: {str(e)}")

                await browser.close()

                # Combine all content for similarity checking
                content_parts = [
                    app_info.get("title", ""),
                    app_info.get("subtitle", ""),
                    app_info.get("description", ""),
                ]
                if app_info.get("reviews"):
                    reviews_text = "\n".join(
                        review["review"] for review in app_info["reviews"]
                    )
                    content_parts.append(reviews_text)

                full_content = "\n\n".join(filter(None, content_parts))

                return {"content": full_content, "app_info": app_info}

        except Exception as e:
            self.logger.error(f"Error fetching App Store content: {str(e)}")
            return {"content": "", "app_info": None}

    async def fetch_playstore_content(self, url: str) -> Dict:
        """Fetch content from Google Play Store."""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self.headless)
                page = await browser.new_page()

                await page.goto(url, wait_until="networkidle")

                # Extract app information
                app_info = {}

                # Get app name
                name_element = await page.query_selector('h1[itemprop="name"]')
                app_info["name"] = (
                    await name_element.inner_text() if name_element else ""
                )

                # Get description
                desc_element = await page.query_selector('[data-g-id="description"]')
                app_info["description"] = (
                    await desc_element.inner_text() if desc_element else ""
                )

                # Get developer info
                dev_element = await page.query_selector('a[href*="developer"]')
                app_info["developer"] = (
                    await dev_element.inner_text() if dev_element else ""
                )

                # Get ratings
                try:
                    rating_element = await page.query_selector('div[class*="BHMmbe"]')
                    app_info["ratings"] = (
                        await rating_element.inner_text() if rating_element else ""
                    )
                except:
                    app_info["ratings"] = ""

                await browser.close()

                return {
                    "content": f"{app_info['name']}\n{app_info['description']}",
                    "app_info": app_info,
                }

        except Exception as e:
            self.logger.error(f"Error fetching Play Store content: {str(e)}")
            return {"content": "", "app_info": None}

    async def fetch_web_content(self, url: str) -> str:
        """Fetch content from general web URLs using Playwright with proper error handling."""
        if not url:
            self.logger.error("URL is empty or None")
            return ""

        try:
            # Import playwright explicitly to check installation
            try:
                from playwright.async_api import async_playwright
            except ImportError:
                self.logger.error(
                    "Playwright not installed. Please run: pip install playwright"
                )
                return ""

            async with async_playwright() as playwright:  # Use async context manager
                # Launch browser with modern configuration
                browser = await playwright.chromium.launch(
                    headless=self.headless,
                    args=[
                        "--enable-javascript",
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--window-size=1920,1080",
                    ],
                )

                if not browser:
                    self.logger.error("Failed to launch browser")
                    return ""

                # Create context with modern browser settings
                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    extra_http_headers={
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                    },
                )

                if not context:
                    self.logger.error("Failed to create browser context")
                    await browser.close()
                    return ""

                # Create new page
                page = await context.new_page()
                if not page:
                    self.logger.error("Failed to create new page")
                    await context.close()
                    await browser.close()
                    return ""

                try:
                    # Navigate to the page
                    response = await page.goto(
                        url, wait_until="networkidle", timeout=30000
                    )

                    if not response:
                        self.logger.error("No response received from the page")
                        return ""

                    if response.status >= 400:
                        self.logger.error(f"HTTP error status: {response.status}")
                        return ""

                    # Wait for content and extract it
                    try:
                        await page.wait_for_selector("body", timeout=5000)
                        content = await page.evaluate(
                            """
                            () => {
                                const text = document.body.innerText;
                                return text ? text.trim() : '';
                            }
                        """
                        )
                        return content.strip()
                    except Exception as e:
                        self.logger.error(f"Error extracting content: {str(e)}")
                        return ""

                finally:
                    await page.close()
                    await context.close()
                    await browser.close()

        except Exception as e:
            self.logger.error(f"Error in fetch_web_content: {str(e)}")
            return ""

    async def fetch_github_content(self, url: str) -> Dict:
        """Fetch languages and README content from a GitHub repository."""
        try:
            # Remove .git from URL if present
            url = url.replace(".git", "")

            path_parts = urlparse(url).path.strip("/").split("/")
            if len(path_parts) < 2:
                raise ValueError("Invalid GitHub URL format")

            owner, repo = path_parts[0], path_parts[1]

            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Project-Verifier-Bot",
            }

            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"

            async with aiohttp.ClientSession() as session:
                # Fetch languages
                async with session.get(
                    f"https://api.github.com/repos/{owner}/{repo}/languages",
                    headers=headers,
                ) as response:
                    languages = await response.json() if response.status == 200 else {}

                # Fetch README content
                async with session.get(
                    f"https://api.github.com/repos/{owner}/{repo}/readme",
                    headers=headers,
                ) as response:
                    readme_content = ""
                    if response.status == 200:
                        readme_data = await response.json()
                        if readme_data.get("content"):
                            readme_content = base64.b64decode(
                                readme_data["content"]
                            ).decode("utf-8")

                return {
                    "content": readme_content,
                    "languages": languages,
                    "tech_stack": ", ".join(languages.keys()),
                }

        except Exception as e:
            self.logger.error(f"Error fetching GitHub content: {str(e)}")
            return {"content": "", "languages": {}, "tech_stack": ""}

    async def verify_project(
        self, url: str, resume_description: str
    ) -> ProjectVerificationResult:
        """Main method to verify project authenticity."""
        url_type = self.determine_url_type(url)

        if url_type == UrlType.INVALID:
            return ProjectVerificationResult(
                similarity_score=0.0,
                content_matched="",
                url_type=UrlType.INVALID,
                error="Invalid URL format",
            )

        try:
            content = ""
            repo_data = None
            app_info = None

            if url_type == UrlType.GITHUB:
                repo_data = await self.fetch_github_content(url)
                content = repo_data["content"]
                similarity_score = await self.check_similarity(
                    resume_description, content, url_type, repo_data
                )
                return ProjectVerificationResult(
                    similarity_score=similarity_score,
                    content_matched=content[:1000],
                    url_type=url_type,
                    repository_stats={"languages": repo_data["languages"]},
                    app_info=None,
                )
            elif url_type == UrlType.APPSTORE:
                app_data = await self.fetch_appstore_content(url)
                content = app_data["content"]
                app_info = app_data["app_info"]
            elif url_type == UrlType.PLAYSTORE:
                app_data = await self.fetch_playstore_content(url)
                content = app_data["content"]
                app_info = app_data["app_info"]
            else:
                content = await self.fetch_web_content(url)

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
                repository_stats=None,
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


def format_verification_output(
    results: List[Tuple[str, ProjectVerificationResult]]
) -> None:
    """Format and print the verification results in a structured manner."""
    print("\n=== PROJECT VERIFICATION RESULTS ===")

    for url, result in results:
        structured_output = result.create_structured_output()

        print("\n" + "=" * 50)
        print(f"URL: {url}")

        # Print Summary Section
        print("\nSUMMARY:")
        summary = structured_output["summary"]
        print(f"• Project Type: {summary['type'].title()}")
        print(f"• Match Score: {summary['match_score']}/10")
        print(f"• Status: {summary['status']}")

        if "app_name" in summary:
            print(f"• App Name: {summary['app_name']}")
            print(f"• Developer: {summary['developer']}")

        if "repository_activity" in summary:
            print(f"• Repository Status: {summary['repository_activity']}")

        # Print Detailed Section
        print("\nDETAILED INFORMATION:")
        detailed = structured_output["detailed"]

        print("\nMatch Details:")
        print(f"• Verification Score: {detailed['match_score']}/10")
        print(f"• Project Type: {detailed['type']}")

        if detailed.get("content_preview"):
            print("\nContent Preview:")
            print(f"{detailed['content_preview']}")

        if detailed.get("repository_statistics"):
            print("\nRepository Statistics:")
            for key, value in detailed["repository_statistics"].items():
                print(f"• {key.replace('_', ' ').title()}: {value}")

        if detailed.get("app_information"):
            print("\nApp Information:")
            app_info = detailed["app_information"]
            important_fields = [
                "name",
                "title",
                "developer",
                "seller",
                "description",
                "category",
                "ratings",
                "stars",
                "price",
            ]

            for field in important_fields:
                if field in app_info and app_info[field]:
                    print(f"• {field.replace('_', ' ').title()}: {app_info[field]}")

            if "reviews" in app_info and app_info["reviews"]:
                print("\nRecent Reviews:")
                for i, review in enumerate(app_info["reviews"][:3], 1):
                    print(f"\nReview {i}:")
                    print(f"User: {review.get('user', 'Anonymous')}")
                    print(f"Date: {review.get('date', 'N/A')}")
                    print(f"Content: {review.get('review', 'No content')}")

        if detailed.get("error"):
            print(f"\nErrors Encountered: {detailed['error']}")

        print("\n" + "=" * 50)


# Example usage
async def main():
    verifier = ProjectVerifier(
        openai_api_key="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",
        github_token="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",  # Optional
        headless=True,
    )

    urls = [
        "https://github.com/hetanthakkar/P2P-file-sharing-application.git",
    ]

    resume_description = "Distributed P2P-File Sync"

    results = []
    for url in urls:
        result = await verifier.verify_project(url, resume_description)
        results.append((url, result))

    # Format and print results using the new structured output
    format_verification_output(results)


if __name__ == "__main__":
    asyncio.run(main())
