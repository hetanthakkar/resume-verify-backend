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


class Scrapper:

    async def fetch_appstore_content(self, url: str) -> Dict:
        """Fetch content from Apple App Store using improved web scraping."""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
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
                    print(f"Error extracting specific content: {str(e)}")

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
            print(f"Error fetching App Store content: {str(e)}")
            return {"content": "", "app_info": None}

    async def fetch_playstore_content(self, url: str) -> Dict:
        """Fetch content from Google Play Store."""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
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
            print(f"Error fetching Play Store content: {str(e)}")
            return {"content": "", "app_info": None}

    async def fetch_web_content(self, url: str) -> str:
        """Fetch content from general web URLs using Playwright with proper error handling."""
        if not url:
            print("URL is empty or None")
            return ""

        try:
            # Import playwright explicitly to check installation
            try:
                from playwright.async_api import async_playwright
            except ImportError:
                print("Playwright not installed. Please run: pip install playwright")
                return ""

            async with async_playwright() as playwright:  # Use async context manager
                # Launch browser with modern configuration
                browser = await playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--enable-javascript",
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--window-size=1920,1080",
                    ],
                )

                if not browser:
                    print("Failed to launch browser")
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
                    print("Failed to create browser context")
                    await browser.close()
                    return ""

                # Create new page
                page = await context.new_page()
                if not page:
                    print("Failed to create new page")
                    await context.close()
                    await browser.close()
                    return ""

                try:
                    # Navigate to the page
                    response = await page.goto(
                        url, wait_until="networkidle", timeout=30000
                    )

                    if not response:
                        print("No response received from the page")
                        return ""

                    if response.status >= 400:
                        print(f"HTTP error status: {response.status}")
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
                        print(f"Error extracting content: {str(e)}")
                        return ""

                finally:
                    await page.close()
                    await context.close()
                    await browser.close()

        except Exception as e:
            print(f"Error in fetch_web_content: {str(e)}")
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

            headers["Authorization"] = f"token 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"

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
            print(f"Error fetching GitHub content: {str(e)}")
            return {"content": "", "languages": {}, "tech_stack": ""}
