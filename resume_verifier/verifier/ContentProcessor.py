import spacy
from collections import defaultdict
import re
from typing import Dict, List, Set, Union


class ContentProcessor:
    def __init__(self):
        # Load the English language model
        self.nlp = spacy.load("en_core_web_sm")

        # Keywords for different types of content
        self.keywords = {
            "mobile_tech": {
                "ios",
                "android",
                "swift",
                "kotlin",
                "react native",
                "flutter",
                "mobile",
                "app",
                "tablet",
                "smartphone",
                "device",
                "native",
            },
            "frameworks": {
                "react",
                "angular",
                "vue",
                "django",
                "flask",
                "express",
                "spring",
                "xamarin",
                "ionic",
                "cordova",
                "unity",
                "unreal",
            },
            "features": {
                "authentication",
                "notification",
                "payment",
                "analytics",
                "sync",
                "backup",
                "cloud",
                "storage",
                "streaming",
                "sharing",
                "social",
            },
            "technical": {
                "api",
                "sdk",
                "database",
                "cache",
                "server",
                "backend",
                "frontend",
                "ui",
                "ux",
                "interface",
                "architecture",
                "protocol",
                "algorithm",
            },
        }

        # Common verbs for technical descriptions
        self.tech_verbs = {
            "implement",
            "develop",
            "build",
            "create",
            "design",
            "integrate",
            "optimize",
            "enhance",
            "support",
            "maintain",
            "update",
            "configure",
        }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r"http[s]?://\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove multiple spaces and newlines
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n+", "\n", text)

        # Remove common boilerplate text
        boilerplate = [
            "Terms of Service",
            "Privacy Policy",
            "All rights reserved",
            "Copyright",
            "Download on the App Store",
            "Get it on Google Play",
        ]
        for phrase in boilerplate:
            text = re.sub(rf"{phrase}.*?\n", "", text, flags=re.IGNORECASE)

        return text.strip()

    def _extract_technical_info(self, text: str) -> Dict[str, Set[str]]:
        """Extract technical information from text."""
        doc = self.nlp(text.lower())

        found_info = defaultdict(set)

        # Extract keywords by category
        for category, terms in self.keywords.items():
            found_info[category] = {term for term in terms if term in text.lower()}

        # Extract technical verbs
        found_info["verbs"] = {
            token.lemma_
            for token in doc
            if token.pos_ == "VERB" and token.lemma_ in self.tech_verbs
        }

        return found_info

    def _process_reviews(self, reviews: List[Dict]) -> str:
        """Process and summarize reviews focusing on technical aspects."""
        if not reviews:
            return ""

        technical_reviews = []

        for review in reviews:
            review_text = review.get("review", "")
            if not review_text:
                continue

            # Check if review contains technical terms
            technical_content = False
            for category_terms in self.keywords.values():
                if any(term in review_text.lower() for term in category_terms):
                    technical_content = True
                    break

            if technical_content:
                technical_reviews.append(review_text)

        # Return concatenated technical reviews
        return "\n".join(technical_reviews)

    def process_appstore_content(self, content: Dict) -> Dict:
        """Process App Store specific content."""
        processed = {}

        if "app_info" in content:
            app_info = content["app_info"]

            # Process description
            if "description" in app_info:
                cleaned_desc = self._clean_text(app_info["description"])
                tech_info = self._extract_technical_info(cleaned_desc)
                processed["technical_features"] = tech_info

            # Process reviews
            if "reviews" in app_info:
                tech_reviews = self._process_reviews(app_info["reviews"])
                if tech_reviews:
                    processed["technical_reviews"] = tech_reviews

            # Keep essential metadata
            for key in ["title", "seller", "category"]:
                if key in app_info:
                    processed[key] = app_info[key]

        return processed

    def process_playstore_content(self, content: Dict) -> Dict:
        """Process Play Store specific content."""
        processed = {}

        if "app_info" in content:
            app_info = content["app_info"]

            # Process description
            if "description" in app_info:
                cleaned_desc = self._clean_text(app_info["description"])
                tech_info = self._extract_technical_info(cleaned_desc)
                processed["technical_features"] = tech_info

            # Keep essential metadata
            for key in ["name", "developer"]:
                if key in app_info:
                    processed[key] = app_info[key]

        return processed

    def process_github_content(self, content: Dict) -> Dict:
        """Process GitHub specific content."""
        processed = {}

        # Process README content
        if "content" in content:
            cleaned_content = self._clean_text(content["content"])
            tech_info = self._extract_technical_info(cleaned_content)
            processed["technical_features"] = tech_info

        # Keep language information
        if "languages" in content:
            processed["languages"] = content["languages"]
        if "tech_stack" in content:
            processed["tech_stack"] = content["tech_stack"]

        return processed

    def process_web_content(self, content: str) -> Dict:
        """Process general web content."""
        if not content:
            return {}

        cleaned_content = self._clean_text(content)
        tech_info = self._extract_technical_info(cleaned_content)

        return {"technical_features": tech_info, "processed_content": cleaned_content}

    def process_content(self, content: Union[Dict, str], source_type: str) -> Dict:
        """Main method to process content based on source type."""
        if not content:
            return {}

        processors = {
            "appstore": self.process_appstore_content,
            "playstore": self.process_playstore_content,
            "github": self.process_github_content,
            "web": self.process_web_content,
        }

        processor = processors.get(source_type.lower())
        if not processor:
            raise ValueError(f"Unsupported source type: {source_type}")

        return processor(content)
