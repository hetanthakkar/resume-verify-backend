import requests
import PyPDF2
import google.generativeai as genai
import re
import os
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
import json
from functools import lru_cache

from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
import numpy as np
from .linkedinExtractor import EnhancedLinkedInExtractor


class ProfileMatcher:
    def __init__(self, gemini_key: str, batch_size: int = 3):
        self.gemini_key = gemini_key
        genai.configure(api_key=gemini_key)
        self.batch_size = batch_size
        self._text_cache = {}

    def _get_cache_key(self, text: str, extraction_type: str = None) -> str:
        """Create a unique cache key from text content."""
        prefix = f"{extraction_type}_" if extraction_type else ""
        # Use string representation for hashing instead of the dict itself
        return f"{prefix}{hash(str(text))}"

    def extract_additional_sections(self, text: str) -> Dict:
        """Extract publications, achievements, and other sections from text with improved accuracy."""
        example = """
        {
            "publications": [
                {
                    "title": "Example Publication",
                    "year": "2023",
                    "authors": ["John Doe"],
                    "venue": "Example Conference",
                    "doi": "10.1234/example",
                    "abstract": "Brief abstract of the publication"
                }
            ],
            "achievements": [
                {
                    "title": "Achievement Title",
                    "year": "2023",
                    "description": "Brief description",
                    "category": "Award/Recognition/Project",
                    "issuer": "Issuing Organization",
                    "impact_metrics": ["metric1", "metric2"]
                }
            ],
            "certifications": [
                {
                    "name": "Certification Name",
                    "issuer": "Issuing Organization",
                    "date": "2023",
                    "id": "CERT123",
                    "expiry": "2024",
                    "skills": ["skill1", "skill2"]
                }
            ]
        }"""

        system_prompt = """You are a precise parser that extracts structured data. Only include information that is explicitly present in the text. Do not make assumptions or infer missing details."""

        user_prompt = f"""Extract publications, achievements, and certifications from the text into a structured format.
        Be very precise and detailed in extracting information. Look for:
        Publications:
        - Full titles with proper capitalization
        - Complete list of authors
        - Exact publication years
        - Conference/journal names
        - DOI if available
        - Brief abstract or summary if present
        
        Achievements:
        - Specific achievement titles
        - Exact dates/years
        - Detailed descriptions
        - Issuing organizations
        - Impact metrics or quantifiable results
        - Categories (Award/Honor/Project)
        
        Certifications:
        - Full certification names
        - Issuing organizations
        - Issue dates and expiry dates
        - Certification IDs
        - Associated skills or competencies
        
        Text: {text}
        
        Return as valid JSON exactly matching this structure:
        {example}
        
        Important: Only include entries that are clearly and unambiguously present in the text."""

        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = model.generate_content(full_prompt)
            content = response.text

            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)

        except Exception as e:
            print(f"Error in extract_additional_sections: {e}")
            return {"publications": [], "achievements": [], "certifications": []}

    def parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        try:
            formats = ["%Y-%m", "%B %Y", "%b %Y", "%m/%Y", "%Y"]
            for fmt in formats:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    async def _extract_sections_single(self, text: str, cache_key: str) -> Dict:
        """Extract additional sections from a single text chunk."""
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]

        try:
            response = await self._make_openai_call(self._get_sections_prompt(), text)
            result = json.loads(response)
            self._text_cache[cache_key] = result
            return result
        except Exception as e:
            print(f"Error in sections extraction: {e}")
            return {"publications": [], "achievements": [], "certifications": []}

    async def batch_process_text(
        self, texts: List[str], extraction_type: str
    ) -> List[Dict]:
        """Process multiple texts in parallel using batched API calls."""

        async def process_batch(batch: List[str]) -> List[Dict]:
            tasks = []
            for text in batch:
                # Create cache key from string representation
                cache_key = self._get_cache_key(text, extraction_type)

                if cache_key in self._text_cache:
                    tasks.append(self._text_cache[cache_key])
                else:
                    if extraction_type == "experience":
                        task = self._extract_experience_single(text, cache_key)
                    else:
                        task = self._extract_sections_single(text, cache_key)
                    tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Handle any exceptions in the results
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({"error": str(result)})
                else:
                    processed_results.append(result)
            return processed_results

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        all_results = []

        async with aiohttp.ClientSession() as session:
            for batch in batches:
                batch_results = await process_batch(batch)
                all_results.extend(batch_results)

        return all_results

    @staticmethod
    def _get_sections_prompt() -> str:
        """Return the prompt template for sections extraction."""
        return """Extract and return JSON with these keys: "publications", "achievements", "certifications". Each should be an array of objects with relevant fields."""

    async def _make_openai_call(self, prompt: str, text: str) -> str:
        """Make a Gemini API call with retry logic and error handling."""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                full_prompt = f"{prompt}\n\nText: {text}"
                response = model.generate_content(full_prompt)
                return response.text.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(retry_delay * (attempt + 1))

    def _compute_similarity_matrix(
        self, resume_exp: List[Dict], linkedin_exp: List[Dict]
    ) -> np.ndarray:
        """
        Compute similarity matrix between resume and LinkedIn experiences.
        Returns a numpy array of shape (len(resume_exp), len(linkedin_exp))
        containing similarity scores.
        """

        matrix = np.zeros((len(resume_exp), len(linkedin_exp)))

        for i, r_exp in enumerate(resume_exp):
            for j, l_exp in enumerate(linkedin_exp):
                score, _ = self.compare_experience_entries(r_exp, l_exp)
                matrix[i, j] = score

        return matrix

    def _compare_publications(
        self, resume_pubs: List[Dict], linkedin_pubs: List[Dict]
    ) -> Tuple[float, List[Dict]]:
        """
        Compare publications between resume and LinkedIn profiles.
        Returns a tuple of (average_score, list_of_matches).
        """
        matches = []
        total_score = 0

        if not resume_pubs or not linkedin_pubs:
            return 0.0, []

        for resume_pub in resume_pubs:
            best_match_score = 0
            best_match = None

            for linkedin_pub in linkedin_pubs:
                # Compare titles (50% weight)
                title_similarity = SequenceMatcher(
                    None,
                    str(resume_pub.get("title", "")).lower(),
                    str(linkedin_pub.get("title", "")).lower(),
                ).ratio()

                # Compare authors (30% weight)
                authors_similarity = SequenceMatcher(
                    None,
                    " ".join(str(a) for a in resume_pub.get("authors", [])).lower(),
                    " ".join(str(a) for a in linkedin_pub.get("authors", [])).lower(),
                ).ratio()

                # Compare year and venue (20% weight)
                year_match = resume_pub.get("year") == linkedin_pub.get("year")
                venue_similarity = SequenceMatcher(
                    None,
                    str(resume_pub.get("venue", "")).lower(),
                    str(linkedin_pub.get("venue", "")).lower(),
                ).ratio()

                # Calculate weighted score
                score = (
                    title_similarity * 0.5
                    + authors_similarity * 0.3
                    + (year_match * 0.1)
                    + (venue_similarity * 0.1)
                ) * 10

                if score > best_match_score:
                    best_match_score = score
                    best_match = {
                        "publication": linkedin_pub,
                        "score": score,
                        "metrics": {
                            "title_match": round(title_similarity * 100, 1),
                            "authors_match": round(authors_similarity * 100, 1),
                            "year_match": year_match * 100,
                            "venue_match": round(venue_similarity * 100, 1),
                        },
                    }

            if best_match:
                matches.append(best_match)
                total_score += best_match_score

        avg_score = total_score / len(resume_pubs) if resume_pubs else 0
        return avg_score, matches

    async def compare_profiles(self, resume_text: str, linkedin_text: str) -> Dict:
        """Compare profiles with improved error handling."""
        try:
            # Convert inputs to strings if they aren't already
            resume_text = str(resume_text)
            linkedin_text = str(linkedin_text)

            experience_task = self.batch_process_text(
                [resume_text, linkedin_text], "experience"
            )
            sections_task = self.batch_process_text(
                [resume_text, linkedin_text], "sections"
            )

            experience_results, sections_results = await asyncio.gather(
                experience_task, sections_task
            )

            # Handle potential errors in results
            if any(
                "error" in result for result in experience_results + sections_results
            ):
                errors = [
                    result["error"]
                    for result in experience_results + sections_results
                    if "error" in result
                ]
                raise Exception(f"Errors in processing: {', '.join(errors)}")

            resume_exp, linkedin_exp = experience_results
            resume_sections, linkedin_sections = sections_results

            # Run comparisons
            loop = asyncio.get_event_loop()
            experience_future = loop.run_in_executor(
                None, self._compare_experiences, resume_exp, linkedin_exp
            )
            publications_future = loop.run_in_executor(
                None,
                self._compare_publications,
                resume_sections.get("publications", []),
                linkedin_sections.get("publications", []),
            )
            achievements_future = loop.run_in_executor(
                None,
                self._compare_achievements,
                resume_sections.get("achievements", []),
                linkedin_sections.get("achievements", []),
            )

            experience_score, detailed_matches = await experience_future
            pub_score, pub_matches = await publications_future
            ach_score, ach_matches = await achievements_future

            return self._create_final_output(
                experience_score,
                detailed_matches,
                pub_score,
                pub_matches,
                ach_score,
                ach_matches,
            )

        except Exception as e:
            raise Exception(f"Profile comparison failed: {str(e)}")

    def _create_final_output(
        self,
        exp_score: float,
        exp_matches: List[Dict],
        pub_score: float,
        pub_matches: List[Dict],
        ach_score: float,
        ach_matches: List[Dict],
    ) -> Dict:
        """Create the final structured output."""
        return {
            "overall_scores": {
                "total_match_score": round((exp_score + pub_score + ach_score) / 3, 2),
                "experience_score": round(exp_score, 2),
                "publications_score": round(pub_score, 2),
                "achievements_score": round(ach_score, 2),
            },
            "detailed_matches": {
                "experience": exp_matches,
                "publications": pub_matches,
                "achievements": ach_matches,
            },
        }

    def _compare_experiences(
        self, resume_exp: List[Dict], linkedin_exp: List[Dict]
    ) -> Tuple[float, List[Dict]]:
        """Optimized experience comparison using vectorized operations where possible."""
        matches = []
        total_score = 0

        # Pre-compute all similarity scores
        similarity_matrix = self._compute_similarity_matrix(resume_exp, linkedin_exp)

        for i, r_exp in enumerate(resume_exp):
            best_match_idx = similarity_matrix[i].argmax()
            best_match_score = similarity_matrix[i][best_match_idx]

            if best_match_score > 0:
                matches.append(
                    {
                        "resume_entry": r_exp,
                        "linkedin_entry": linkedin_exp[best_match_idx],
                        "score": best_match_score,
                        "metrics": self._compute_detailed_metrics(
                            r_exp, linkedin_exp[best_match_idx]
                        ),
                    }
                )
                total_score += best_match_score

        avg_score = total_score / len(resume_exp) if resume_exp else 0
        return avg_score, matches

    async def _extract_experience_single(self, text: str) -> Dict:
        """Extract experience details from a single text chunk."""
        if text in self._text_cache:
            return self._text_cache[text]

        example = """[{"company_name": "Example Corp","job_title": "Software Engineer","start_date": "01/2020","end_date": "Present","location": "New York, NY","responsibilities": ["Led team of 5", "Developed API"]}]"""

        prompt = f"""Extract work experience details from the text into JSON format exactly like this: {example}"""

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)

        try:
            content = response.text.strip()
            # Find the first '[' and last ']' to extract just the JSON array
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                result = json.loads(json_str)
                self._text_cache[text] = result
                return result
            return []
        except Exception as e:
            print(f"Error parsing experience details: {e}")
            return []

    def compare_publications(
        self, resume_pubs: List[Dict], linkedin_pubs: List[Dict]
    ) -> Tuple[float, List[Dict]]:
        """Compare publications between resume and LinkedIn."""
        matches = []
        total_score = 0

        for resume_pub in resume_pubs:
            best_match_score = 0
            best_match = None

            for linkedin_pub in linkedin_pubs:
                title_similarity = SequenceMatcher(
                    None,
                    resume_pub.get("title", "").lower(),
                    linkedin_pub.get("title", "").lower(),
                ).ratio()

                authors_similarity = SequenceMatcher(
                    None,
                    " ".join(resume_pub.get("authors", [])).lower(),
                    " ".join(linkedin_pub.get("authors", [])).lower(),
                ).ratio()

                score = (title_similarity * 0.7 + authors_similarity * 0.3) * 10

                if score > best_match_score:
                    best_match_score = score
                    best_match = {"publication": linkedin_pub, "score": score}

            if best_match:
                matches.append(best_match)
                total_score += best_match_score

        avg_score = total_score / len(resume_pubs) if resume_pubs else 0
        return avg_score, matches

    def compare_achievements(
        self, resume_achievements: List[Dict], linkedin_achievements: List[Dict]
    ) -> Tuple[float, List[Dict]]:
        """Compare achievements between resume and LinkedIn."""
        matches = []
        total_score = 0

        for resume_ach in resume_achievements:
            best_match_score = 0
            best_match = None

            for linkedin_ach in linkedin_achievements:
                title_similarity = SequenceMatcher(
                    None,
                    str(resume_ach.get("title", "")).lower(),
                    str(linkedin_ach.get("title", "")).lower(),
                ).ratio()

                desc_similarity = SequenceMatcher(
                    None,
                    str(resume_ach.get("description", "")).lower(),
                    str(linkedin_ach.get("description", "")).lower(),
                ).ratio()

                score = (title_similarity * 0.6 + desc_similarity * 0.4) * 10

                if score > best_match_score:
                    best_match_score = score
                    best_match = {"achievement": linkedin_ach, "score": score}

            if best_match:
                matches.append(best_match)
                total_score += best_match_score

        avg_score = total_score / len(resume_achievements) if resume_achievements else 0
        return avg_score, matches

    def extract_experience_details(self, text: str) -> List[Dict]:
        """Extract structured experience information from text."""
        example = """[
            {
                "company_name": "Example Corp",
                "job_title": "Software Engineer",
                "start_date": "01/2020",
                "end_date": "Present",
                "location": "New York, NY",
                "responsibilities": ["Led team of 5", "Developed API", "Improved performance"]
            }
        ]"""

        prompt = f"""Extract work experience details from the following text into a structured format.
        Format the response EXACTLY as a JSON array of objects with the following fields:
        - "company_name"
        - "job_title"
        - "start_date"
        - "end_date"
        - "location"
        - "responsibilities" (as array of strings)

        Text: {text}

        Ensure the response is valid JSON. Format it exactly like this example:
        {example}"""

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)

        try:
            content = response.text.strip()
            # Find the first '[' and last ']' to extract just the JSON array
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                return json.loads(json_str)
            return []
        except Exception as e:
            print(f"Error parsing experience details: {e}")
            return []

    def compare_experience_entries(
        self, resume_exp: Dict, linkedin_exp: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Compare two experience entries and return a similarity score and detailed metrics."""
        metrics = {}

        # Company name comparison (30% weight)
        company_similarity = SequenceMatcher(
            None,
            str(resume_exp.get("company_name", "")).lower(),
            str(linkedin_exp.get("company_name", "")).lower(),
        ).ratio()
        metrics["company_match"] = company_similarity

        # Job title comparison (25% weight)
        title_similarity = SequenceMatcher(
            None,
            str(resume_exp.get("job_title", "")).lower(),
            str(linkedin_exp.get("job_title", "")).lower(),
        ).ratio()
        metrics["title_match"] = title_similarity

        # Date comparison (25% weight)
        date_score = 0
        if all(
            key in resume_exp and key in linkedin_exp
            for key in ["start_date", "end_date"]
        ):
            resume_start = self.parse_date(str(resume_exp["start_date"]))
            linkedin_start = self.parse_date(str(linkedin_exp["start_date"]))

            # Handle "present" case before parsing dates
            resume_end_date = str(resume_exp["end_date"])
            linkedin_end_date = str(linkedin_exp["end_date"])

            resume_end = (
                datetime.now()
                if resume_end_date.lower() == "present"
                else self.parse_date(resume_end_date)
            )
            linkedin_end = (
                datetime.now()
                if linkedin_end_date.lower() == "present"
                else self.parse_date(linkedin_end_date)
            )

            if all([resume_start, linkedin_start, resume_end, linkedin_end]):
                start_match = abs((resume_start - linkedin_start).days) <= 31
                end_match = abs((resume_end - linkedin_end).days) <= 31
                date_score = (start_match + end_match) / 2

        metrics["date_match"] = date_score

        # Location comparison (10% weight)
        location_similarity = SequenceMatcher(
            None,
            str(resume_exp.get("location", "")).lower(),
            str(linkedin_exp.get("location", "")).lower(),
        ).ratio()
        metrics["location_match"] = location_similarity

        # Responsibilities comparison (10% weight)
        resp_similarity = 0
        if "responsibilities" in resume_exp and "responsibilities" in linkedin_exp:
            resume_resp = " ".join(
                str(r) for r in resume_exp["responsibilities"]
            ).lower()
            linkedin_resp = " ".join(
                str(r) for r in linkedin_exp["responsibilities"]
            ).lower()
            resp_similarity = SequenceMatcher(None, resume_resp, linkedin_resp).ratio()
        metrics["responsibilities_match"] = resp_similarity

        # Calculate weighted score
        final_score = (
            company_similarity * 0.30
            + title_similarity * 0.25
            + date_score * 0.25
            + location_similarity * 0.10
            + resp_similarity * 0.10
        ) * 10

        return final_score, metrics

    def get_final_match_score(
        self, resume_experiences: List[Dict], linkedin_experiences: List[Dict]
    ) -> Tuple[float, List[Dict]]:
        """Calculate overall match score and provide detailed matching metrics."""
        detailed_matches = []
        total_score = 0

        for resume_exp in resume_experiences:
            best_match_score = 0
            best_match_metrics = {}
            matched_linkedin_exp = None

            for linkedin_exp in linkedin_experiences:
                score, metrics = self.compare_experience_entries(
                    resume_exp, linkedin_exp
                )
                if score > best_match_score:
                    best_match_score = score
                    best_match_metrics = metrics
                    matched_linkedin_exp = linkedin_exp

            if matched_linkedin_exp:
                detailed_matches.append(
                    {
                        "resume_entry": resume_exp,
                        "linkedin_entry": matched_linkedin_exp,
                        "score": best_match_score,
                        "metrics": best_match_metrics,
                    }
                )
                total_score += best_match_score

        if not resume_experiences:
            return 0, []

        average_score = total_score / len(resume_experiences)

        # Apply penalties
        if len(resume_experiences) != len(linkedin_experiences):
            penalty = abs(len(resume_experiences) - len(linkedin_experiences)) * 0.5
            average_score = max(0, average_score - penalty)

        return average_score, detailed_matches

    def create_structured_output(
        self,
        detailed_matches: List[Dict],
        pub_matches: List[Dict],
        ach_matches: List[Dict],
    ) -> Dict[str, Dict[str, any]]:
        """Create structured output with summary and detailed views."""

        # Process Experience
        experience_summary = []
        experience_detailed = []

        for match in detailed_matches:
            resume_exp = match["resume_entry"]
            linkedin_exp = match["linkedin_entry"]

            # Create summary (only most important info)
            summary_entry = {
                "company": resume_exp.get("company_name"),
                "title": resume_exp.get("job_title"),
                "duration": f"{resume_exp.get('start_date')} - {resume_exp.get('end_date')}",
                "match_score": round(match["score"], 2),
            }
            experience_summary.append(summary_entry)

            # Create detailed entry
            detailed_entry = {
                "resume_data": resume_exp,
                "linkedin_data": linkedin_exp,
                "match_metrics": {
                    "overall_score": round(match["score"], 2),
                    "detailed_scores": {
                        k: round(v * 100, 1) for k, v in match["metrics"].items()
                    },
                },
            }
            experience_detailed.append(detailed_entry)

        # Process Publications
        publications_summary = []
        publications_detailed = []

        for match in pub_matches:
            pub = match["publication"]

            # Create summary
            summary_entry = {
                "title": pub.get("title"),
                "year": pub.get("year"),
                "match_score": round(match["score"], 2),
            }
            publications_summary.append(summary_entry)

            # Create detailed entry
            publications_detailed.append(
                {"publication_data": pub, "match_score": round(match["score"], 2)}
            )

        # Process Achievements
        achievements_summary = []
        achievements_detailed = []

        for match in ach_matches:
            ach = match["achievement"]

            # Create summary
            summary_entry = {
                "title": ach.get("title"),
                "year": ach.get("year"),
                "match_score": round(match["score"], 2),
            }
            achievements_summary.append(summary_entry)

            # Create detailed entry
            achievements_detailed.append(
                {"achievement_data": ach, "match_score": round(match["score"], 2)}
            )

        return {
            "experience": {
                "summary": experience_summary,
                "detailed": experience_detailed,
            },
            "publications": {
                "summary": publications_summary,
                "detailed": publications_detailed,
            },
            "achievements": {
                "summary": achievements_summary,
                "detailed": achievements_detailed,
            },
        }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None

    def process_profiles(self, pdf_path: str, linkedin_data: Dict) -> Dict:
        """Main method to process and compare profiles."""
        try:
            # Extract text from PDF
            extractor = EnhancedLinkedInExtractor()
            print("pdf path", pdf_path)
            resume_text = extractor.extract_text_from_pdf(pdf_path)
            if not resume_text:
                raise ValueError("Could not extract text from PDF")

            # Extract experience details from both sources
            resume_experiences = self.extract_experience_details(resume_text)
            linkedin_experiences = self.extract_experience_details(str(linkedin_data))

            # Get overall experience match score
            overall_score, detailed_matches = self.get_final_match_score(
                resume_experiences, linkedin_experiences
            )

            # Extract and compare additional sections
            resume_sections = self.extract_additional_sections(resume_text)
            linkedin_sections = self.extract_additional_sections(str(linkedin_data))

            # Compare publications
            pub_score, pub_matches = self.compare_publications(
                resume_sections.get("publications", []),
                linkedin_sections.get("publications", []),
            )

            # Compare achievements
            ach_score, ach_matches = self.compare_achievements(
                resume_sections.get("achievements", []),
                linkedin_sections.get("achievements", []),
            )

            # Create structured output
            structured_results = self.create_structured_output(
                detailed_matches, pub_matches, ach_matches
            )

            # Add overall scores
            structured_results["overall_scores"] = {
                "total_match_score": round(overall_score, 2),
                "experience_score": round(overall_score, 2),
                "publications_score": round(pub_score, 2),
                "achievements_score": round(ach_score, 2),
            }

            return structured_results
            
        except Exception as e:
            print(f"Error in process_profiles: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return a default structure in case of error
            return {
                "experience": {
                    "summary": [],
                    "detailed": []
                },
                "publications": {
                    "summary": [],
                    "detailed": []
                },
                "achievements": {
                    "summary": [],
                    "detailed": []
                },
                "overall_scores": {
                    "total_match_score": 0,
                    "experience_score": 0,
                    "publications_score": 0,
                    "achievements_score": 0
                }
            }
