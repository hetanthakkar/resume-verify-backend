from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import JSONParser
import asyncio
from asgiref.sync import sync_to_async
import requests
import os
import tempfile
from .models import ProfileMatchResult, VerificationResult
from .serializers import ProfileMatchResultSerializer, VerificationResultSerializer
from .profile_matcher import ProfileMatcher
from .resume_extractor import ResumeProjectExtractor
from .projectverifier import ProjectVerifier
from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
import asyncio
from asgiref.sync import sync_to_async
import aiohttp


class ProjectVerificationView(APIView):
    def post(self, request):
        try:
            return asyncio.run(self._async_post(request))
        except Exception as e:
            return Response(
                {"error": f"Async operation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    async def _async_post(self, request):
        try:
            # Validate PDF file
            if "pdf_file" not in request.FILES:
                return Response(
                    {"error": "No PDF file provided"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            pdf_file = request.FILES["pdf_file"]
            openai_api_key = "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"

            try:
                # Initialize extractor and extract text
                extractor = ResumeProjectExtractor(openai_api_key)
                resume_text, links = extractor.extract_text_from_pdf(pdf_file)

                if not resume_text:
                    return Response(
                        {"error": "Failed to extract text from PDF"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                # Extract projects
                projects = await extractor.extract_projects(resume_text, links)

                # Verify projects
                async with ProjectVerifier(openai_api_key) as verifier:
                    verification_tasks = [
                        self.verify_single_project(project, verifier)
                        for project in projects
                    ]
                    verification_results = await asyncio.gather(*verification_tasks)

                # Calculate statistics
                total_projects = len(projects)
                projects_with_urls = sum(1 for p in projects if p.get("url"))
                verified_projects = sum(
                    1
                    for r in verification_results
                    if r["verification"]["summary"]["match_score"] >= 7.0
                )

                # Create final report
                final_report = {
                    "summary": {
                        "total_projects": total_projects,
                        "projects_with_urls": projects_with_urls,
                        "verified_projects": verified_projects,
                        "verification_rate": (
                            f"{(verified_projects/projects_with_urls*100):.1f}%"
                            if projects_with_urls
                            else "0%"
                        ),
                    },
                    "projects": [
                        {
                            "name": r["project"]["name"],
                            "url": r["project"].get("url"),
                            "verification_score": r["verification"]["summary"][
                                "match_score"
                            ],
                            "status": r["verification"]["summary"]["status"],
                            "type": r["verification"]["summary"]["type"],
                            "details": r["verification"]["detailed"],
                        }
                        for r in verification_results
                    ],
                }

                # Save results
                create_result = sync_to_async(VerificationResult.objects.create)
                verification_result = await create_result(
                    pdf_file=pdf_file, results=final_report
                )
                serializer = VerificationResultSerializer(verification_result)
                return Response(serializer.data, status=status.HTTP_200_OK)

            except Exception as e:
                return Response(
                    {"error": f"Processing error: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        except Exception as e:
            return Response(
                {"error": f"Request processing error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    async def verify_single_project(self, project, verifier):
        try:
            if not project.get("url"):
                return {
                    "project": project,
                    "verification": {
                        "summary": {
                            "type": "none",
                            "match_score": 0,
                            "status": "No URL",
                        },
                        "detailed": {
                            "type": "none",
                            "match_score": 0,
                            "content_preview": "No URL provided",
                            "error": "Project URL not available for verification",
                        },
                    },
                }

            result = await verifier.verify_project(
                project["url"], project["description"]
            )
            return {
                "project": project,
                "verification": result.create_structured_output(),
            }
        except Exception as e:
            return {
                "project": project,
                "verification": {
                    "summary": {
                        "type": "error",
                        "match_score": 0,
                        "status": "Verification Failed",
                    },
                    "detailed": {
                        "type": "error",
                        "match_score": 0,
                        "content_preview": str(e),
                        "error": "Failed to verify project",
                    },
                },
            }


class ProfileMatchView(APIView):
    def post(self, request):
        try:
            return asyncio.run(self._async_post(request))
        except Exception as e:
            return Response(
                {"error": f"Async operation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    async def _async_post(self, request):
        temp_pdf_path = None
        try:
            # Validate input
            pdf_url = request.data.get("pdf_url")
            linkedin_profile_url = request.data.get("linkedin_profile_url")
            auth_token = request.data.get("auth_token")

            if not all([pdf_url, linkedin_profile_url, auth_token]):
                return Response(
                    {
                        "error": "Missing required fields: pdf_url, linkedin_profile_url, or auth_token"
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                # Initialize matcher
                matcher = ProfileMatcher(
                    "sk-3ieS4zm3RDbTv9bKm7_PuvcRPjfaFhhoSJ6lJ6VdRmT3BlbkFJMCh_LyeVG1zGzvilyHhibyl3TfLVlhX8CBTlyu1VIA"
                )

                # Download PDF and fetch LinkedIn data concurrently
                download_task = self.download_pdf(pdf_url)
                linkedin_task = self.fetch_linkedin_data(
                    linkedin_profile_url, auth_token
                )

                temp_pdf_path, linkedin_data = await asyncio.gather(
                    download_task, linkedin_task
                )

                # Process profiles - this needs to run synchronously
                async def run_sync_compare():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, matcher.process_profiles, temp_pdf_path, linkedin_data
                    )

                results = await run_sync_compare()

                # Create database record - this needs to run synchronously
                @sync_to_async
                def create_db_record():
                    return ProfileMatchResult.objects.create(
                        pdf_url=pdf_url,
                        linkedin_url=linkedin_profile_url,
                        results=results,
                    )

                match_result = await create_db_record()

                # Serialize result - this can run synchronously
                @sync_to_async
                def serialize_result():
                    return ProfileMatchResultSerializer(match_result).data

                serialized_data = await serialize_result()
                return Response(serialized_data, status=status.HTTP_200_OK)

            except Exception as e:
                return Response(
                    {"error": f"Processing error: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        except Exception as e:
            return Response(
                {"error": f"Request processing error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.unlink(temp_pdf_path)
                except Exception as e:
                    print(f"Error removing temporary file: {e}")

    async def download_pdf(self, original_pdf_url: str) -> str:
        async def _download(pdf_url: str):
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as temp_pdf:
                    # Get the actual download URL before making the request
                    download_url = (
                        self.get_direct_download_url(pdf_url)
                        if "drive.google.com" in pdf_url
                        else pdf_url
                    )

                    async with aiohttp.ClientSession() as session:
                        async with session.get(download_url) as response:
                            if response.status != 200:
                                raise Exception(
                                    f"Failed to download PDF. Status: {response.status}"
                                )
                            content = await response.read()
                            if not content:
                                raise Exception("Downloaded PDF is empty")

                            temp_pdf.write(content)
                            return temp_pdf.name

            except aiohttp.ClientError as e:
                raise Exception(f"Network error while downloading PDF: {str(e)}")
            except Exception as e:
                raise Exception(f"Error downloading PDF: {str(e)}")

        if not original_pdf_url:
            raise Exception("PDF URL is required")

        try:
            return await _download(original_pdf_url)
        except Exception as e:
            # Re-raise with more context if needed
            raise type(e)(f"{str(e)} (URL: {original_pdf_url})")

    async def fetch_linkedin_data(
        self, linkedin_profile_url: str, auth_token: str
    ) -> dict:
        async def _fetch():
            try:
                headers = {
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json",
                }
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        linkedin_profile_url, headers=headers
                    ) as response:
                        response.raise_for_status()
                        return await response.json()
            except Exception as e:
                raise Exception(f"Error fetching LinkedIn profile: {str(e)}")

        return await _fetch()


def get_direct_download_url(self, gdrive_url: str) -> str:
    """Convert Google Drive sharing URL to direct download URL."""
    try:
        file_id = None
        # Handle different Google Drive URL formats
        if "drive.google.com/file/d/" in gdrive_url:
            file_id = gdrive_url.split("/file/d/")[1].split("/")[0]
        elif "drive.google.com/open?id=" in gdrive_url:
            file_id = gdrive_url.split("id=")[1].split("&")[0]
        elif "drive.google.com/uc?id=" in gdrive_url:
            file_id = gdrive_url.split("id=")[1].split("&")[0]

        if not file_id:
            raise ValueError("Could not extract file ID from Google Drive URL")

        return f"https://drive.google.com/uc?export=download&id={file_id}"
    except Exception as e:
        raise ValueError(f"Invalid Google Drive URL format: {str(e)}")


class CombinedVerificationView(APIView):
    def post(self, request):
        try:
            return asyncio.run(self._async_post(request))
        except Exception as e:
            return Response(
                {"error": f"Async operation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    async def _async_post(self, request):
        try:
            # Validate inputs for both operations
            pdf_file = request.FILES.get("pdf_file")
            pdf_url = request.data.get("pdf_url")
            linkedin_profile_url = request.data.get("linkedin_profile_url")
            auth_token = request.data.get("auth_token")

            if not pdf_file:
                return Response(
                    {"error": "No PDF file provided"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if not all([pdf_url, linkedin_profile_url, auth_token]):
                return Response(
                    {
                        "error": "Missing required fields: pdf_url, linkedin_profile_url, or auth_token"
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Create instances of both views
            project_view = ProjectVerificationView()
            profile_view = ProfileMatchView()

            # Create request objects for both views
            project_request = type(
                "TempRequest", (), {"FILES": request.FILES, "data": request.data}
            )

            profile_request = type(
                "TempRequest",
                (),
                {
                    "data": {
                        "pdf_url": pdf_url,
                        "linkedin_profile_url": linkedin_profile_url,
                        "auth_token": auth_token,
                    }
                },
            )

            # Run both verifications in parallel
            project_verification_task = project_view._async_post(project_request)
            profile_match_task = profile_view._async_post(profile_request)

            # Wait for both tasks to complete
            project_results, profile_results = await asyncio.gather(
                project_verification_task, profile_match_task, return_exceptions=True
            )

            # Process results
            combined_results = {
                "project_verification": (
                    project_results.data
                    if not isinstance(project_results, Exception)
                    else {"error": str(project_results)}
                ),
                "profile_match": (
                    profile_results.data
                    if not isinstance(profile_results, Exception)
                    else {"error": str(profile_results)}
                ),
            }

            return Response(combined_results, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": f"Combined verification failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
