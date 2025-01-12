# Python standard library
import asyncio
import io
import re

# Django imports
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.db import models, transaction
from django.http import Http404, HttpRequest
from django.shortcuts import get_object_or_404, render

# Third-party async/sync utilities
import aiohttp
from asgiref.sync import async_to_sync, sync_to_async

# Django REST Framework imports
from rest_framework import generics, permissions, status, views, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

# Local imports
from .JobMatcher import JobMatcher
from .JobParser import JobParser
from .linkedinExtractor import EnhancedLinkedInExtractor
from .models import (
    Candidate,
    Job,
    JobRecruiter,
    Resume,
    ResumeAnalysis,
    Shortlist,
)
from .profile_matcher import ProfileMatcher
from .projectverifier import ProjectVerifier
from .resume_extractor import ResumeProjectExtractor
from .serializers import (
    CandidateSerializer,
    JobSerializer,
    LoginSerializer,
    RecruiterSerializer,
    RegisterSerializer,
    ResumeVersionSerializer,
    ShortlistSerializer,
)


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

            return Response(final_report, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": f"Processing error: {str(e)}"},
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
        try:
            # Validate input
            if "pdf_file" not in request.FILES:
                return Response(
                    {"error": "No PDF file provided"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            pdf_file = request.FILES["pdf_file"]

            # Initialize LinkedIn extractor and matcher
            linkedin_extractor = EnhancedLinkedInExtractor()
            matcher = ProfileMatcher(
                "sk-3ieS4zm3RDbTv9bKm7_PuvcRPjfaFhhoSJ6lJ6VdRmT3BlbkFJMCh_LyeVG1zGzvilyHhibyl3TfLVlhX8CBTlyu1VIA"
            )

            # Extract text from PDF
            resume_text = linkedin_extractor.extract_text_from_pdf(pdf_file)
            if not resume_text:
                return Response(
                    {"error": "Failed to extract text from PDF"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Extract LinkedIn info from resume text
            linkedin_info = linkedin_extractor.extract_linkedin_info(resume_text)

            if not linkedin_info:
                return Response(
                    {"error": "No LinkedIn profile information found in resume"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            username, linkedin_url = linkedin_info

            # Fetch LinkedIn data
            linkedin_data = await linkedin_extractor.fetch_linkedin_data(username)

            # Process profiles
            async def run_sync_compare():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, matcher.process_profiles, pdf_file, linkedin_data
                )

            results = await run_sync_compare()

            return Response(
                {"linkedin_url": linkedin_url, "results": results},
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {"error": f"Processing error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class CombinedVerificationView(APIView):
    def __init__(self):
        super().__init__()
        self.project_view = ProjectVerificationView()
        self.profile_view = ProfileMatchView()
        self.job_matcher = JobMatcher(
            "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"
        )

    def post(self, request):
        try:
            return asyncio.run(self._async_post(request))
        except Exception as e:
            return Response(
                {"error": f"Async operation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    async def extract_email(self, text):
        print("enterred")
        try:
            headers = {
                "x-api-key": settings.CLAUDE_API_KEY,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            }

            prompt = f"""Given this resume text, find and return ONLY the email address. Return just the email, nothing else:
            {text}"""

            payload = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1024,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        return None

                    result = await response.json()
                    email = result["content"][0]["text"].strip()

                    # Validate email format
                    if re.match(r"[^@]+@[^@]+\.[^@]+", email):
                        return email
                    return None

        except Exception as e:
            print(f"Error extracting email: {e}")
            return None

    async def create_memory_file(self, original_file):
        """Create a new memory file from the original file content."""
        if not hasattr(original_file, "seek"):
            return None

        try:
            original_file.seek(0)
            pdf_content = original_file.read()
            original_file.seek(0)

            return InMemoryUploadedFile(
                file=io.BytesIO(pdf_content),
                field_name="pdf_file",
                name=original_file.name,
                content_type=original_file.content_type,
                size=len(pdf_content),
                charset=original_file.charset,
            )
        except Exception as e:
            print(f"Error creating memory file: {e}")
            return None

    async def _async_post(self, request):
        if "pdf_file" not in request.FILES or "job_id" not in request.data:
            return Response(
                {"error": "PDF file and job ID are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            original_file = request.FILES["pdf_file"]
            job_id = request.data.get("job_id")

            # Create file copies first
            memory_files = []
            for _ in range(3):  # We need 3 copies
                original_file.seek(0)
                pdf_content = original_file.read()
                memory_file = InMemoryUploadedFile(
                    file=io.BytesIO(pdf_content),
                    field_name="pdf_file",
                    name=original_file.name,
                    content_type=original_file.content_type,
                    size=len(pdf_content),
                    charset=original_file.charset,
                )
                memory_files.append(memory_file)

            # Extract text and LinkedIn info
            linkedin_extractor = EnhancedLinkedInExtractor()
            resume_text = linkedin_extractor.extract_text_from_pdf(memory_files[0])

            if not resume_text:
                return Response(
                    {"error": "Failed to extract text from resume"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Extract LinkedIn URL
            linkedin_info = linkedin_extractor.extract_linkedin_info(resume_text)
            linkedin_url = linkedin_info[1] if linkedin_info else None

            # Extract email
            email = await self.extract_email(resume_text)
            if not email:
                return Response(
                    {"error": "Could not find email in resume"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Get or create candidate with transaction
            try:
                candidate = await sync_to_async(Candidate.objects.get)(email=email)
            except Candidate.DoesNotExist:
                candidate = await sync_to_async(Candidate.objects.create)(
                    email=email, name=request.data.get("name", "")
                )

            # Get job
            try:
                job = await sync_to_async(Job.objects.get)(id=job_id)
            except Job.DoesNotExist:
                return Response(
                    {"error": "Job not found"}, status=status.HTTP_404_NOT_FOUND
                )

            # Create resume entry with proper version
            resume_count = await sync_to_async(
                Resume.objects.filter(candidate=candidate).count
            )()

            # Create resume with original file
            resume = await sync_to_async(Resume.objects.create)(
                candidate=candidate,
                pdf_file=original_file,
                version=resume_count + 1,
                uploaded_by_id=request.data.get("uploaded_by"),
            )

            # Prepare job requirements
            job_requirements = {
                "required_skills": job.required_skills,
                "preferred_skills": job.preferred_skills,
                "education": job.education,
                "years_of_experience": job.years_of_experience,
            }

            # Create requests with separate file copies for analysis
            project_request = type(
                "TempRequest",
                (),
                {"FILES": {"pdf_file": memory_files[1]}, "data": request.data},
            )
            profile_request = type(
                "TempRequest",
                (),
                {"FILES": {"pdf_file": memory_files[2]}, "data": request.data},
            )
            print("job requirements", job_requirements)

            # Run verifications
            results = await asyncio.gather(
                self.project_view._async_post(project_request),
                self.profile_view._async_post(profile_request),
                self.job_matcher.match_job_requirements(resume_text, job_requirements),
                return_exceptions=True,
            )

            # Create ResumeAnalysis entry
            analysis_data = {
                "project_verification": (
                    results[0].data
                    if not isinstance(results[0], Exception)
                    else {"error": str(results[0])}
                ),
                "profile_match": (
                    results[1].data
                    if not isinstance(results[1], Exception)
                    else {"error": str(results[1])}
                ),
                "job_match": (
                    results[2]
                    if not isinstance(results[2], Exception)
                    else {"error": str(results[2])}
                ),
            }

            await sync_to_async(ResumeAnalysis.objects.create)(
                resume=resume,
                job=job,
                linkedin_url=linkedin_url,
                analysis_data=analysis_data,
            )

            return Response(
                {
                    "candidate_id": candidate.id,
                    "resume_id": resume.id,
                    "job_id": job.id,
                    "analysis": analysis_data,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {"error": f"Processing error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class RegisterView(APIView):
    def post(self, request: HttpRequest) -> Response:
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            return Response(
                {
                    "refresh": str(refresh),
                    "access": str(refresh.access_token),
                    "user": RecruiterSerializer(user).data,
                },
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    def post(self, request: HttpRequest) -> Response:
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data
            refresh = RefreshToken.for_user(user)
            return Response(
                {
                    "refresh": str(refresh),
                    "access": str(refresh.access_token),
                    "user": RecruiterSerializer(user).data,
                }
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class GetProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: HttpRequest) -> Response:
        serializer = RecruiterSerializer(request.user)
        return Response(serializer.data)


class JobViewSet(viewsets.ModelViewSet):
    serializer_class = JobSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """
        Filter jobs to show only those the current recruiter has access to:
        - Jobs they created
        - Jobs they are assigned to via JobRecruiter
        """
        user = self.request.user

        # Fetch job IDs where the current user is a recruiter
        job_ids = JobRecruiter.objects.filter(
            recruiter=user, status="ACTIVE"  # Filter only active status if needed
        ).values_list("job_id", flat=True)

        # Fetch jobs created by the user or assigned via JobRecruiter
        return (
            Job.objects.filter(models.Q(created_by=user) | models.Q(id__in=job_ids))
            .distinct()
            .prefetch_related("job_recruiters")
        )

    @action(detail=True, methods=["POST"])
    def join(self, request, pk=None):
        """
        Allow any authenticated recruiter to join any job.
        The get_object() method here will use the unfiltered queryset.

        Endpoint: POST /api/jobs/{id}/join

        Returns:
            200: Successfully joined the job
            400: Already joined or invalid request
            404: Job not found
        """
        try:
            with transaction.atomic():
                # Use unfiltered queryset to get any job
                job = Job.objects.get(pk=pk)

                # Check if recruiter is already joined using JobRecruiter table
                if JobRecruiter.objects.filter(
                    job=job, recruiter=request.user, status="ACTIVE"
                ).exists():
                    return Response(
                        {"error": "You have already joined this job"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                # Create the JobRecruiter relationship
                JobRecruiter.objects.create(
                    job=job, recruiter=request.user, status="ACTIVE"
                )

                return Response(
                    {
                        "message": "Successfully joined the job",
                        "job_id": job.id,
                        "job_title": job.title,
                        "company_name": job.company_name,
                    },
                    status=status.HTTP_200_OK,
                )

        except Job.DoesNotExist:
            return Response(
                {"error": "Job not found"}, status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": f"An unexpected error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def get_object(self):
        """
        Override get_object to use unfiltered queryset for actions like 'join'
        """
        if self.action == "join":
            # For join action, use unfiltered queryset
            queryset = Job.objects.all()
        else:
            # For other actions, use filtered queryset
            queryset = self.get_queryset()

        # Lookup the object
        lookup_url_kwarg = self.lookup_url_kwarg or self.lookup_field
        filter_kwargs = {self.lookup_field: self.kwargs[lookup_url_kwarg]}
        obj = get_object_or_404(queryset, **filter_kwargs)

        # May raise a permission denied
        self.check_object_permissions(self.request, obj)

        return obj

    def create(self, request, *args, **kwargs):
        """
        Create a job from LinkedIn URL
        """
        linkedin_url = request.data.get("linkedin_url")
        if not linkedin_url:
            return Response(
                {"error": "linkedin_url is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        print(request.user)
        # Initialize job parser

        job_parser = JobParser(user=request.user)

        # Use async_to_sync to handle the async operation
        results = async_to_sync(job_parser.analyze_jobs)(urls=[linkedin_url])

        if not results or not results[0]["success"]:
            return Response(
                {"error": results[0].get("error", "Failed to parse job details")},
                status=status.HTTP_400_BAD_REQUEST,
            )

        job_data = results[0]["data"]

        # Transform the parsed data to match your Job model fields
        job_create_data = {
            "company_name": job_data["company"]["name"],  # Accessing nested key
            "description": job_data["job_description"],  # Directly accessing key
            "source_url": linkedin_url,  # A variable containing the LinkedIn URL
            **job_data,  # This unpacks and includes all keys/values from job_data
        }

        # Use your existing serializer to create the job
        serializer = self.get_serializer(data=job_create_data)
        serializer.is_valid(raise_exception=True)
        job = serializer.save(created_by=request.user)
        JobRecruiter.objects.create(job=job, recruiter=request.user, status="ACTIVE")
        headers = self.get_success_headers(serializer.data)
        return Response(
            serializer.data, status=status.HTTP_201_CREATED, headers=headers
        )

        headers = self.get_success_headers(serializer.data)
        return Response(
            serializer.data, status=status.HTTP_201_CREATED, headers=headers
        )

    @action(detail=True, methods=["get"])
    def recruiters(self, request, pk=None):
        """
        Get all recruiters associated with a job
        """
        job = self.get_object()
        serializer = RecruiterSerializer(job.recruiters.all(), many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["delete"])
    def leave(self, request, pk=None):
        """
        Allow a recruiter to leave a job
        """
        job = self.get_object()
        if not job.recruiters.filter(id=request.user.id).exists():
            return Response(
                {"detail": "Not joined to this job"}, status=status.HTTP_400_BAD_REQUEST
            )
        JobRecruiter.objects.filter(job=job, recruiter=request.user).delete()
        return Response({"detail": "Successfully left the job"})


class ShortlistViewSet(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated]

    def get_job(self, job_id):
        """Helper method to get job and verify access"""
        has_access = JobRecruiter.objects.filter(
            recruiter_id=self.request.user.id, job_id=job_id, status="ACTIVE"
        ).exists()

        if not has_access:
            raise Http404("Job not found or you don't have permission")

        return get_object_or_404(Job, id=job_id)

    def remove_from_shortlist(self, request, pk=None, resume_id=None):
        """
        Remove a resume from job's shortlist
        """
        try:
            job = self.get_job(pk)
            shortlist = get_object_or_404(Shortlist, job_id=pk, resume_id=resume_id)

            if (
                shortlist.shortlisted_by != request.user
                and job.created_by != request.user
            ):
                return Response(
                    {"error": "You do not have permission to remove this shortlist"},
                    status=status.HTTP_403_FORBIDDEN,
                )

            shortlist.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Http404:
            return Response(
                {"error": "Shortlist not found"}, status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def add_to_shortlist(self, request, pk=None, resume_id=None):
        try:
            with transaction.atomic():
                job = self.get_job(pk)
                resume = get_object_or_404(Resume, id=resume_id)

                if Shortlist.objects.filter(job=job, resume=resume).exists():
                    return Response(
                        {"error": "Resume already shortlisted for this job"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                shortlist = Shortlist.objects.create(
                    job=job, resume=resume, shortlisted_by=request.user
                )

                serializer = ShortlistSerializer(shortlist)
                return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def get_shortlisted(self, request, pk=None):
        try:
            self.get_job(pk)
            shortlists = Shortlist.objects.filter(job_id=pk)
            serializer = ShortlistSerializer(shortlists, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CandidateDetailView(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = CandidateSerializer
    queryset = Candidate.objects.all()

    def get(self, request, *args, **kwargs):
        try:
            candidate = self.get_object()
            serializer = self.get_serializer(candidate)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CandidateResumesView(generics.ListAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = ResumeVersionSerializer

    def get_queryset(self):
        candidate_id = self.kwargs.get("pk")
        return Resume.objects.filter(candidate_id=candidate_id).order_by("-version")

    def get(self, request, *args, **kwargs):
        try:
            # Verify candidate exists
            get_object_or_404(Candidate, id=self.kwargs.get("pk"))

            resumes = self.get_queryset()
            serializer = self.get_serializer(resumes, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
