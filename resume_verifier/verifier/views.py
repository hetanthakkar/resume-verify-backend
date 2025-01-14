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
from django.core.mail import send_mail

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
    OTPVerification,
    Recruiter,
)
from rest_framework_simplejwt.tokens import RefreshToken, TokenError

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
    VerifyOTPSerializer,
    ForgotPasswordSerializer,
    ResetPasswordSerializer,
    RecruiterEmailUpdateRequestSerializer,
    RecruiterEmailUpdateConfirmSerializer,
    RecruiterProfileUpdateSerializer,
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
            if "pdf_file" not in request.FILES:
                return Response(
                    {"error": "No PDF file provided"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            pdf_file = request.FILES["pdf_file"]
            linkedin_extractor = EnhancedLinkedInExtractor()
            matcher = ProfileMatcher(
                "sk-3ieS4zm3RDbTv9bKm7_PuvcRPjfaFhhoSJ6lJ6VdRmT3BlbkFJMCh_LyeVG1zGzvilyHhibyl3TfLVlhX8CBTlyu1VIA"
            )

            resume_text = linkedin_extractor.extract_text_from_pdf(pdf_file)
            if not resume_text:
                return Response(
                    {"error": "Failed to extract text from PDF"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            linkedin_response = await linkedin_extractor.extract_linkedin_info(
                resume_text
            )
            if not linkedin_response:
                return Response(
                    {"error": "No LinkedIn profile information found in resume"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            username, linkedin_url = linkedin_response
            print("My results", username, linkedin_url)
            linkedin_data = await linkedin_extractor.fetch_linkedin_data(username)
            print("My results1", linkedin_data)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, matcher.process_profiles, pdf_file, linkedin_data
            )

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

    async def extract_email_and_name(self, text):
        try:
            headers = {
                "x-api-key": settings.CLAUDE_API_KEY,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            }

            prompt = f"""Given this resume text, find and return ONLY the full name and email address in the following format:
            NAME: <full name>
            EMAIL: <email>

            Resume text:
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
                        return None, None

                    result = await response.json()
                    response_text = result["content"][0]["text"].strip()

                    # Extract name and email from response
                    name_match = re.search(r"NAME:\s*(.+)", response_text)
                    email_match = re.search(r"EMAIL:\s*(.+)", response_text)

                    name = name_match.group(1).strip() if name_match else None
                    email = email_match.group(1).strip() if email_match else None

                    if email and not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                        email = None

                    return name, email

        except Exception as e:
            print(f"Error extracting email and name: {e}")
            return None, None

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
            # print("extracted resume text", resume_text)
            # Extract LinkedIn URL
            linkedin_info = await linkedin_extractor.extract_linkedin_info(resume_text)
            # print("extracted linkedin info", linkedin_info)
            linkedin_url = linkedin_info[1] if linkedin_info else None

            # Extract email
            name, email = await self.extract_email_and_name(resume_text)
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
                    email=email, name=name
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
                uploaded_by=request.user,
                candidate_id=candidate,
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
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(
                {"message": "Please verify your email with the OTP sent"},
                status=status.HTTP_200_OK,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RecentAnalysisView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            analyses = (
                ResumeAnalysis.objects.filter(uploaded_by=request.user)
                .select_related("resume", "job", "resume__candidate")
                .order_by("-analyzed_at")[:10]
            )

            return Response(
                [
                    {
                        "id": analysis.id,
                        "job_id": analysis.job.id,
                        "job_title": analysis.job.title,
                        "candidate_name": analysis.resume.candidate.name,
                        "candidate_email": analysis.resume.candidate.email,
                        "analysis_data": analysis.analysis_data,
                        "analyzed_at": analysis.analyzed_at,
                        "resume_id": analysis.resume.id,
                    }
                    for analysis in analyses
                ]
            )
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class JobAnalysisView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, job_id):
        try:
            analyses = (
                ResumeAnalysis.objects.filter(
                    uploaded_by_id=request.user.id, job_id=job_id
                )
                .select_related("resume", "job", "resume__candidate")
                .order_by("-analyzed_at")[:10]
            )

            return Response(
                [
                    {
                        "id": analysis.id,
                        "candidate_name": analysis.resume.candidate.name,
                        "candidate_email": analysis.resume.candidate.email,
                        "analysis_data": analysis.analysis_data,
                        "analyzed_at": analysis.analyzed_at,
                        "resume_id": analysis.resume.id,
                    }
                    for analysis in analyses
                ]
            )
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CheckEmailRegisteredView(APIView):
    def post(self, request):
        email = request.data.get("email")

        if not email:
            return Response(
                {"error": "Email field is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if Recruiter.objects.filter(email=email).exists():
            return Response(
                {"message": "Email is registered."},
                status=status.HTTP_200_OK,
            )

        return Response(
            {"message": "Email is not registered."},
            status=status.HTTP_404_NOT_FOUND,
        )


class RefreshTokenView(APIView):
    """
    This endpoint allows users to refresh their access token using a valid refresh token.
    """

    def post(self, request):
        refresh_token = request.data.get("refresh")

        if not refresh_token:
            return Response(
                {"error": "Refresh token is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Decode and validate the refresh token
            refresh = RefreshToken(refresh_token)

            # Create a new access token from the refresh token
            new_access_token = str(refresh.access_token)

            return Response(
                {"access": new_access_token},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response(
                {"error": "Invalid or expired refresh token."},
                status=status.HTTP_400_BAD_REQUEST,
            )


class VerifyOTPView(APIView):
    def post(self, request):
        serializer = VerifyOTPSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            verification = OTPVerification.objects.get(
                email=serializer.validated_data["email"],
                otp=serializer.validated_data["otp"],
                is_verified=False,
            )

            if not verification.is_valid():
                return Response(
                    {"error": "OTP has expired"}, status=status.HTTP_400_BAD_REQUEST
                )

            register_serializer = RegisterSerializer(
                data=verification.registration_data, context={"registration_data": True}
            )

            if register_serializer.is_valid():
                user = register_serializer.save()
                verification.is_verified = True
                verification.save()

                refresh = RefreshToken.for_user(user)
                return Response(
                    {
                        "refresh": str(refresh),
                        "access": str(refresh.access_token),
                        "user": RecruiterSerializer(user).data,
                    },
                    status=status.HTTP_201_CREATED,
                )
            return Response(
                register_serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )

        except OTPVerification.DoesNotExist:
            return Response(
                {"error": "Invalid OTP"}, status=status.HTTP_400_BAD_REQUEST
            )


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

    def update(self, request, *args, **kwargs):
        """
        Update an existing job
        Endpoint: PUT /api/jobs/{id}/
        """
        partial = kwargs.pop("partial", False)
        instance = self.get_object()

        # Ensure that only the job creator can update the job
        if instance.created_by != request.user:
            return Response(
                {"error": "You do not have permission to update this job"},
                status=status.HTTP_403_FORBIDDEN,
            )

        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        return Response(serializer.data, status=status.HTTP_200_OK)

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
            "company_name": job_data["company"]["name"],
            "description": job_data["job_description"],
            "source_url": linkedin_url,
            **job_data,
        }

        # Use your existing serializer to create the job
        serializer = self.get_serializer(data=job_create_data)
        serializer.is_valid(raise_exception=True)
        job = serializer.save(created_by=request.user)
        JobRecruiter.objects.create(job=job, recruiter=request.user, status="ACTIVE")

        # Only one response block
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
            recruiter_id=self.request.user.id, job_id=job_id
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
    lookup_field = "id"

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


class ResumeDetailView(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = ResumeVersionSerializer
    queryset = Resume.objects.all()
    lookup_field = "id"

    def get(self, request, *args, **kwargs):
        try:
            resume = self.get_object()
            serializer = self.get_serializer(resume)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ForgotPasswordView(APIView):
    def post(self, request):
        serializer = ForgotPasswordSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data["email"]
            try:
                recruiter = Recruiter.objects.get(email=email)
                verification = OTPVerification.generate_otp(email)

                # Send password reset email
                send_mail(
                    "Password Reset Code",
                    f"Your password reset code is: {verification.otp}",
                    settings.EMAIL_FROM_ADDRESS,
                    [email],
                    fail_silently=False,
                )

                return Response(
                    {"message": "Password reset code has been sent to your email"},
                    status=status.HTTP_200_OK,
                )
            except Recruiter.DoesNotExist:
                # For security reasons, still return success even if email doesn't exist
                return Response(
                    {
                        "message": "If an account exists with this email, a password reset code has been sent"
                    },
                    status=status.HTTP_200_OK,
                )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ResetPasswordView(APIView):
    def post(self, request):
        serializer = ResetPasswordSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data["email"]
            new_password = serializer.validated_data["new_password"]

            try:
                verification = OTPVerification.objects.get(
                    email=email, otp=serializer.validated_data["otp"], is_verified=False
                )

                recruiter = Recruiter.objects.get(email=email)
                recruiter.set_password(new_password)
                recruiter.save()

                # Mark OTP as verified
                verification.is_verified = True
                verification.save()

                return Response(
                    {"message": "Password has been reset successfully"},
                    status=status.HTTP_200_OK,
                )
            except (OTPVerification.DoesNotExist, Recruiter.DoesNotExist):
                return Response(
                    {"message": "Invalid reset attempt"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RecruiterProfileUpdateView(APIView):
    permission_classes = [IsAuthenticated]

    def put(self, request):
        serializer = RecruiterProfileUpdateSerializer(
            request.user, data=request.data, partial=True
        )

        if serializer.is_valid():
            serializer.save()
            return Response(
                {"message": "Profile updated successfully", "data": serializer.data}
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RecruiterEmailUpdateRequestView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):

        serializer = RecruiterEmailUpdateRequestSerializer(data=request.data)
        if serializer.is_valid():
            new_email = serializer.validated_data["new_email"]
            # Generate and send OTP
            verification = OTPVerification.generate_otp(new_email)

            # Send email with OTP
            send_mail(
                "Email Change Verification",
                f"Your verification code to change your email is: {verification.otp}",
                settings.EMAIL_FROM_ADDRESS,
                [new_email],
                fail_silently=False,
            )

            return Response(
                {
                    "message": "Verification code sent to new email address",
                    "new_email": new_email,
                }
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RecruiterEmailUpdateConfirmView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = RecruiterEmailUpdateConfirmSerializer(data=request.data)
        if serializer.is_valid():
            new_email = serializer.validated_data["new_email"]

            # Verify OTP
            verification = OTPVerification.objects.get(
                email=new_email, otp=serializer.validated_data["otp"], is_verified=False
            )

            # Update email
            request.user.email = new_email
            request.user.save()

            # Mark OTP as verified
            verification.is_verified = True
            verification.save()

            return Response(
                {"message": "Email updated successfully", "email": new_email}
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LogoutView(APIView):
    """
    Logout endpoint that blacklists the refresh token to prevent reuse.
    Requires a refresh token in the request body.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get("refresh")
            if not refresh_token:
                return Response(
                    {"error": "Refresh token is required."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Get token and blacklist it
            token = RefreshToken(refresh_token)
            token.blacklist()

            return Response(
                {"message": "Successfully logged out."}, status=status.HTTP_200_OK
            )
        except TokenError:
            return Response(
                {"error": "Invalid or expired refresh token."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            return Response(
                {"error": "Failed to logout."}, status=status.HTTP_400_BAD_REQUEST
            )
