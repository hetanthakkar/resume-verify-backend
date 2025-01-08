from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import VerificationResult
from .serializers import VerificationResultSerializer
from .resume_extractor import ResumeProjectExtractor
from .projectverifier import ProjectVerifier
import asyncio


class ProjectVerificationView(APIView):
    def post(self, request):
        try:
            if "pdf_file" not in request.FILES:
                return Response(
                    {"error": "No PDF file provided"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            pdf_file = request.FILES["pdf_file"]
            openai_api_key = "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"
            extractor = ResumeProjectExtractor(openai_api_key)
            verifier = ProjectVerifier(openai_api_key)

            # Extract text and links from PDF
            resume_text, links = extractor.extract_text_from_pdf(pdf_file)
            if not resume_text:
                return Response(
                    {"error": "Failed to extract text from PDF"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Extract projects
            projects = extractor.extract_projects(resume_text, links)

            # Verify projects
            verification_results = []
            for project in projects:
                if project.get("url"):
                    result = asyncio.run(
                        verifier.verify_project(project["url"], project["description"])
                    )
                    # Convert ProjectVerificationResult to structured dictionary
                    structured_result = result.create_structured_output()
                    verification_results.append(
                        {"project": project, "verification": structured_result}
                    )
                else:
                    verification_results.append(
                        {
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
                    )

            # Create final report
            total_projects = len(projects)
            projects_with_urls = sum(1 for p in projects if p.get("url"))
            verified_projects = sum(
                1
                for r in verification_results
                if r["verification"]["summary"]["match_score"] >= 7.0
            )

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

            # Save the results
            verification_result = VerificationResult.objects.create(
                pdf_file=pdf_file, results=final_report
            )

            serializer = VerificationResultSerializer(verification_result)
            return Response(serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
