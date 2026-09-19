from fastapi import Request

from app.services.document_service import DocumentService
from app.services.research_service import ResearchService
from app.services.search_service import SearchService


def get_research_service(request: Request) -> ResearchService:
    return request.app.state.research_service


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service