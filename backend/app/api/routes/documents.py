from fastapi import APIRouter

from app.schemas.document import DocumentRequest, ParsedDocument
from app.services.document_service import DocumentService

router = APIRouter(prefix="/api/v1", tags=["documents"])


@router.post("/documents", response_model=ParsedDocument)
async def get_document(request: DocumentRequest) -> ParsedDocument:
    service = DocumentService()
    return service.get_document(request.url)