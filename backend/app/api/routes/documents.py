import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_document_service
from app.domain.documents import ParsedDocument
from app.api.schemas.document import DocumentRequest
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["documents"])


@router.post("/documents", response_model=ParsedDocument)
def get_document(
    request: DocumentRequest,
    service: DocumentService = Depends(get_document_service),
) -> ParsedDocument:
    try:
        return service.get_document(str(request.url))
    except Exception:
        logger.warning("Failed to fetch document url=%r", request.url, exc_info=True)
        raise HTTPException(
            status_code=502, detail="Could not retrieve the requested page"
        ) from None