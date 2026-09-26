from pydantic import BaseModel, HttpUrl


class DocumentRequest(BaseModel):
    url: HttpUrl
