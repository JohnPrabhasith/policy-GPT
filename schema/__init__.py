from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL of the PDF document to process.", example="https://hackrx.blob.core.windows.net/assets/policy.pdf?...")
    questions: list[str] = Field(..., description="A list of questions to ask about the document.", example=["What is the grace period for premium payment?"])

class QueryResponse(BaseModel):
    answers: list[str] = Field(..., description="A list of answers corresponding to the questions.", example=["A grace period of thirty days is provided..."])
