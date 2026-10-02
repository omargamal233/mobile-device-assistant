from fastapi import FastAPI, HTTPException

from app.schema.device import ExtractRequest, ExtractResponse, SearchRequest, SearchResponse, SearchResult
from app.services.extraction import extract_device
from app.services.security.security_check_point import UnsafeInputError, check_input
from app.services.search import hybrid_search


app = FastAPI(title="Mobile Device Assistant")


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> ExtractResponse:
	try:
		cleaned_text = check_input(request.text)
	except UnsafeInputError as error:
		raise HTTPException(status_code=400, detail=str(error)) from error

	device = extract_device(cleaned_text)
	return ExtractResponse.model_validate(device.model_dump())


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
	matches = hybrid_search(request.q)
	return SearchResponse(
		results=[SearchResult(document=document, score=score) for document, score in matches]
	)
