from datetime import date
from typing import Annotated, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SpecKey = Annotated[str, Field(min_length=2, max_length=50)]
SpecValue = Annotated[str, Field(min_length=2, max_length=100)]


class Device(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    brand: str = Field(min_length=2, max_length=50)
    model: str = Field(min_length=2, max_length=100)
    specs: Dict[SpecKey, SpecValue] = Field(min_length=1)
    release_year: int = Field(ge=1990)
    price_tier: Literal["budget", "mid-range", "flagship"]

    @field_validator("release_year")
    @classmethod
    def year_not_too_far_ahead(cls, v: int) -> int:
        if v > date.today().year + 1:
            raise ValueError("release_year is too far in the future")
        return v


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)


class ExtractResponse(Device):
    pass


class SearchRequest(BaseModel):
    q: str = Field(min_length=1, max_length=500)


class SearchResult(BaseModel):
    document: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]