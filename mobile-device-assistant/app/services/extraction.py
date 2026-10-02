import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from app.schema.device import Device 
from ..prompts.templates import build_prompt      

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_RETRIES = 2


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_device(text: str) -> Device:
    messages = build_prompt(text)   
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content

        try:
            data = json.loads(raw)
            return Device.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Your previous response failed validation:\n{e}\n\n"
                    "Return corrected JSON only, matching the schema exactly."
                ),
            })

    raise ValueError(f"Extraction failed after {MAX_RETRIES} retries: {last_error}")
