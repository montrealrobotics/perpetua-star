import re
from typing import List

from google import genai
from google.genai import errors
from pydantic import BaseModel, Field

import logging

# A logger for this file
logger = logging.getLogger(__name__)

_DAY_TIME = r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun): \d{2}:\d{2}"
_INTERVAL = re.compile(rf"^{_DAY_TIME} - {_DAY_TIME}$")


class WeeklyOutput(BaseModel):
    schedule: List[str] = Field(
        description="A List of strings with the format ['Day: HH:MM - Day: HH:MM', 'Day: HH:MM - Day: HH:MM', ...] indicating the time intervals when an object is present"
    )


def query_weekly_schedule(prompt: str, model: str = "gemini-pro-latest", n_attempts: int = 10) -> List[str]:
    try:
        client = genai.Client()
    except ValueError as e:
        raise RuntimeError("Set GEMINI_API_KEY to query Gemini for LLM schedules.") from e
    error = None
    for attempt in range(n_attempts):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": WeeklyOutput.model_json_schema(),
                },
            )
            output = WeeklyOutput.model_validate_json(response.text).schedule
            if not output or not all(_INTERVAL.match(item) for item in output):
                raise ValueError(f"Unexpected schedule format: {output}")
            return output
        except Exception as e:
            if isinstance(e, errors.ClientError) and e.code != 429:
                raise RuntimeError(f"Gemini query failed: {e}") from e
            error = e
            logger.info(f"Attempt {attempt + 1} of the Gemini query failed: {e}")
    raise RuntimeError(f"Gemini query failed after {n_attempts} attempts. Last error: {error}")
