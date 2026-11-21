"""Verify the configured OpenAI model with one small request."""

import os
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
    if not api_key:
        print("OPENAI_API_KEY is missing. Add a project API key to .env.", file=sys.stderr)
        return 1
    if not model:
        print("OPENAI_MODEL is empty. Set it to an accessible model in .env.", file=sys.stderr)
        return 1

    try:
        with openai.OpenAI(api_key=api_key, max_retries=0, timeout=30.0) as client:
            response = client.responses.create(
                model=model,
                input="Reply with OK.",
                max_output_tokens=32,
            )
        if response.status != "completed" or not response.output_text.strip():
            print(
                "Model check failed: the response was incomplete or contained no text. "
                "Check the model and project access.",
                file=sys.stderr,
            )
            return 1
    except openai.APIStatusError as exc:
        status = exc.status_code
        if status in (401, 403):
            fix = "Check the project API key and model permissions."
        elif status == 404:
            fix = "Check the model name and whether the project can access it."
        elif status == 429:
            fix = "Check project quota, billing, and rate limits."
        elif status == 400:
            fix = "Check that the model supports the Responses API and this request."
        else:
            fix = "Check the API service status and project configuration."
        print(f"Model check failed (HTTP {status}). {fix}", file=sys.stderr)
        return 1
    except openai.APIConnectionError:
        print("Model check failed: connection or timeout error. Check network access.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"Model check failed ({type(exc).__name__}). Check the local SDK and configuration.",
            file=sys.stderr,
        )
        return 1

    print("Model check passed: received a completed text response.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
