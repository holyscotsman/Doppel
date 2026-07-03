"""Ollama VLM plumbing. Every VLM call goes through OllamaClient:
JSON-schema-forced output, versioned prompt files from prompts/, and the
model + prompt_version stored with each result (CLAUDE.md rule).

The VLM never scans the library — stages 1-3 nominate a small candidate
set with cheap math; the VLM only rules on nominees.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

PROMPTS_DIR = Path("prompts")


class VlmClient(Protocol):
    """A vision model that answers with schema-conforming JSON."""

    def chat_json(
        self, prompt: str, images: list[bytes], schema: dict[str, Any]
    ) -> dict[str, Any]: ...


class OllamaClient:
    """Real client for a local Ollama server.

    One request is in flight at a time by construction: all VLM work runs
    on the single JobRunner worker thread. The model must be vision-capable
    and, for adjudication, accept multiple images per prompt (qwen3-vl,
    gemma3 — llama3.2-vision cannot compare a pair).
    """

    def __init__(self, host: str, model: str) -> None:
        self.host = host
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.host)
        return self._client

    def chat_json(
        self, prompt: str, images: list[bytes], schema: dict[str, Any]
    ) -> dict[str, Any]:
        response = self._get_client().chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": images}],
            format=schema,  # forces schema-conforming JSON output
        )
        return json.loads(response["message"]["content"])


def latest_prompt(task: str, prompts_dir: Path | str = PROMPTS_DIR) -> tuple[str, str]:
    """(text, version) of the highest-versioned prompts/{task}_v{N}.txt.

    Bumping the prompt version means adding a new file; results are keyed
    by (model, prompt_version) so prior results are never clobbered.
    """
    prompts_dir = Path(prompts_dir)
    candidates: list[tuple[int, Path]] = []
    for path in prompts_dir.glob(f"{task}_v*.txt"):
        match = re.fullmatch(rf"{re.escape(task)}_v(\d+)\.txt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(
            f"no prompt files matching {task}_v*.txt in {prompts_dir}"
        )
    version, path = max(candidates)
    return path.read_text(), f"v{version}"
