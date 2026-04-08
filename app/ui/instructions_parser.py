"""Parse an instructions.md markdown file into a structured dict.

All sections are optional. Unrecognised lines go into a general notes list.
"""

from __future__ import annotations

import re


def parse_instructions(markdown_text: str) -> dict:
    """Parse instructions markdown into structured sections.

    Expected top-level ``## `` headers (case-insensitive):
      Dataset, Priorities, Features, Models, Visualization

    Within Features and Models, recognised prefixes:
      Must include:, Avoid:, Consider:, Preferred:, Notes:
    """
    result: dict = {
        "raw": markdown_text,
        "dataset": {},
        "priorities": [],
        "features": {"must_include": [], "avoid": [], "consider": []},
        "models": {"preferred": [], "avoid": [], "notes": []},
        "visualization": [],
    }

    if not markdown_text or not markdown_text.strip():
        return result

    # Split into sections on ## headers
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for line in markdown_text.splitlines():
        header_match = re.match(r"^##\s+(.+)", line)
        if header_match:
            current_section = header_match.group(1).strip().lower()
            sections.setdefault(current_section, [])
            continue
        if current_section is not None:
            sections.setdefault(current_section, []).append(line)

    # --- Dataset section ---
    for key in ("dataset", "data"):
        if key in sections:
            _parse_dataset_section(sections[key], result["dataset"])
            break

    # --- Priorities section ---
    for key in ("priorities", "priority", "goals"):
        if key in sections:
            result["priorities"] = _extract_bullets(sections[key])
            break

    # --- Features section ---
    for key in ("features", "feature engineering", "feature"):
        if key in sections:
            _parse_keyed_section(
                sections[key],
                result["features"],
                known_keys={"must_include", "must include", "avoid", "consider"},
            )
            # Normalise "must include" → "must_include"
            if "must include" in result["features"]:
                result["features"]["must_include"] = result["features"].pop("must include")
            break

    # --- Models section ---
    for key in ("models", "model", "modeling"):
        if key in sections:
            _parse_keyed_section(
                sections[key],
                result["models"],
                known_keys={"preferred", "prefer", "avoid", "notes", "note"},
            )
            # Normalise aliases
            for alias, canon in [("prefer", "preferred"), ("note", "notes")]:
                if alias in result["models"]:
                    result["models"][canon] = result["models"].pop(alias)
            break

    # --- Visualization section ---
    for key in ("visualization", "visualizations", "viz", "charts"):
        if key in sections:
            result["visualization"] = _extract_bullets(sections[key])
            break

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)")
_KV_RE = re.compile(r"^(\w[\w\s]*?):\s*(.*)")


def _extract_bullets(lines: list[str]) -> list[str]:
    """Return a flat list of bullet-point contents."""
    items: list[str] = []
    for line in lines:
        m = _BULLET_RE.match(line)
        if m:
            text = m.group(1).strip()
            if text:
                items.append(text)
    return items


def _parse_dataset_section(lines: list[str], dest: dict) -> None:
    """Parse key: value pairs and bullets in the dataset section."""
    key_map = {
        "dtype": "dtype",
        "type": "dtype",
        "format": "dtype",
        "target_column": "target_column",
        "target column": "target_column",
        "target": "target_column",
        "problem_type": "problem_type",
        "problem type": "problem_type",
        "problem": "problem_type",
    }
    for line in lines:
        m = _KV_RE.match(line.strip().lstrip("-* "))
        if m:
            raw_key = m.group(1).strip().lower()
            value = m.group(2).strip()
            canon = key_map.get(raw_key)
            if canon:
                dest[canon] = value


def _parse_keyed_section(
    lines: list[str],
    dest: dict,
    known_keys: set[str],
) -> None:
    """Parse a section with ``Key: value, value`` lines and plain bullets.

    Recognised key prefixes (case-insensitive) are collected into lists;
    unrecognised bullets go into ``dest["notes"]`` (created if missing).
    """
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Try prefix match: "Must include: a, b, c" or "- Must include: a, b"
        text = _BULLET_RE.match(stripped)
        content = text.group(1) if text else stripped

        m = _KV_RE.match(content)
        if m:
            raw_key = m.group(1).strip().lower()
            values_str = m.group(2).strip()
            if raw_key in known_keys:
                # Split comma-separated values
                items = [v.strip() for v in values_str.split(",") if v.strip()]
                dest.setdefault(raw_key, []).extend(items)
                continue

        # Plain bullet — treat as note
        bullet = _BULLET_RE.match(stripped)
        if bullet:
            dest.setdefault("notes", []).append(bullet.group(1).strip())
