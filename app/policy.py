"""Loading and validation of the expense policy document."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import get_settings

# Matches subsection headings like "### 3.2 Missing or lost receipts".
# Top-level "## 3. Receipts and documentation" headings are deliberately ignored.
SECTION_PATTERN = re.compile(r"^###[ \t]+(\d+\.\d+)[ \t]+(.+?)[ \t]*$", re.MULTILINE)


class PolicyLoadError(RuntimeError):
    """The policy file is missing or contains no usable sections.

    Raised at startup so the service never accepts traffic it cannot ground in
    the policy.
    """


@dataclass(frozen=True)
class Policy:
    """The policy document and the sections that can be cited from it."""

    text: str
    sections: dict[str, str]


def load_policy(path: str | Path) -> Policy:
    """Read the policy markdown and index its subsection headings."""
    policy_path = Path(path)

    try:
        text = policy_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PolicyLoadError(
            f"Policy file not found: {policy_path.resolve()}. "
            "Check POLICY_PATH in your environment."
        ) from exc
    except OSError as exc:
        raise PolicyLoadError(
            f"Could not read policy file {policy_path.resolve()}: {exc}"
        ) from exc

    sections = {
        match.group(1): match.group(2).strip()
        for match in SECTION_PATTERN.finditer(text)
    }
    if not sections:
        raise PolicyLoadError(
            f"No '### N.N <title>' headings found in {policy_path.resolve()}. "
            "The policy cannot be cited without them."
        )

    return Policy(text=text, sections=sections)


@lru_cache
def get_policy() -> Policy:
    """Return the policy, loading it from POLICY_PATH on first use."""
    return load_policy(get_settings().POLICY_PATH)


def validate_sources(sources: list[str], policy: Policy) -> list[str]:
    """Return the section ids that do not exist in the policy."""
    return [source for source in sources if source not in policy.sections]
