"""Tests for loading and validating the expense policy."""

import pytest

from app.policy import PolicyLoadError, get_policy, load_policy, validate_sources


def test_real_policy_loads_with_known_sections():
    """The shipped policy file parses and exposes its subsection ids."""
    policy = get_policy()

    assert "3.2" in policy.sections
    assert "7.4" in policy.sections
    assert policy.sections["3.2"] == "Missing or lost receipts"
    assert policy.text.startswith("# Sample Expense & Finance Policy")


def test_top_level_headings_are_ignored():
    """Only '### N.N' subsections are indexed, not '## N.' headings."""
    policy = get_policy()

    assert all("." in section_id for section_id in policy.sections)
    assert "3" not in policy.sections


def test_validate_sources_reports_only_unknown_ids():
    """Known ids pass through; unknown ones are returned."""
    policy = get_policy()

    assert validate_sources(["3.2", "9.9"], policy) == ["9.9"]
    assert validate_sources(["3.2", "7.4"], policy) == []
    assert validate_sources([], policy) == []


def test_policy_without_subsections_is_rejected(tmp_path):
    """A file with no '### N.N' headings fails loudly."""
    broken = tmp_path / "broken_policy.md"
    broken.write_text("# Title\n\n## 1. Purpose\n\nNo subsections here.\n", encoding="utf-8")

    with pytest.raises(PolicyLoadError, match="No '### N.N"):
        load_policy(broken)


def test_missing_policy_file_is_rejected(tmp_path):
    """A missing file fails with a clear message rather than at request time."""
    with pytest.raises(PolicyLoadError, match="not found"):
        load_policy(tmp_path / "does_not_exist.md")
