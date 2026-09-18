"""Tests for the policy page."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.policy import Policy, get_policy
from app.policy_view import render_policy_html


@pytest.fixture(scope="module")
def html() -> str:
    return render_policy_html(get_policy())


def test_every_section_gets_an_anchor(html):
    """A citation is only followable if its id is an anchor on the page."""
    for section_id in get_policy().sections:
        assert f'id="{section_id}"' in html


def test_headings_and_body_survive_rendering(html):
    assert "<h1>Sample Expense &amp; Finance Policy</h1>" in html
    assert 'id="3.2"' in html and "Missing or lost receipts" in html
    assert "missing-receipt declaration in Brex" in html
    assert "<ul>" in html and "<blockquote>" in html


def test_markdown_syntax_is_not_left_visible(html):
    body = html.split('<div class="card">')[1]
    assert "### " not in body
    assert "\n## " not in body


def test_policy_content_is_escaped():
    """The policy is our file, but it is still rendered as text, not markup."""
    hostile = Policy(
        text="# Title\n\n### 1.1 Section\n\n<script>alert('x')</script> & more\n",
        sections={"1.1": "Section"},
    )

    out = render_policy_html(hostile)

    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "&amp; more" in out


def test_policy_routes_serve_html_and_markdown():
    with TestClient(app) as client:
        page = client.get("/policy")
        raw = client.get("/policy.md")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert 'id="4.2"' in page.text

    assert raw.status_code == 200
    assert raw.headers["content-type"].startswith("text/markdown")
    assert raw.text == get_policy().text
