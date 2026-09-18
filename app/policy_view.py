"""Render the policy markdown as a readable page.

Deliberately not a general markdown implementation. It covers exactly the
constructs the policy file uses, so that citations like "section 4.2" can be
followed to the text they came from without adding a dependency.
"""

from __future__ import annotations

import re
from html import escape

from app.policy import SECTION_PATTERN, Policy

HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
BULLET = re.compile(r"^[-*]\s+(.*)$")
QUOTE = re.compile(r"^>\s?(.*)$")

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Expense &amp; Finance Policy</title>
<style>
  :root {{
    --bg: #f6f7f9; --card: #fff; --ink: #17191c; --muted: #6b7280;
    --line: #e3e6ea; --accent: #2f5fe0; --mark: #fff3bf;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #14161a; --card: #1c1f24; --ink: #e9ecf1; --muted: #9aa3b0;
      --line: #2c313a; --accent: #6b93ff; --mark: #4a4223;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink); padding: 32px 16px 80px;
    font: 16px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  main {{ max-width: 760px; margin: 0 auto; }}
  .card {{
    background: var(--card); border: 1px solid var(--line);
    border-radius: 12px; padding: 8px 28px 28px;
  }}
  a {{ color: var(--accent); }}
  .back {{ display: inline-block; margin-bottom: 16px; font-size: .9rem; }}
  h1 {{ font-size: 1.6rem; letter-spacing: -0.01em; }}
  h2 {{ font-size: 1.15rem; margin-top: 34px; padding-top: 14px; border-top: 1px solid var(--line); }}
  h3 {{ font-size: 1rem; margin-top: 24px; scroll-margin-top: 20px; }}
  h3 .id {{ color: var(--muted); font-weight: 400; margin-right: 6px; }}
  h3:target {{ background: var(--mark); border-radius: 6px;
    padding: 4px 8px; margin-left: -8px; }}
  blockquote {{
    margin: 0 0 24px; padding: 12px 16px; color: var(--muted);
    background: var(--bg); border-left: 3px solid var(--line); border-radius: 0 8px 8px 0;
    font-size: .92rem;
  }}
  ul {{ padding-left: 22px; }}
  li {{ margin: 6px 0; }}
  footer {{ margin-top: 24px; color: var(--muted); font-size: .82rem; }}
</style>
</head>
<body>
<main>
  <a class="back" href="/">&larr; Back to the assistant</a>
  <div class="card">
{body}
  </div>
  <footer>{count} citable sections. Answers link here by section id,
    for example <a href="/policy#3.2">/policy#3.2</a>.</footer>
</main>
</body>
</html>
"""


def _close(state: dict, out: list[str]) -> None:
    """Close whichever block is open, if any."""
    if state["block"]:
        out.append(f"</{state['block']}>")
        state["block"] = None


def _open(tag: str, state: dict, out: list[str]) -> None:
    """Open a block, closing a different one first."""
    if state["block"] != tag:
        _close(state, out)
        out.append(f"<{tag}>")
        state["block"] = tag


def render_policy_html(policy: Policy) -> str:
    """Turn the policy markdown into a page whose sections can be linked to."""
    out: list[str] = []
    state = {"block": None}

    for line in policy.text.splitlines():
        stripped = line.strip()

        if not stripped:
            _close(state, out)
            continue

        quote = QUOTE.match(stripped)
        if quote:
            _open("blockquote", state, out)
            out.append(escape(quote.group(1)) + "<br>")
            continue

        bullet = BULLET.match(stripped)
        if bullet:
            _open("ul", state, out)
            out.append(f"<li>{escape(bullet.group(1))}</li>")
            continue

        heading = HEADING.match(stripped)
        if heading:
            _close(state, out)
            level = len(heading.group(1))
            text = heading.group(2)
            section = SECTION_PATTERN.match(stripped)
            if section:
                # The id is the citation itself, so /policy#3.2 lands here.
                out.append(
                    f'<h3 id="{escape(section.group(1))}">'
                    f'<span class="id">{escape(section.group(1))}</span>'
                    f"{escape(section.group(2))}</h3>"
                )
            else:
                out.append(f"<h{level}>{escape(text)}</h{level}>")
            continue

        _open("p", state, out)
        out.append(escape(stripped))

    _close(state, out)
    return PAGE.format(body="\n".join(out), count=len(policy.sections))
