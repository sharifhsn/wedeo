#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "beautifulsoup4"]
# ///
"""Scrape 'Rust Atomics and Locks' by Mara Bos into local Markdown files.

Source: https://mara.nl/atomics/
License: CC BY-NC-ND 4.0 (non-commercial, no derivatives, attribution required)

This script downloads each chapter as HTML and converts to Markdown,
preserving code blocks, headings, lists, and notes. The output is stored
verbatim (no modifications to content) for personal study reference.
"""

import re
import sys
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

BASE_URL = "https://mara.nl/atomics/"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "rust-atomics-and-locks"

# Chapter slugs in reading order
CHAPTERS = [
    "foreword",
    "preface",
    "basics",        # Ch 1: Basics of Rust Concurrency
    "atomics",       # Ch 2: Atomics
    "memory-ordering",  # Ch 3: Memory Ordering
    "building-spinlock",  # Ch 4: Building Our Own Spin Lock
    "building-channels",  # Ch 5: Building Our Own Channels
    "building-arc",  # Ch 6: Building Our Own "Arc"
    "hardware",      # Ch 7: Understanding the Processor
    "os-primitives", # Ch 8: Operating System Primitives
    "building-locks",  # Ch 9: Building Our Own Locks
    "inspiration",   # Ch 10: Ideas and Inspiration
]

ATTRIBUTION = (
    "> **Rust Atomics and Locks** by Mara Bos (O'Reilly).\n"
    "> Copyright 2023 Mara Bos. Licensed under CC BY-NC-ND 4.0.\n"
    "> Source: https://mara.nl/atomics/\n"
)


def fetch_chapter(slug: str) -> str:
    """Fetch a chapter's HTML from the website."""
    url = f"{BASE_URL}{slug}.html"
    print(f"  Fetching {url} ...", end=" ", flush=True)
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    print(f"OK ({len(resp.text)} bytes)")
    return resp.text


def html_to_markdown(html: str, slug: str) -> str:
    """Convert a chapter's HTML to clean Markdown."""
    soup = BeautifulSoup(html, "html.parser")

    # Extract the <article> content (main body)
    article = soup.find("article")
    if not article:
        # Fallback: try the whole body
        article = soup.find("body")
    if not article:
        return f"# {slug}\n\n(Could not extract content)\n"

    lines = []
    _convert_element(article, lines)
    content = "\n".join(lines)

    # Clean up excessive blank lines
    content = re.sub(r"\n{4,}", "\n\n\n", content)
    # Trim trailing whitespace per line
    content = "\n".join(line.rstrip() for line in content.split("\n"))

    return f"{ATTRIBUTION}\n---\n\n{content}\n"


def _convert_element(el: Tag, lines: list[str], depth: int = 0):
    """Recursively convert an HTML element tree to Markdown lines."""
    for child in el.children:
        if isinstance(child, NavigableString):
            text = str(child)
            # Skip pure whitespace between block elements
            if text.strip():
                lines.append(text.strip())
            continue

        if not isinstance(child, Tag):
            continue

        tag = child.name

        # Headings
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            prefix = "#" * level
            heading_text = child.get_text(strip=True)
            lines.append(f"\n{prefix} {heading_text}\n")

        # Paragraphs
        elif tag == "p":
            text = _inline_text(child)
            if text.strip():
                lines.append(f"\n{text}\n")

        # Code blocks (pre > code or just pre)
        elif tag == "pre":
            code_el = child.find("code")
            if code_el:
                code_text = code_el.get_text()
            else:
                code_text = child.get_text()
            # Detect language from class
            lang = ""
            for cls in (child.get("class") or []) + (
                (code_el.get("class") or []) if code_el else []
            ):
                if cls.startswith("language-"):
                    lang = cls.replace("language-", "")
                    break
                if cls == "rust" or "rust" in cls:
                    lang = "rust"
                    break
            lines.append(f"\n```{lang}")
            lines.append(code_text.rstrip())
            lines.append("```\n")

        # Inline code (standalone, not inside <p>)
        elif tag == "code" and child.parent and child.parent.name not in ("p", "li", "td", "th", "a"):
            lines.append(f"`{child.get_text()}`")

        # Lists
        elif tag in ("ul", "ol"):
            lines.append("")
            for i, li in enumerate(child.find_all("li", recursive=False)):
                prefix = f"{i + 1}." if tag == "ol" else "-"
                text = _inline_text(li)
                lines.append(f"{prefix} {text}")
            lines.append("")

        # Blockquotes
        elif tag == "blockquote":
            text = _inline_text(child)
            for line in text.split("\n"):
                lines.append(f"> {line}")
            lines.append("")

        # Asides (notes, tips, warnings)
        elif tag == "aside":
            label = child.get("aria-label", "Note")
            lines.append(f"\n> **{label.title()}:**")
            for p in child.find_all("p"):
                text = _inline_text(p)
                lines.append(f"> {text}")
            lines.append("")

        # Tables
        elif tag == "table":
            _convert_table(child, lines)

        # Definition lists
        elif tag == "dl":
            for dt in child.find_all("dt"):
                lines.append(f"\n**{dt.get_text(strip=True)}**")
                dd = dt.find_next_sibling("dd")
                if dd:
                    lines.append(f": {_inline_text(dd)}\n")

        # Sections (recurse)
        elif tag == "section":
            _convert_element(child, lines, depth + 1)

        # Divs and other containers (recurse)
        elif tag in ("div", "span", "figure", "figcaption", "main", "details", "summary"):
            _convert_element(child, lines, depth)

        # Images
        elif tag == "img":
            alt = child.get("alt", "")
            src = child.get("src", "")
            lines.append(f"\n![{alt}]({src})\n")

        # Skip nav, header, footer
        elif tag in ("nav", "header", "footer", "script", "style"):
            pass

        # Anything else: try to recurse
        else:
            _convert_element(child, lines, depth)


def _inline_text(el: Tag) -> str:
    """Convert inline HTML to Markdown text (bold, italic, code, links)."""
    parts = []
    for child in el.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name == "code":
                parts.append(f"`{child.get_text()}`")
            elif child.name in ("strong", "b"):
                parts.append(f"**{child.get_text()}**")
            elif child.name in ("em", "i"):
                parts.append(f"*{child.get_text()}*")
            elif child.name == "a":
                href = child.get("href", "")
                text = child.get_text()
                if href.startswith("http"):
                    parts.append(f"[{text}]({href})")
                else:
                    parts.append(text)
            elif child.name == "br":
                parts.append("\n")
            elif child.name == "sub":
                parts.append(f"_{child.get_text()}_")
            elif child.name == "sup":
                parts.append(f"^{child.get_text()}^")
            elif child.name in ("span", "small"):
                parts.append(_inline_text(child))
            elif child.name == "pre":
                code_el = child.find("code")
                code_text = code_el.get_text() if code_el else child.get_text()
                parts.append(f"\n```\n{code_text.rstrip()}\n```\n")
            else:
                parts.append(child.get_text())
    return "".join(parts)


def _convert_table(table: Tag, lines: list[str]):
    """Convert an HTML table to Markdown table."""
    rows = []
    for tr in table.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            cells.append(_inline_text(td).strip().replace("|", "\\|"))
        rows.append(cells)

    if not rows:
        return

    # Header row
    lines.append("")
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
    for row in rows[1:]:
        # Pad to same length
        while len(row) < len(rows[0]):
            row.append("")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scraping Rust Atomics and Locks → {OUTPUT_DIR}/")
    print(f"Chapters: {len(CHAPTERS)}")
    print()

    for i, slug in enumerate(CHAPTERS):
        print(f"[{i + 1}/{len(CHAPTERS)}] {slug}")
        try:
            html = fetch_chapter(slug)
            md = html_to_markdown(html, slug)

            out_path = OUTPUT_DIR / f"{i:02d}-{slug}.md"
            out_path.write_text(md, encoding="utf-8")
            print(f"  → {out_path.name} ({len(md)} chars)")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

        # Be polite
        if i < len(CHAPTERS) - 1:
            time.sleep(0.5)

    # Write a README with attribution
    readme = OUTPUT_DIR / "README.md"
    readme.write_text(
        "# Rust Atomics and Locks — Local Reference\n\n"
        "**Author:** Mara Bos\n"
        "**Publisher:** O'Reilly Media (2023)\n"
        "**License:** CC BY-NC-ND 4.0\n"
        "**Source:** https://mara.nl/atomics/\n\n"
        "These files were scraped for personal, non-commercial study reference.\n"
        "No modifications have been made to the content.\n"
        "Please purchase the book to support the author: https://marabos.nl/atomics/\n",
        encoding="utf-8",
    )

    print(f"\nDone! {len(CHAPTERS)} chapters saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
