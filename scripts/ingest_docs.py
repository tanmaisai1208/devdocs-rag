import hashlib
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


URLS = [
    "https://docs.python.org/3/library/exceptions.html",
    "https://docs.sqlalchemy.org/en/20/errors.html",
    "https://www.postgresql.org/docs/current/libpq-connect.html",
]

OUTPUT_PATH = Path("data/processed/chunks.jsonl")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

BLOCK_TAGS = {
    "p",
    "pre",
    "ul",
    "ol",
    "dl",
    "table",
    "blockquote",
}


CONTENT_SELECTORS = [
    "#docs-body",
    "div.documentwrapper div.bodywrapper div.body",
    "div.body[role='main']",
    "div#docContent",
    "main",
    "article",
]

REMOVE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    ".sphinxsidebar",
    ".sphinxsidebarwrapper",
    ".related",
    ".headerlink",
    ".navheader",
    ".navfooter",
    ".breadcrumbs",
    ".wy-nav-side",
    "#sidebar",
    "#searchbox",
]


def normalize_text(text):
    """Normalize whitespace while preserving word boundaries and paragraphs."""
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse spaces and tabs, but preserve newline characters.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around line breaks.
    text = re.sub(r" *\n *", "\n", text)

    # Collapse 3+ consecutive newlines into a paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def clean_heading(text):
    """Normalize heading text and remove trailing PostgreSQL anchor markers."""
    text = normalize_text(text)
    text = re.sub(r"\s*#\s*$", "", text)
    return text.strip()


def fetch_page(url):
    """Download a documentation page and parse its HTML."""
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "EngineeringKnowledgePlatform/0.1"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    # Remove common navigation and boilerplate elements.
    for selector in REMOVE_SELECTORS:
        for element in soup.select(selector):
            element.decompose()

    # Prefer the main documentation container over the entire page.
    root = None

    for selector in CONTENT_SELECTORS:
        root = soup.select_one(selector)
        if root is not None:
            break

    if root is None:
        raise ValueError(
            f"Could not identify the main documentation content: {url}"
        )

    print(
        "  Selected content container:",
        root.name,
        root.get("id", ""),
        root.get("class", ""),
    )

    title_tag = soup.find("title")
    title = (
        normalize_text(title_tag.get_text(" ", strip=True))
        if title_tag
        else url
    )

    return root, title


def extract_sections(root, url, title):
    """Extract heading-aware sections and individual parameter definitions."""
    sections = []
    heading_path = {}
    current_heading = title
    current_parts = []

    def save_current_section():
        """Save accumulated content under its current heading."""
        if not current_parts:
            return

        sections.append(
            Document(
                page_content=(
                    current_heading
                    + "\n\n"
                    + "\n\n".join(current_parts)
                ),
                metadata={
                    "source": url,
                    "title": title,
                    "section": current_heading,
                },
            )
        )
        current_parts.clear()

    elements = root.find_all(list(HEADINGS | BLOCK_TAGS))

    for element in elements:
        # Skip blocks nested inside another block. Their text will be
        # captured by the outer block, preventing duplicate extraction.
        if any(
            parent.name in BLOCK_TAGS
            for parent in element.parents
            if parent is not root
        ):
            continue

        if element.name in HEADINGS:
            save_current_section()

            level = int(element.name[1])
            heading_text = clean_heading(
                element.get_text(" ", strip=True)
            )

            heading_path = {
                depth: text
                for depth, text in heading_path.items()
                if depth < level
            }

            if heading_text:
                heading_path[level] = heading_text

            current_heading = " > ".join(
                heading_path[depth]
                for depth in sorted(heading_path)
            ) or title

            continue

        if element.name == "dl":
            # Save ordinary section text before processing parameters.
            save_current_section()

            parent_heading = current_heading
            children = element.find_all(
                ["dt", "dd"],
                recursive=False,
            )

            current_term = None
            descriptions = []

            def save_definition():
                if not current_term:
                    return

                definition = normalize_text(
                    "\n\n".join(descriptions)
                )

                if not definition:
                    return

                section_name = (
                    f"{parent_heading} > {current_term}"
                )

                sections.append(
                    Document(
                        page_content=(
                            f"{section_name}\n\n{definition}"
                        ),
                        metadata={
                            "source": url,
                            "title": title,
                            "section": section_name,
                        },
                    )
                )

            for child in children:
                text = (
                    clean_heading(child.get_text(" ", strip=True))
                    if child.name == "dt"
                    else normalize_text(child.get_text(" ", strip=True))
                )

                if child.name == "dt":
                    save_definition()
                    current_term = text
                    descriptions = []

                elif child.name == "dd" and current_term:
                    if text:
                        descriptions.append(text)

            save_definition()
            continue

        text = normalize_text(
            element.get_text(" ", strip=True)
        )

        if not text:
            continue

        if text.lower() in {
            "prev",
            "previous",
            "next",
            "up",
            "home",
            "table of contents",
            "submit correction",
        }:
            continue

        current_parts.append(text)

    save_current_section()
    return sections

def load_documents():
    """Fetch all pages and extract heading-aware sections."""
    documents = []

    for url in URLS:
        print(f"Loading: {url}")

        try:
            root, title = fetch_page(url)
            sections = extract_sections(root, url, title)

            if not sections:
                print("  WARNING: No usable sections extracted.")
                continue

            documents.extend(sections)
            print(f"  Sections extracted: {len(sections)}")

        except (requests.RequestException, ValueError) as error:
            print(f"  ERROR: {error}")

    if not documents:
        raise RuntimeError(
            "No documentation was loaded. Check the URLs and errors above."
        )

    return documents



def split_documents(documents):
    """Split sections while preserving section context in metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    prepared_chunks = []

    for chunk in chunks:
        content = normalize_text(chunk.page_content)
        section = chunk.metadata.get("section", "").strip()

        if not content:
            continue

        # Remove only fragments that consist entirely of the section heading.
        if section and content == section:
            continue

        chunk.page_content = content
        prepared_chunks.append(chunk)

    return prepared_chunks

def save_chunks(chunks):
    """Write one JSON record per line."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            source = chunk.metadata["source"]
            section = chunk.metadata.get("section", "")

            # Content-based ID: identical source/section/content gives
            # the same ID, regardless of its position in the output.
            id_input = (
                f"{source}|{section}|{chunk.page_content}"
            )

            chunk_id = hashlib.sha256(
                id_input.encode("utf-8")
            ).hexdigest()

            record = {
                "chunk_id": chunk_id,
                "content": chunk.page_content,
                "metadata": chunk.metadata,
            }

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def report_quality(chunks):
    """Check basic content and metadata quality."""
    print("\n--- Chunk quality report ---")
    print(f"Total chunks: {len(chunks)}")

    if not chunks:
        print("No chunks available to inspect.")
        return

    lengths = [len(chunk.page_content) for chunk in chunks]

    print(f"Shortest chunk: {min(lengths)} characters")
    print(f"Longest chunk:  {max(lengths)} characters")
    print(
        f"Average length: "
        f"{sum(lengths) / len(lengths):.0f} characters"
    )

    # 1. Common website navigation and footer text
    navigation_phrases = [
        "home | download this documentation",
        "project versions",
        "submit correction",
        "code of conduct",
        "copyright ©",
        "table of contents",
        "on this page:",
        "previous:",
        "next:",
        "up: home",
        "report a documentation issue",
    ]

    noisy_chunks = []

    for chunk in chunks:
        content = chunk.page_content.lower()

        matches = [
            phrase
            for phrase in navigation_phrases
            if phrase in content
        ]

        if matches:
            noisy_chunks.append((chunk, matches))

    print(
        "Chunks containing navigation/footer phrases:",
        len(noisy_chunks),
    )

    for chunk, matches in noisy_chunks[:5]:
        print(
            f"  - {chunk.metadata.get('section', 'Unknown section')}: "
            f"{matches}"
        )

    # 2. Common signs of character-encoding corruption
    malformed_patterns = [
        "â€™",
        "â€œ",
        "â€",
        "â€“",
        "â€”",
        "Ã",
        "\ufffd",
    ]

    malformed_chunks = [
        chunk
        for chunk in chunks
        if any(
            pattern in chunk.page_content
            for pattern in malformed_patterns
        )
    ]

    print(
        "Chunks with possible encoding corruption:",
        len(malformed_chunks),
    )

    for chunk in malformed_chunks[:5]:
        print(
            f"  - {chunk.metadata.get('section', 'Unknown section')}"
        )

    # 3. Required metadata
    required_fields = ["source", "title", "section"]
    missing_metadata = []

    for index, chunk in enumerate(chunks):
        missing = [
            field
            for field in required_fields
            if not chunk.metadata.get(field)
        ]

        if missing:
            missing_metadata.append((index, missing))

    print("Chunks with missing metadata:", len(missing_metadata))

    for index, missing in missing_metadata[:5]:
        print(f"  - Chunk {index}: missing {missing}")

    # 4. Short fragments: report them rather than automatically
    # discarding them, because short definitions may still be useful.
    short_chunks = [
        chunk for chunk in chunks
        if len(chunk.page_content.strip()) < 200
    ]

    print("Chunks shorter than 200 characters:", len(short_chunks))

    for chunk in short_chunks[:5]:
        print(
            f"  - {chunk.metadata.get('section', 'Unknown section')}: "
            f"{len(chunk.page_content)} characters"
        )
        print(f"    {chunk.page_content[:150]}")

    # 5. Inspect representative chunks
    print("\n--- Three sample chunks ---")

    sample_indexes = sorted({
        0,
        len(chunks) // 2,
        len(chunks) - 1,
    })

    for index in sample_indexes:
        chunk = chunks[index]

        print(f"\nChunk {index + 1}")
        print(f"Section: {chunk.metadata.get('section', 'Unknown')}")
        print(f"Source: {chunk.metadata.get('source', 'MISSING')}")
        print(chunk.page_content[:500])

def diagnose_spacing():
    """Compare extracted sections against the resulting chunks."""
    sections = load_documents()
    chunks = split_documents(sections)

    terms = [
        "codeto",
        "code to",
        "notrelated",
        "aboutthe",
        "will beset",
        "purposes(e.g.",
        "fromthe",
        "connectionpool",
        "makeuse",
        "not related",
        "about the",
        "will be set",
    ]

    print("\n--- SPACING DIAGNOSTIC ---")

    for term in terms:
        section_matches = [
            doc for doc in sections
            if term.lower() in doc.page_content.lower()
        ]
        chunk_matches = [
            doc for doc in chunks
            if term.lower() in doc.page_content.lower()
        ]

        print(f"\nTerm: {term!r}")
        print(f"  In extracted sections: {len(section_matches)}")
        print(f"  In resulting chunks:   {len(chunk_matches)}")

        for doc in (section_matches or chunk_matches)[:2]:
            content = doc.page_content
            pos = content.lower().find(term.lower())
            print("  Context:", content[max(0, pos - 100):pos + len(term) + 120])


def main():
    sections = load_documents()

    suspicious_terms = [
        "notrelated",
        "aboutthe context",
        "multiplePostgreSQL",
        "eachnode",
        "Youmay",
        "scopelist",
    ]

    print("\n--- SPACING DIAGNOSTIC: EXTRACTED SECTIONS ---")

    for term in suspicious_terms:
        matches = [
            doc.page_content
            for doc in sections
            if term.lower() in doc.page_content.lower()
        ]

        print(f"\n{term!r}: {len(matches)} matches")

        for content in matches[:2]:
            position = content.lower().find(term.lower())
            start = max(0, position - 70)
            end = min(len(content), position + len(term) + 70)
            print("  ", repr(content[start:end]))

    print(f"\nTotal sections extracted: {len(sections)}")

    chunks = split_documents(sections)
    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        raise RuntimeError(
            "No chunks remain after cleaning and splitting."
        )

    save_chunks(chunks)
    print(f"Saved to: {OUTPUT_PATH}")

    report_quality(chunks)


if __name__ == "__main__":
    main()