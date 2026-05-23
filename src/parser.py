"""Document parser — reads academic papers in XML, TXT, or other formats.

Supports two parsing modes:
  1. LLM-based: Strips XML tags, then uses GPT to structure text into sections.
     Handles all XML variants (DOCTYPE, <?xml>, fragments) robustly.
  2. Rule-based fallback: Uses lxml for structured XML or heuristics for TXT.
"""

import hashlib
import json
import logging
import os
import re
from html import unescape
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from lxml import etree

from src.models import PaperInfo, PaperSection

logger = logging.getLogger(__name__)

# Default cache directory for parsed papers
DEFAULT_CACHE_DIR = Path(__file__).parent / "parser_cache"


# ---------------------------------------------------------------------------
# XML tag stripping (Stage 1 — deterministic, handles any XML format)
# ---------------------------------------------------------------------------

# Tags whose content should be removed entirely (not just the tags)
_REMOVE_CONTENT_TAGS = re.compile(
    r"<\s*(script|style|xref|ext-link|inline-formula|disp-formula|"
    r"table-wrap-foot|supplementary-material|graphic|media|object-id)[^>]*>.*?</\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)

# References section — remove to avoid bibliography software mentions
_REFERENCES_PATTERN = re.compile(
    r"<\s*ref-list[^>]*>.*?</\s*ref-list\s*>",
    re.DOTALL | re.IGNORECASE,
)

# All remaining XML/HTML tags
_TAG_RE = re.compile(r"<[^>]+>")

# XML/HTML entities
_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;")

# Multiple whitespace / blank lines
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def strip_xml_tags(xml_str: str, keep_references: bool = False) -> str:
    """Convert XML/HTML to clean plain text by stripping all tags.

    This is a fast, deterministic Stage-1 transform that handles any XML
    format (DOCTYPE, <?xml>, fragments, malformed) without needing a
    proper XML parser.

    Args:
        xml_str: Raw XML string.
        keep_references: If True, keep the references section.

    Returns:
        Clean plain text with tags removed.
    """
    text = xml_str

    # Remove XML declaration and DOCTYPE
    text = re.sub(r"<\?xml[^?]*\?>", "", text)
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text)
    # Remove CDATA sections (keep content)
    text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.DOTALL)
    # Remove XML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove reference list before stripping tags (to avoid FP software extraction)
    if not keep_references:
        text = _REFERENCES_PATTERN.sub("", text)

    # Remove tags whose content is not useful
    text = _REMOVE_CONTENT_TAGS.sub("", text)

    # Convert common block elements to newlines for readability
    text = re.sub(r"</?(p|div|sec|section|h[1-6]|title|abstract|body|article|front|back|tr|li|dd|dt)\b[^>]*>", "\n", text, flags=re.IGNORECASE)

    # Strip all remaining tags
    text = _TAG_RE.sub("", text)

    # Decode HTML/XML entities
    text = unescape(text)
    # Catch remaining numeric entities
    text = _ENTITY_RE.sub(" ", text)

    # Normalize whitespace
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# LLM-based structuring (Stage 2 — uses GPT to identify sections)
# ---------------------------------------------------------------------------

_PARSER_SYSTEM_PROMPT = """You are an expert at parsing academic papers. Given the plain text of an academic paper, extract its structure.

Return a JSON object with:
- "title": The paper title
- "abstract": The abstract text (empty string if not found)
- "sections": Array of {"title": "Section Name", "content": "Section text..."} objects

Rules:
1. Identify sections by headings (e.g., "Introduction", "Methods", "Results", "Discussion", "Conclusion")
2. Do NOT include the References/Bibliography section — skip it entirely
3. Do NOT include Supplementary Materials
4. Keep footnotes and figure/table captions if they appear in the main text
5. Preserve the original text content faithfully — do not summarize or rewrite
6. If you cannot identify clear sections, put all text in a single section titled "Body"
7. Return valid JSON only"""

_PARSER_USER_TEMPLATE = """Parse the following academic paper text into structured sections.
Return ONLY a JSON object with "title", "abstract", and "sections" fields.

--- Paper Text ---
{text}
--- End of Text ---"""


def parse_with_llm(
    text: str,
    llm: ChatOpenAI,
    title: str = "",
    abstract: str = "",
) -> PaperInfo:
    """Use GPT to structure clean text into a PaperInfo object.

    Args:
        text: Clean plain text (already tag-stripped).
        llm: LangChain ChatOpenAI instance.
        title: Pre-extracted title (overrides LLM extraction if provided).
        abstract: Pre-extracted abstract (overrides LLM extraction if provided).

    Returns:
        PaperInfo with structured sections.
    """
    # For very long papers, we don't need to send everything to the structuring LLM.
    # We just need section boundaries. Truncate to ~100K chars for the parser.
    truncated = text[:100_000] if len(text) > 100_000 else text

    try:
        messages = [
            SystemMessage(content=_PARSER_SYSTEM_PROMPT),
            HumanMessage(content=_PARSER_USER_TEMPLATE.format(text=truncated)),
        ]

        response = llm.invoke(messages)
        content = response.content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        data = json.loads(content)

        # Use pre-extracted values if provided
        parsed_title = title or data.get("title", "")
        parsed_abstract = abstract or data.get("abstract", "")

        sections = []
        for sec in data.get("sections", []):
            sec_title = sec.get("title", "")
            sec_content = sec.get("content", "")
            # Skip reference sections the LLM might have included
            if sec_title.lower().strip() in ("references", "bibliography", "works cited", "literature cited"):
                continue
            if sec_content.strip():
                sections.append(PaperSection(title=sec_title, content=sec_content))

        # Fallback: if LLM returned no useful sections, use the full text
        if not sections:
            sections = [PaperSection(title="Body", content=text)]

        return PaperInfo(
            title=parsed_title,
            authors="",
            abstract=parsed_abstract,
            sections=sections,
            source_file="",
            file_format="llm_parsed",
        )

    except Exception as e:
        logger.warning(f"LLM parsing failed: {e}. Falling back to simple text split.")
        return _simple_text_split(text, title=title, abstract=abstract)


def _simple_text_split(
    text: str, title: str = "", abstract: str = ""
) -> PaperInfo:
    """Fallback: split text into sections using simple heuristics."""
    sections = []
    current_title = ""
    current_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if _is_section_header(stripped):
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append(PaperSection(title=current_title, content=content))
            current_title = stripped.rstrip(":")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(PaperSection(title=current_title, content=content))

    # Filter out reference sections
    sections = [s for s in sections
                if s.title.lower().strip() not in ("references", "bibliography", "works cited", "literature cited")]

    if not sections:
        sections = [PaperSection(title="Body", content=text)]

    # Try to extract abstract if not provided
    if not abstract:
        for sec in sections:
            if sec.title.lower() in ("abstract", "summary"):
                abstract = sec.content
                break

    return PaperInfo(
        title=title or (sections[0].title if sections else ""),
        authors="",
        abstract=abstract,
        sections=sections,
        source_file="",
        file_format="txt",
    )


# ---------------------------------------------------------------------------
# Unified entry points
# ---------------------------------------------------------------------------

def parse_xml_string(
    xml_str: str,
    llm: ChatOpenAI | None = None,
    pub_id: str = "",
    title: str = "",
    abstract: str = "",
) -> PaperInfo:
    """Parse an XML/text string into PaperInfo.

    Process:
      1. Strip XML tags (deterministic, handles any format)
      2. Structure text into sections (heuristic split; LLM optional)
      3. Fallback to title+abstract if fulltext is empty

    Args:
        xml_str: Raw XML/HTML string of the paper.
        llm: Optional LangChain ChatOpenAI instance for LLM structuring.
             If None, uses fast heuristic text splitting.
        pub_id: Publication identifier.
        title: Pre-extracted title (if available).
        abstract: Pre-extracted abstract (if available).

    Returns:
        PaperInfo with structured sections.
    """
    # Stage 1: Strip XML tags → clean text
    clean_text = ""
    if xml_str and xml_str.strip():
        clean_text = strip_xml_tags(xml_str, keep_references=False)

    # Fallback: use title + abstract if fulltext is empty/stripped to nothing
    if not clean_text.strip():
        fallback_parts = []
        if title and title.strip():
            fallback_parts.append(title.strip())
        if abstract and abstract.strip():
            fallback_parts.append(abstract.strip())

        if fallback_parts:
            logger.info(f"{pub_id}: Using title+abstract as fallback (no fulltext)")
            return PaperInfo(
                title=title,
                authors="",
                abstract=abstract,
                sections=[PaperSection(title="Body", content="\n\n".join(fallback_parts))],
                source_file=pub_id,
                file_format="fallback",
            )
        else:
            logger.warning(f"{pub_id}: No text content at all (empty fulltext, title, abstract)")
            return PaperInfo(
                title=title,
                authors="",
                abstract=abstract,
                sections=[],
                source_file=pub_id,
                file_format="empty",
            )

    # Stage 2: Structure text into sections
    if llm is not None:
        paper = parse_with_llm(clean_text, llm, title=title, abstract=abstract)
    else:
        paper = _simple_text_split(clean_text, title=title, abstract=abstract)

    paper.source_file = pub_id
    paper.file_format = "parsed"

    return paper


def parse_xml_string_cached(
    xml_str: str,
    llm: ChatOpenAI | None = None,
    pub_id: str = "",
    title: str = "",
    abstract: str = "",
    cache_dir: Path | str | None = None,
) -> PaperInfo:
    """Parse an XML/text string with local file caching.

    If the document was previously parsed and a cache file exists,
    the cached result is loaded directly without re-parsing.
    Otherwise, parses normally and saves the result to the cache.

    Args:
        xml_str: Raw XML/HTML string of the paper.
        llm: Optional LangChain ChatOpenAI instance for LLM structuring.
        pub_id: Publication identifier (used as cache filename if provided).
        title: Pre-extracted title.
        abstract: Pre-extracted abstract.
        cache_dir: Directory to store cache files. Defaults to src/parser_cache/.

    Returns:
        PaperInfo with structured sections.
    """
    cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)

    # Determine cache key: prefer pub_id, fallback to content hash
    if pub_id:
        safe_id = pub_id.replace("/", "_").replace("\\", "_")
        cache_file = cache_path / f"{safe_id}.json"
    else:
        content_hash = hashlib.md5(xml_str.encode("utf-8")).hexdigest()
        cache_file = cache_path / f"{content_hash}.json"

    # Check cache — reuse if available
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            paper = PaperInfo(**data)
            logger.info(f"Parser cache HIT: {pub_id or cache_file.stem}")
            return paper
        except Exception as e:
            logger.warning(f"Parser cache read failed ({cache_file}): {e}. Re-parsing.")

    # Cache miss — parse normally
    logger.info(f"Parser cache MISS: {pub_id or cache_file.stem}. Parsing...")
    paper = parse_xml_string(xml_str, llm=llm, pub_id=pub_id, title=title, abstract=abstract)

    # Save to cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(paper.model_dump(), f, ensure_ascii=False, indent=2)
        logger.debug(f"Parser cache saved: {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save parser cache ({cache_file}): {e}")

    return paper


def detect_format(file_path: str) -> str:
    """Detect the format of an academic paper file.

    Args:
        file_path: Path to the paper file.

    Returns:
        Format string: 'xml', 'txt', or 'unknown'.
    """
    ext = Path(file_path).suffix.lower()

    format_map = {
        ".xml": "xml",
        ".nxml": "xml",
        ".txt": "txt",
        ".text": "txt",
        ".md": "txt",
    }

    if ext in format_map:
        return format_map[ext]

    # Content sniffing as fallback
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            head = f.read(500)
        if head.strip().startswith("<?xml") or head.strip().startswith("<"):
            return "xml"
    except (UnicodeDecodeError, IOError):
        pass

    return "txt"


def parse_document(file_path: str, llm: ChatOpenAI | None = None) -> PaperInfo:
    """Parse an academic paper and return structured sections.

    If an LLM is provided and the file is XML, uses the LLM-based parser.
    Otherwise falls back to rule-based parsing.

    Args:
        file_path: Path to the paper file.
        llm: Optional LangChain ChatOpenAI instance for LLM-based parsing.

    Returns:
        PaperInfo with structured sections.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Paper file not found: {file_path}")

    fmt = detect_format(file_path)

    if fmt == "xml" and llm is not None:
        with open(file_path, "r", encoding="utf-8") as f:
            xml_str = f.read()
        paper = parse_xml_string(xml_str, llm, pub_id=file_path)
        paper.source_file = file_path
        return paper
    elif fmt == "xml":
        return _parse_xml_rulebased(file_path)
    elif fmt == "txt":
        return _parse_text(file_path)
    else:
        raise ValueError(f"Unsupported file format: {fmt}")


# ---------------------------------------------------------------------------
# Rule-based parsers (kept as fallback when no LLM is available)
# ---------------------------------------------------------------------------

def _parse_xml_rulebased(file_path: str) -> PaperInfo:
    """Parse a JATS/NLM XML academic paper using lxml (rule-based fallback)."""
    tree = etree.parse(file_path)
    root = tree.getroot()

    # Remove namespace prefixes
    for elem in root.iter():
        if isinstance(elem.tag, str) and "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    title = ""
    title_elem = root.find(".//article-title")
    if title_elem is not None:
        title = _get_text(title_elem)

    authors_parts = []
    for contrib in root.findall(".//contrib[@contrib-type='author']"):
        surname = contrib.findtext(".//surname", default="")
        given = contrib.findtext(".//given-names", default="")
        if surname:
            name = f"{given} {surname}".strip()
            authors_parts.append(name)
    authors = ", ".join(authors_parts)

    abstract = ""
    abstract_elem = root.find(".//abstract")
    if abstract_elem is not None:
        abstract = _get_text(abstract_elem)

    sections: list[PaperSection] = []
    body = root.find(".//body")
    if body is not None:
        for sec in body.findall(".//sec"):
            sec_title_elem = sec.find("title")
            sec_title = _get_text(sec_title_elem) if sec_title_elem is not None else ""
            sec_content = _get_text(sec)
            if sec_content.strip():
                sections.append(PaperSection(title=sec_title, content=sec_content))

        if not sections:
            body_text = _get_text(body)
            if body_text.strip():
                sections.append(PaperSection(title="Body", content=body_text))

    # Note: References section intentionally excluded to reduce false positives

    return PaperInfo(
        title=title,
        authors=authors,
        abstract=abstract,
        sections=sections,
        source_file=file_path,
        file_format="xml",
    )


def _parse_text(file_path: str) -> PaperInfo:
    """Parse a plain text academic paper."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")

    title = ""
    for line in lines:
        if line.strip():
            title = line.strip()
            break

    sections: list[PaperSection] = []
    current_title = ""
    current_content: list[str] = []

    for line in lines:
        stripped = line.strip()
        if _is_section_header(stripped):
            if current_content:
                sections.append(
                    PaperSection(title=current_title, content="\n".join(current_content))
                )
            current_title = stripped.rstrip(":")
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append(
            PaperSection(title=current_title, content="\n".join(current_content))
        )

    abstract = ""
    for sec in sections:
        if sec.title.lower() in ("abstract", "summary"):
            abstract = sec.content
            break

    return PaperInfo(
        title=title,
        authors="",
        abstract=abstract,
        sections=sections,
        source_file=file_path,
        file_format="txt",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_section_header(line: str) -> bool:
    """Heuristic check if a line is a section header."""
    if not line:
        return False
    if line.isupper() and len(line) < 80:
        return True
    if line.endswith(":") and len(line) < 80 and not line.startswith(" "):
        return True
    if len(line) < 80:
        if re.match(r"^\d+\.?\s+[A-Z]", line):
            return True
    return False


def _get_text(element) -> str:
    """Recursively extract all text from an lxml element."""
    return "".join(element.itertext()).strip()
