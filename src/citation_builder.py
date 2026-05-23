"""Citation Builder — constructs FORCE11-compliant software citations."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.models import SoftwareCitation, SoftwareMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CITATION_SYSTEM_PROMPT = """You are an expert in academic citation formatting, specializing in software citations following the FORCE11 Software Citation Principles.

The FORCE11 Software Citation Principles require:
1. **Importance**: Software is a legitimate, citable research product
2. **Credit and Attribution**: Proper credit to all contributors
3. **Unique Identification**: Use persistent identifiers (DOI preferred)
4. **Persistence**: Identifiers remain accessible over time
5. **Accessibility**: Citation enables access to software and documentation
6. **Specificity**: Cite the exact version used

Standard citation format:
Author(s). (Year). Title (Version X.Y.Z) [Software]. Publisher/Repository. DOI-or-URL

Examples:
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library (Version 2.1.0) [Software]. Meta AI. https://doi.org/10.5281/zenodo.2530456
- Pedregosa, F., et al. (2011). scikit-learn: Machine Learning in Python (Version 1.3.2) [Software]. https://scikit-learn.org/

Rules:
1. Always include [Software] type indicator
2. Use the specific version from the paper context
3. Prefer DOI over regular URLs when available
4. If authors are unknown, use the project/organization name
5. If year is unknown, use "n.d." (no date)
6. Note any missing information in your completeness assessment"""

CITATION_USER_TEMPLATE = """Build a FORCE11-compliant software citation from this metadata:

- Name: {name}
- Version: {version}
- Authors: {authors}
- Year: {year}
- Publisher: {publisher}
- DOI: {doi}
- URL: {url}
- License: {license}
- Description: {description}

Return ONLY the citation string (nothing else). Follow the format:
Author(s). (Year). Title (Version X.Y.Z) [Software]. Publisher. DOI-or-URL"""


# ---------------------------------------------------------------------------
# Citation building
# ---------------------------------------------------------------------------

def build_citation(
    metadata: SoftwareMetadata,
    llm: ChatOpenAI,
    llm_logger=None,
    pub_id: str = "",
) -> SoftwareCitation:
    """Build a FORCE11-compliant software citation from enriched metadata.

    Uses the LLM to format and polish the citation, then validates
    completeness against the six principles.

    Args:
        metadata: Enriched software metadata.
        llm: LangChain ChatOpenAI instance.
        llm_logger: Optional LLMLogger for recording LLM interactions.
        pub_id: Publication ID for logging.

    Returns:
        SoftwareCitation with formatted text and completeness notes.
    """
    # Generate citation text via LLM
    messages = [
        SystemMessage(content=CITATION_SYSTEM_PROMPT),
        HumanMessage(content=CITATION_USER_TEMPLATE.format(
            name=metadata.name,
            version=metadata.version,
            authors=metadata.authors,
            year=metadata.year,
            publisher=metadata.publisher,
            doi=metadata.doi,
            url=metadata.url,
            license=metadata.license,
            description=metadata.description,
        )),
    ]

    response = llm.invoke(messages)
    citation_text = response.content.strip()

    # Log LLM interaction
    if llm_logger:
        llm_logger.log("builder", "build_citation", pub_id, metadata.name, messages, citation_text)

    # Remove any surrounding quotes the LLM might add
    if citation_text.startswith('"') and citation_text.endswith('"'):
        citation_text = citation_text[1:-1]

    # Validate completeness
    completeness_notes = _assess_completeness(metadata)

    citation = SoftwareCitation(
        software_name=metadata.name,
        citation_text=citation_text,
        metadata=metadata,
        completeness_notes=completeness_notes,
    )

    logger.info(f"Built citation for {metadata.name}: {citation_text}")
    return citation


def build_citation_simple(metadata: SoftwareMetadata) -> SoftwareCitation:
    """Build a citation using template rules only (no LLM needed).

    Useful as a fallback or for testing.

    Args:
        metadata: Enriched software metadata.

    Returns:
        SoftwareCitation with template-formatted text.
    """
    # Format authors
    authors = metadata.authors if metadata.authors else metadata.name
    year = metadata.year if metadata.year else "n.d."
    version_str = f" (Version {metadata.version})" if metadata.version else ""
    publisher_str = f" {metadata.publisher}." if metadata.publisher else ""
    identifier = metadata.doi if metadata.doi else metadata.url

    citation_text = f"{authors}. ({year}). {metadata.name}{version_str} [Software].{publisher_str}"
    if identifier:
        citation_text += f" {identifier}"

    completeness_notes = _assess_completeness(metadata)

    return SoftwareCitation(
        software_name=metadata.name,
        citation_text=citation_text,
        metadata=metadata,
        completeness_notes=completeness_notes,
    )


def _assess_completeness(metadata: SoftwareMetadata) -> str:
    """Assess citation completeness against FORCE11 principles."""
    missing = []

    if not metadata.authors:
        missing.append("authors (Credit & Attribution)")
    if not metadata.version:
        missing.append("version (Specificity)")
    if not metadata.doi:
        missing.append("DOI (Unique Identification & Persistence)")
    if not metadata.year:
        missing.append("year (Accessibility)")
    if not metadata.url and not metadata.doi:
        missing.append("URL or DOI (Accessibility)")
    if not metadata.publisher:
        missing.append("publisher (Accessibility)")

    if missing:
        return "Missing fields: " + "; ".join(missing)
    return "All FORCE11 required fields are present."
