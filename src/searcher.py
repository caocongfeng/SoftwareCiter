"""Web Searcher — enriches software mentions with metadata via DuckDuckGo search."""

import json
import logging

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.models import SoftwareMention, SoftwareMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """You are an expert research librarian specializing in software metadata.
Given search results about a software tool, extract structured metadata for building a proper citation.

Return a JSON object with these fields:
- name: The software name AS USED IN THE PAPER (keep the original name, do NOT change it to the full official name)
- version: The specific version (use the version from the paper if provided)
- authors: Author(s) or main contributors (format: "Last, F., Last2, F2.")
- year: Year of the relevant release or publication
- publisher: Publisher or hosting platform (e.g., "Zenodo", "CRAN", "PyPI", "GitHub")
- doi: DOI if available (format: "https://doi.org/10.xxxx/xxxxx")
- url: Official URL or repository link
- license: Software license if found (e.g., "MIT", "BSD-3-Clause")
- description: Brief one-sentence description of what the software does

CRITICAL RULES:
1. KEEP the original software name from the paper — do NOT rename "SPSS" to "IBM SPSS Statistics" or "Excel" to "Microsoft Excel"
2. KEEP the version from the paper context — do NOT replace it with the latest version
3. Prefer DOIs from Zenodo, JOSS, or other archival services
4. If no DOI exists, use the most persistent URL available
5. For authors, prioritize the CITATION.cff or official citation guidance
6. Always return valid JSON, even if some fields are empty strings

EVIDENCE-AWARE VALIDATION (VERY IMPORTANT):
7. The "Context from paper" tells you HOW the software was mentioned in the paper.
   Use it to validate that search results refer to the SAME software in the SAME context.
8. If the evidence sentence mentions MULTIPLE software tools together, make sure you ONLY
   assign metadata that belongs to the TARGET software, not to neighboring software.
   For example:
   - Evidence: "SPSS 15.0 for Windows" → if target is "Windows", version should be EMPTY (15.0 belongs to SPSS)
   - Evidence: "GraphPad Prism version 6.0f for Mac OS X, (GraphPad Software, ...)"
     → if target is "Mac OS X", do NOT assign GraphPad as author or www.graphpad.com as URL
9. If the evidence sentence clearly shows the target software is just a platform/OS mentioned
   alongside the actual analysis software, the metadata should reflect the TARGET software only.
10. If search results don't clearly match the target software, leave fields empty rather than
    filling them with wrong information."""

SYNTHESIS_USER_TEMPLATE = """Software mentioned in paper:
- Name: {name}
- Version from paper: {version}
- Publisher/Author from paper: {publisher}
- URL from paper: {url}
- Context from paper: "{context}"

Search results:
{search_results}

Extract the metadata as a JSON object."""


# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------

def search_software_metadata(
    mention: SoftwareMention,
    llm: ChatOpenAI,
    max_results: int = 5,
    llm_logger=None,
    pub_id: str = "",
) -> SoftwareMetadata:
    """Search the web for software metadata and synthesize results.

    Uses DuckDuckGo Search (free, no API key) to find information
    about a software tool, then uses an LLM to extract structured metadata.
    Retries up to 3 times with simplified queries if no results found.

    Args:
        mention: The software mention from the paper.
        llm: LangChain ChatOpenAI instance.
        max_results: Maximum number of search results to fetch.
        llm_logger: Optional LLMLogger for recording LLM interactions.
        pub_id: Publication ID for logging.

    Returns:
        Enriched SoftwareMetadata.
    """
    search_tool = DuckDuckGoSearchResults(
        max_results=max_results,
        output_format="list",
    )

    # Build multiple targeted queries
    queries = _build_search_queries(mention)
    all_results: list[str] = []

    for query in queries:
        logger.info(f"Searching: {query}")
        try:
            results = search_tool.invoke(query)
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict):
                        snippet = f"Title: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}\nLink: {r.get('link', '')}"
                        all_results.append(snippet)
                    else:
                        all_results.append(str(r))
            elif isinstance(results, str):
                all_results.append(results)
        except Exception as e:
            logger.warning(f"Search failed for query '{query}': {e}")

    # Retry up to 3 times with simplified queries if no results
    if not all_results:
        retry_queries = [
            f"{mention.name} software",
            f"{mention.name} official website",
            f"{mention.name} download",
        ]
        for attempt, retry_query in enumerate(retry_queries, 1):
            logger.info(f"Retry {attempt}/3: {retry_query}")
            try:
                results = search_tool.invoke(retry_query)
                if isinstance(results, list):
                    for r in results:
                        if isinstance(r, dict):
                            snippet = f"Title: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}\nLink: {r.get('link', '')}"
                            all_results.append(snippet)
                        else:
                            all_results.append(str(r))
                elif isinstance(results, str):
                    all_results.append(results)
                if all_results:
                    logger.info(f"Retry {attempt}/3 succeeded for {mention.name}")
                    break
            except Exception as e:
                logger.warning(f"Retry {attempt}/3 failed for '{retry_query}': {e}")

    if not all_results:
        logger.warning(f"No search results found for {mention.name} after 3 retries")
        return SoftwareMetadata(
            name=mention.name,
            version=mention.version,
            authors=mention.publisher,
            url=mention.url,
        )

    # Combine results and send to LLM for synthesis
    combined_results = "\n\n---\n\n".join(all_results[:15])  # Limit to avoid token overflow
    metadata = _synthesize_metadata(mention, combined_results, llm, llm_logger=llm_logger, pub_id=pub_id)

    # Log enriched fields
    logger.info(f"{pub_id}/{mention.name}: Search enriched fields: "
                f"version={metadata.version}, authors={metadata.authors}, "
                f"year={metadata.year}, url={metadata.url}, doi={metadata.doi}")

    return metadata


def _build_search_queries(mention: SoftwareMention) -> list[str]:
    """Build targeted search queries using name AND context clues."""
    name = mention.name
    context = mention.context or ""
    queries = []

    # 1. Detect domain / type hints from context
    domain_hint = _extract_domain_hint(name, context)
    type_label = _infer_type_label(name, context)

    # 2. Primary: software + citation/DOI (with domain hint if available)
    if domain_hint:
        queries.append(f"{name} {domain_hint} software citation DOI")
    queries.append(f"{name} software citation DOI")

    # 3. Type-labeled search (e.g., "MIRA software", "C++ language", "Windows system")
    if type_label and type_label != "software":
        queries.append(f"{name} {type_label}")

    # 4. Package manager specific (from context)
    if _is_r_package(name, context):
        queries.append(f"{name} CRAN R package")
    elif _is_python_package(name, context):
        queries.append(f"{name} PyPI Python package")
    elif _is_bioinformatics(name, context):
        queries.append(f"{name} bioinformatics tool bioconductor")

    # 5. Official site / repository
    queries.append(f"{name} software official site GitHub repository")

    # 6. Version-specific
    if mention.version:
        queries.append(f"{name} version {mention.version} release Zenodo")

    # 7. CITATION.cff
    queries.append(f"{name} software CITATION.cff how to cite")

    return queries


def _extract_domain_hint(name: str, context: str) -> str:
    """Extract domain keywords from context to disambiguate searches."""
    context_lower = context.lower()

    # Bioinformatics keywords
    bio_kw = ["genome", "assembly", "sequencing", "alignment", "phylogen",
              "variant", "snp", "rna", "dna", "protein", "blast",
              "bioinformatics", "molecular", "metagenom", "transcriptom",
              "proteom", "mapping", "reads", "contigs", "scaffold"]
    if any(kw in context_lower for kw in bio_kw):
        return "bioinformatics"

    # Statistics keywords
    stat_kw = ["statistical", "regression", "anova", "p-value", "p value",
               "chi-square", "survival analysis", "mixed model", "logistic",
               "t-test", "correlation", "confidence interval"]
    if any(kw in context_lower for kw in stat_kw):
        return "statistics"

    # Machine learning
    ml_kw = ["machine learning", "deep learning", "neural network",
             "classification", "training", "model training", "inference"]
    if any(kw in context_lower for kw in ml_kw):
        return "machine learning"

    # Image processing
    img_kw = ["image", "microscopy", "fluorescence", "photograph",
              "pixel", "segmentation"]
    if any(kw in context_lower for kw in img_kw):
        return "image analysis"

    return ""


def _infer_type_label(name: str, context: str) -> str:
    """Infer a type label for the software to improve search specificity.

    Returns labels like 'language', 'system', 'software', 'R package', etc.
    """
    name_lower = name.lower().strip()
    context_lower = context.lower()

    # Programming languages
    languages = {"c", "c++", "c#", "java", "python", "perl", "ruby",
                 "fortran", "matlab", "julia", "scala", "go", "rust",
                 "javascript", "php", "swift", "kotlin", "r",
                 "visual basic", "objective-c", "lua", "haskell"}
    if name_lower in languages:
        return "programming language"

    # Operating systems
    os_names = {"windows", "linux", "macos", "mac os x", "ubuntu",
                "centos", "debian", "unix", "red hat", "fedora",
                "freebsd", "android", "ios"}
    if name_lower in os_names:
        return "operating system"

    # R package indicators
    if _is_r_package(name, context):
        return "R package"

    # Python package indicators
    if _is_python_package(name, context):
        return "Python package"

    return "software"


def _is_r_package(name: str, context: str) -> bool:
    """Check if software is likely an R package from context."""
    context_lower = context.lower()
    return ("r package" in context_lower or
            "cran" in context_lower or
            "bioconductor" in context_lower or
            f"library({name.lower()})" in context_lower or
            f"library( {name.lower()}" in context_lower)


def _is_python_package(name: str, context: str) -> bool:
    """Check if software is likely a Python package from context."""
    context_lower = context.lower()
    return ("python package" in context_lower or
            "python library" in context_lower or
            "pypi" in context_lower or
            f"pip install {name.lower()}" in context_lower or
            f"import {name.lower()}" in context_lower)


def _is_bioinformatics(name: str, context: str) -> bool:
    """Check if software is in a bioinformatics context."""
    context_lower = context.lower()
    return any(kw in context_lower for kw in
               ["genome", "sequencing", "assembly", "alignment",
                "variant", "phylogen", "metagenom", "bioinformatics",
                "transcriptom", "proteom"])


def _synthesize_metadata(
    mention: SoftwareMention,
    search_results: str,
    llm: ChatOpenAI,
    llm_logger=None,
    pub_id: str = "",
) -> SoftwareMetadata:
    """Use LLM to synthesize search results into structured metadata."""
    messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=SYNTHESIS_USER_TEMPLATE.format(
            name=mention.name,
            version=mention.version,
            publisher=mention.publisher,
            url=mention.url,
            context=mention.context,
            search_results=search_results,
        )),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    # Log LLM interaction
    if llm_logger:
        llm_logger.log("searcher", "synthesis", pub_id, mention.name, messages, content)

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        data = json.loads(content)

        # CONSISTENCY: Preserve original_mention values — they come from the paper
        # Only use LLM-enriched values for fields the paper didn't provide
        enriched_name = mention.name  # Always keep original name
        enriched_version = mention.version or data.get("version", "")  # Paper version takes priority
        enriched_url = mention.url or data.get("url", "")  # Paper URL takes priority
        enriched_publisher = mention.publisher or data.get("publisher", "")

        return SoftwareMetadata(
            name=enriched_name,
            version=enriched_version,
            authors=str(data.get("authors", mention.publisher or "")),
            year=str(data.get("year", "")),
            publisher=enriched_publisher,
            doi=str(data.get("doi", "")),
            url=enriched_url,
            license=str(data.get("license", "")),
            description=str(data.get("description", "")),
        )
    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM synthesis as JSON:\n{content[:500]}")
        return SoftwareMetadata(
            name=mention.name,
            version=mention.version,
            authors=mention.publisher,
            url=mention.url,
        )
