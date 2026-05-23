"""Verifier — validates and corrects software citations.

Performs 8 automated checks on generated citations:
  1. Placeholder detection ("unknown", "N/A", empty)
  2. DOI format validation (10.xxxx/...)
  3. DOI resolution check (does it actually resolve via doi.org?)
  4. DOI relevance check (points to software, not a paper about it?)
  5. URL relevance check (points to software, not papers)
  6. Year anomaly detection (future, pre-1950)
  7. Author format validation (array → citation string)
  8. Version format validation (no embedded placeholders)
  9. LLM cross-field consistency (DOI/URL/authors belong to the software)

Ground-truth protection: original_data fields from the source dataset
are treated as immutable and never overwritten during correction.
"""

import ast
import json
import logging
import re
import time
from dataclasses import dataclass, field

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.models import SoftwareCitation, SoftwareMetadata

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

PLACEHOLDER_PATTERNS = [
    r"^unknown$", r"^n\.?d\.?$", r"^n/a$", r"^none$", r"^not available$",
    r"^not found$", r"^unavailable$", r"^unspecified$", r"^\?+$",
    r"^tbd$", r"^to be determined$",
]
_PLACEHOLDER_RE = re.compile("|".join(PLACEHOLDER_PATTERNS), re.IGNORECASE)

CURRENT_YEAR = 2026

SKIP_NAMES = {
    "c", "c++", "java", "perl", "python", "r", "ruby", "fortran",
    "javascript", "php", "c#", "visual basic", "matlab",
    "unix", "linux", "windows", "macosx", "mac os x", "macos",
}

_INSTRUCTION_KEYWORDS = [
    "check if", "provide a", "identify the", "look up",
    "search for", "find the", "verify", "determine",
    "should be", "not found", "no correction",
]


# ──────────────────────────────────────────────────────────────────────
# Data classes for verification results
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FieldIssue:
    """A single issue detected in a citation field."""
    field: str
    issue_type: str      # e.g., "empty_or_placeholder", "future_year", "array_format"
    detail: str = ""     # human-readable description


@dataclass
class VerificationResult:
    """Result of verifying a single citation."""
    software_name: str
    is_correct: bool
    issues: list[FieldIssue] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)   # field → new value
    protected_fields: set[str] = field(default_factory=set)      # fields from original_data


@dataclass
class VerificationStats:
    """Aggregated statistics across all verified citations."""
    total: int = 0
    already_correct: int = 0
    with_issues: int = 0
    corrected: int = 0
    field_quality: dict = field(default_factory=lambda: {
        f: {"valid": 0, "invalid": 0, "empty": 0, "corrected": 0}
        for f in ("doi", "url", "year", "authors", "version")
    })
    issue_counts: dict[str, int] = field(default_factory=dict)
    correction_counts: dict[str, int] = field(default_factory=dict)
    original_restored: int = 0

    def to_dict(self) -> dict:
        return {
            "total_citations": self.total,
            "citations_already_correct": self.already_correct,
            "citations_with_issues": self.with_issues,
            "citations_corrected": self.corrected,
            "field_quality": self.field_quality,
            "issue_types": self.issue_counts,
            "corrections_made": self.correction_counts,
            "original_restored": self.original_restored,
        }


# ──────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────

def is_placeholder(value: str) -> bool:
    """Check if a value is a placeholder / empty / garbage."""
    if not value or not value.strip():
        return True
    return bool(_PLACEHOLDER_RE.match(value.strip()))


# Cache for DOI resolution results: doi -> (resolves: bool, resolved_url: str)
_doi_resolution_cache: dict[str, tuple[bool, str]] = {}


def _validate_doi_resolves(doi: str) -> tuple[bool, str]:
    """Check if a DOI resolves by making an HTTP HEAD request to doi.org.

    Args:
        doi: The DOI string (e.g., "10.5281/zenodo.2530456").

    Returns:
        Tuple of (resolves: bool, resolved_url: str).
        resolved_url is the final URL after redirects, or empty string if failed.
    """
    # Strip common prefixes
    clean_doi = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if clean_doi.lower().startswith(prefix.lower()):
            clean_doi = clean_doi[len(prefix):]
            break

    if clean_doi in _doi_resolution_cache:
        return _doi_resolution_cache[clean_doi]

    if not _HTTPX_AVAILABLE:
        logger.debug("httpx not available, skipping DOI resolution check")
        return True, ""  # Assume valid if we can't check

    doi_url = f"https://doi.org/{clean_doi}"
    try:
        with httpx.Client(follow_redirects=True, timeout=10.0) as client:
            resp = client.head(doi_url)
            resolves = resp.status_code < 400
            resolved_url = str(resp.url) if resolves else ""
            _doi_resolution_cache[clean_doi] = (resolves, resolved_url)
            time.sleep(0.3)  # Rate limit doi.org requests
            return resolves, resolved_url
    except Exception as e:
        logger.debug(f"DOI resolution check failed for {doi}: {e}")
        # On network error, don't penalize — assume valid
        _doi_resolution_cache[clean_doi] = (True, "")
        return True, ""


def _validate_doi_is_software(
    doi: str,
    software_name: str,
    llm: ChatOpenAI,
) -> list[FieldIssue]:
    """Check if a DOI points to the actual software (not a paper about it).

    Uses the resolved URL domain and path to make a judgment.

    Args:
        doi: The DOI string.
        software_name: Name of the software.
        llm: LangChain ChatOpenAI instance.

    Returns:
        List of FieldIssue if the DOI points to a paper, empty if OK.
    """
    resolves, resolved_url = _validate_doi_resolves(doi)
    if not resolves or not resolved_url:
        return []

    # Quick heuristic checks for known software DOI registries
    software_doi_patterns = [
        r"zenodo\.org",          # Zenodo software deposits
        r"joss\.theoj\.org",     # Journal of Open Source Software
        r"cran\.r-project",      # CRAN
        r"pypi\.org",            # PyPI
        r"bioconductor\.org",    # Bioconductor
        r"github\.com",          # GitHub
        r"gitlab\.com",          # GitLab
        r"npmjs\.com",           # npm
    ]
    for pat in software_doi_patterns:
        if re.search(pat, resolved_url, re.I):
            return []  # Likely a software DOI

    # Known paper DOI patterns
    paper_doi_patterns = [
        r"pubmed\.ncbi",
        r"ncbi\.nlm\.nih\.gov/pmc",
        r"scholar\.google",
        r"sciencedirect\.com",
        r"springer\.com/article",
        r"wiley\.com/.*/abstract",
        r"nature\.com/articles",
        r"plos\.org/.*article",
        r"journals\.",
    ]
    for pat in paper_doi_patterns:
        if re.search(pat, resolved_url, re.I):
            return [FieldIssue(
                "doi",
                "doi_is_paper_not_software",
                f"DOI resolves to a paper page: {resolved_url}",
            )]

    return []


def _validate_doi(doi: str) -> list[FieldIssue]:
    if not doi or is_placeholder(doi):
        return [FieldIssue("doi", "empty_or_placeholder")]
    if not re.search(r'10\.\d{4,}/[^\s]+', doi):
        return [FieldIssue("doi", "invalid_format", f"'{doi}' is not a valid DOI")]
    # Check if DOI actually resolves
    resolves, resolved_url = _validate_doi_resolves(doi)
    if not resolves:
        return [FieldIssue("doi", "doi_not_resolving", f"DOI '{doi}' does not resolve")]
    return []


def _validate_url(url: str, name: str) -> list[FieldIssue]:
    if not url or is_placeholder(url):
        return [FieldIssue("url", "empty_or_placeholder")]
    issues = []
    if not re.match(r'https?://', url) and not url.startswith("www."):
        issues.append(FieldIssue("url", "missing_protocol", url))
    for pat in (r'pubmed\.ncbi', r'scholar\.google', r'ncbi\.nlm\.nih\.gov/pmc'):
        if re.search(pat, url, re.I):
            issues.append(FieldIssue("url", "url_is_paper_not_software", url))
            break
    return issues


def _validate_year(year: str) -> list[FieldIssue]:
    if not year or is_placeholder(year):
        return [FieldIssue("year", "empty_or_placeholder")]
    m = re.search(r'\d{4}', str(year))
    if not m:
        return [FieldIssue("year", "not_a_valid_year", year)]
    y = int(m.group())
    issues = []
    if y > CURRENT_YEAR:
        issues.append(FieldIssue("year", "future_year", str(y)))
    if y < 1950:
        issues.append(FieldIssue("year", "suspiciously_old", str(y)))
    return issues


def _validate_authors(authors: str, name: str) -> list[FieldIssue]:
    if not authors or is_placeholder(authors):
        return [FieldIssue("authors", "empty_or_placeholder")]
    issues = []
    if authors.strip().startswith("[") and authors.strip().endswith("]"):
        issues.append(FieldIssue("authors", "array_format", "Python list instead of citation string"))
    if authors.strip().lower() == name.lower():
        issues.append(FieldIssue("authors", "author_is_software_name"))
    return issues


def _validate_version(version: str) -> list[FieldIssue]:
    if not version or is_placeholder(version):
        return [FieldIssue("version", "empty_or_placeholder")]
    if re.search(r'unknown|n/a|unspecified', version, re.I):
        return [FieldIssue("version", "contains_placeholder")]
    return []


def _fix_author_array(authors: str) -> str:
    """Convert ['Name1', 'Name2'] to citation-style string."""
    authors = authors.strip()
    if authors.startswith("[") and authors.endswith("]"):
        try:
            lst = ast.literal_eval(authors)
            if isinstance(lst, list):
                return ", ".join(str(a) for a in lst)
        except (ValueError, SyntaxError):
            return authors[1:-1].replace("'", "").replace('"', '')
    return authors


# ──────────────────────────────────────────────────────────────────────
# LLM consistency check
# ──────────────────────────────────────────────────────────────────────

_CONSISTENCY_PROMPT = """You are a software metadata verification expert. Check if the metadata is internally consistent and factually plausible.

Software: {name}
Authors: {authors}  |  Version: {version}  |  Year: {year}
Publisher: {publisher}  |  DOI: {doi}  |  URL: {url}
Context from paper: {context}

Check:
1. Are the authors the actual SOFTWARE DEVELOPERS (not paper authors who used it)?
2. Is the DOI for the software itself (not a paper about it)?
3. Does the URL point to the actual software?
4. Is the year/version plausible?

Return JSON:
{{"is_correct": true/false, "issues": ["..."], "corrections": {{"field": "actual_value"}}}}

CRITICAL: corrections must be FACTUAL values you know, NOT instructions.
Use lowercase field keys (authors, version, year, publisher, doi, url).
Return {{}} for corrections if unsure. JSON only."""


def _check_consistency(
    metadata: SoftwareMetadata,
    context: str,
    llm: ChatOpenAI,
) -> dict:
    """LLM-based cross-field consistency check."""
    try:
        resp = llm.invoke([
            SystemMessage(content="Verify software citation metadata. Return valid JSON only."),
            HumanMessage(content=_CONSISTENCY_PROMPT.format(
                name=metadata.name,
                authors=metadata.authors,
                version=metadata.version,
                year=metadata.year,
                publisher=metadata.publisher,
                doi=metadata.doi,
                url=metadata.url,
                context=(context or "")[:500],
            )),
        ])
        content = resp.content.strip()
        if content.startswith("```"):
            content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Consistency check failed: {e}")
        return {"is_correct": True, "issues": [], "corrections": {}}


# ──────────────────────────────────────────────────────────────────────
# Web re-search for correction
# ──────────────────────────────────────────────────────────────────────

_CORRECTION_PROMPT = """Software "{name}" has citation errors: {issues}

Current: authors={authors}, version={version}, year={year}, publisher={publisher}, doi={doi}, url={url}

Search results:
{search_results}

Provide corrected values as JSON ({{"field": "value"}}).
Rules: authors in "Last, F." format (software DEVELOPERS), DOI for the software (Zenodo/JOSS preferred), URL to official page.
Return {{}} if no corrections possible. JSON only."""


def _search_and_correct(
    metadata: SoftwareMetadata,
    issues: list[str],
    llm: ChatOpenAI,
) -> dict[str, str]:
    """Targeted re-search to correct identified issues."""
    name = metadata.name
    queries = []
    for i in issues:
        if "author" in i:
            queries.append(f'"{name}" software developers CITATION.cff')
        elif "doi" in i:
            queries.append(f'"{name}" software DOI Zenodo JOSS')
        elif "url" in i or "paper_not_software" in i:
            queries.append(f'"{name}" software official site GitHub')
        elif "year" in i:
            queries.append(f'"{name}" software release changelog')
        elif "version" in i:
            queries.append(f'"{name}" software latest version')
    if not queries:
        queries.append(f'"{name}" software citation how to cite')

    search_tool = DuckDuckGoSearchResults(max_results=5, output_format="list")
    results = []
    for q in queries[:2]:
        try:
            raw = search_tool.invoke(q)
            if isinstance(raw, list):
                for r in raw:
                    if isinstance(r, dict):
                        results.append(f"Title: {r.get('title','')}\nSnippet: {r.get('snippet','')}\nLink: {r.get('link','')}")
                    else:
                        results.append(str(r))
            elif isinstance(raw, str):
                results.append(raw)
        except Exception as e:
            logger.warning(f"Search failed: {e}")

    if not results:
        return {}

    try:
        resp = llm.invoke([
            SystemMessage(content="Correct software citation metadata. Return valid JSON only."),
            HumanMessage(content=_CORRECTION_PROMPT.format(
                name=name, issues=", ".join(issues),
                authors=metadata.authors, version=metadata.version,
                year=metadata.year, publisher=metadata.publisher,
                doi=metadata.doi, url=metadata.url,
                search_results="\n---\n".join(results[:8]),
            )),
        ])
        content = resp.content.strip()
        if content.startswith("```"):
            content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Correction synthesis failed: {e}")
        return {}


def _is_instruction(value: str) -> bool:
    """Check if a value looks like an instruction rather than real data."""
    v = value.lower()
    return any(kw in v for kw in _INSTRUCTION_KEYWORDS)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def verify_citation(
    citation: SoftwareCitation,
    llm: ChatOpenAI,
    *,
    original_data: dict | None = None,
    do_correct: bool = True,
    context: str = "",
    llm_logger=None,
    pub_id: str = "",
) -> VerificationResult:
    """Verify a single citation and optionally correct errors.

    Args:
        citation: The citation to verify.
        llm: LangChain LLM instance.
        original_data: Ground-truth fields from the dataset (protected from overwrite).
            Expected keys: version, developer, url
        do_correct: If True, attempt corrections via LLM and web search.
        context: Original paper context / evidence sentences.
        llm_logger: Optional LLMLogger for recording LLM interactions.
        pub_id: Publication ID for logging.

    Returns:
        VerificationResult with issues found and corrections applied.
    """
    meta = citation.metadata
    name = citation.software_name

    # ── Phase 0: Determine protected fields (ground truth) ──
    original_data = original_data or {}
    protected = set()
    mapping = {"version": "version", "developer": "authors", "url": "url"}

    for orig_key, meta_key in mapping.items():
        orig_val = original_data.get(orig_key, "").strip()
        if orig_val:
            protected.add(meta_key)
            current = getattr(meta, meta_key, "")
            if current != orig_val:
                # For developer→authors, keep enriched if it's a proper expansion
                if orig_key == "developer" and current and orig_val.lower() in current.lower():
                    continue
                logger.info(f"Restored original {meta_key} for {name}: '{current}' → '{orig_val}'")
                setattr(meta, meta_key, orig_val)

    result = VerificationResult(
        software_name=name,
        is_correct=True,
        protected_fields=protected,
    )

    # ── Phase 1: Rule-based validation ──
    validators = [
        ("doi", _validate_doi(meta.doi)),
        ("url", _validate_url(meta.url, name)),
        ("year", _validate_year(meta.year)),
        ("authors", _validate_authors(meta.authors, name)),
        ("version", _validate_version(meta.version)),
    ]
    for _, field_issues in validators:
        result.issues.extend(field_issues)

    # Additional DOI check: is it for the software or a paper about it?
    if meta.doi and not any(i.issue_type in ("empty_or_placeholder", "invalid_format", "doi_not_resolving") for i in result.issues if i.field == "doi"):
        doi_relevance_issues = _validate_doi_is_software(meta.doi, name, llm)
        result.issues.extend(doi_relevance_issues)

    # Auto-fix: array-format authors
    author_array_issues = [i for i in result.issues if i.issue_type == "array_format"]
    if author_array_issues and "authors" not in protected:
        fixed = _fix_author_array(meta.authors)
        if fixed != meta.authors:
            result.corrections["authors"] = fixed
            meta.authors = fixed
            logger.info(f"Auto-fixed author format for {name}")

    if not result.issues:
        result.is_correct = True
        return result

    result.is_correct = False
    skip = name.lower().strip() in SKIP_NAMES

    # ── Phase 2: LLM consistency check ──
    if not skip and do_correct:
        has_data = any(getattr(meta, f, "") and not is_placeholder(getattr(meta, f, ""))
                       for f in ("doi", "authors", "url", "year"))
        if has_data:
            consistency = _check_consistency(meta, context, llm)
            if not consistency.get("is_correct", True):
                for issue_text in consistency.get("issues", []):
                    result.issues.append(FieldIssue("consistency", issue_text[:60]))
                for fld, val in (consistency.get("corrections") or {}).items():
                    fld = fld.lower()
                    if fld in protected:
                        continue
                    val = str(val) if val else ""
                    if not val or is_placeholder(val) or _is_instruction(val):
                        continue
                    if hasattr(meta, fld):
                        result.corrections[fld] = val
                        setattr(meta, fld, val)
                        logger.info(f"LLM corrected {name}.{fld}")

    # ── Phase 3: Re-search correction ──
    if not skip and do_correct:
        correctable = [f"{i.field}:{i.issue_type}" for i in result.issues
                       if i.issue_type != "empty_or_placeholder"]
        important_empty = [f"{i.field}:{i.issue_type}" for i in result.issues
                           if i.issue_type == "empty_or_placeholder" and i.field in ("authors", "url", "doi")]
        search_issues = correctable + important_empty

        if search_issues:
            try:
                corrections = _search_and_correct(meta, search_issues, llm)
                for fld, val in corrections.items():
                    fld = fld.lower()
                    val = str(val).strip() if val else ""
                    if fld in protected or not val or is_placeholder(val) or _is_instruction(val):
                        continue
                    if hasattr(meta, fld):
                        result.corrections[fld] = val
                        setattr(meta, fld, val)
                        logger.info(f"Search corrected {name}.{fld}")
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"Correction failed for {name}: {e}")
                time.sleep(1.5)

    # Rebuild citation text if corrections were made
    if result.corrections:
        citation.citation_text = _rebuild_citation(meta)
        citation.completeness_notes = _assess_completeness(meta)

    return result


def verify_citations(
    citations: list[SoftwareCitation],
    llm: ChatOpenAI,
    *,
    original_data_list: list[dict] | None = None,
    do_correct: bool = True,
    contexts: list[str] | None = None,
) -> tuple[list[VerificationResult], VerificationStats]:
    """Verify a batch of citations.

    Args:
        citations: List of citations to verify.
        llm: LangChain LLM instance.
        original_data_list: Per-citation ground-truth dicts (parallel to citations).
        do_correct: Whether to attempt corrections.
        contexts: Per-citation context strings.

    Returns:
        Tuple of (list of VerificationResults, aggregate VerificationStats).
    """
    original_data_list = original_data_list or [{}] * len(citations)
    contexts = contexts or [""] * len(citations)
    stats = VerificationStats()

    results = []
    for i, (cit, orig, ctx) in enumerate(zip(citations, original_data_list, contexts)):
        result = verify_citation(cit, llm, original_data=orig, do_correct=do_correct, context=ctx)
        results.append(result)
        stats.total += 1

        if result.is_correct:
            stats.already_correct += 1
        else:
            stats.with_issues += 1
            if result.corrections:
                stats.corrected += 1

        # Aggregate field quality
        for issue in result.issues:
            key = f"{issue.field}:{issue.issue_type}"
            stats.issue_counts[key] = stats.issue_counts.get(key, 0) + 1
            if issue.field in stats.field_quality:
                if issue.issue_type == "empty_or_placeholder":
                    stats.field_quality[issue.field]["empty"] += 1
                else:
                    stats.field_quality[issue.field]["invalid"] += 1

        for fld in stats.field_quality:
            if not any(i.field == fld for i in result.issues):
                stats.field_quality[fld]["valid"] += 1

        for fld in result.corrections:
            stats.correction_counts[fld] = stats.correction_counts.get(fld, 0) + 1
            if fld in stats.field_quality:
                stats.field_quality[fld]["corrected"] += 1

    return results, stats


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _rebuild_citation(meta: SoftwareMetadata) -> str:
    """Rebuild FORCE11 citation text from metadata."""
    authors = meta.authors or meta.name
    year = meta.year or "n.d."
    ver = f" (Version {meta.version})" if meta.version and not is_placeholder(meta.version) else ""
    pub = f" {meta.publisher}." if meta.publisher and not is_placeholder(meta.publisher) else ""
    ident = meta.doi or meta.url
    if ident and not is_placeholder(ident):
        if not ident.startswith("http") and re.match(r'10\.\d{4,}/', ident):
            ident = f"https://doi.org/{ident}"
        ident = f" {ident}"
    else:
        ident = ""
    return f"{authors}. ({year}). {meta.name}{ver} [Software].{pub}{ident}"


def _assess_completeness(meta: SoftwareMetadata) -> str:
    """Re-assess FORCE11 completeness."""
    missing = []
    if not meta.authors or is_placeholder(meta.authors):
        missing.append("authors (Credit & Attribution)")
    if not meta.version or is_placeholder(meta.version):
        missing.append("version (Specificity)")
    if not meta.doi or is_placeholder(meta.doi):
        missing.append("DOI (Unique Identification & Persistence)")
    if not meta.year or is_placeholder(meta.year):
        missing.append("year (Accessibility)")
    if (not meta.url or is_placeholder(meta.url)) and (not meta.doi or is_placeholder(meta.doi)):
        missing.append("URL or DOI (Accessibility)")
    if not meta.publisher or is_placeholder(meta.publisher):
        missing.append("publisher (Accessibility)")
    return "Missing fields: " + "; ".join(missing) if missing else "All FORCE11 required fields are present."
