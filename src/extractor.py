"""Software Extractor — uses LLM to identify software mentions in parsed papers."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.models import PaperInfo, SoftwareMention

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert at identifying SOFTWARE mentioned in academic papers.

Your task is to extract ALL software mentions from the given text. For each software, extract:
- name: The software name EXACTLY as it appears in the paper (e.g., "SPSS", "R", "MATLAB", not the full official name)
- version: The version number if mentioned (e.g., "2.1.0", "21", "3.6")
- publisher: The publisher, organization, or author(s) if mentioned
- url: Any URL or link provided for the software
- context: The sentence where the software is mentioned

INCLUDE the following types — ALL of these count as software:
- Statistical software (SPSS, SAS, Stata, R, GraphPad Prism, Excel)
- Programming languages and environments (R, Python, Java, C++, Perl, MATLAB, Fortran) — these ARE software
- Operating systems (Windows, Linux, macOS, Ubuntu, CentOS, Red Hat, Debian, Unix, Macintosh prevous name for Mac) — these ARE software
- Bioinformatics tools (BLAST, Bowtie, samtools, BWA, GATK, SPM, FSL)
- Data analysis packages and libraries (NumPy, pandas, scikit-learn, ggplot2, lme4, car)
- Image/signal processing software (ImageJ/Image j, Fiji, Photoshop, Praat)
- Computational tools and simulators (GROMACS, CASTEP, Gaussian, BEAST)
- Development platforms (PyTorch, TensorFlow, Bioconductor, Galaxy)
- Specialized research tools (MrBayes, PAUP*, ClustalW, Mothur)
- Version control and collaboration platforms (GitHub, GitLab, PyPI, CRAN) — only when the paper uses them as a tool

IMPORTANT — CONTEXT MATTERS for operating systems and programming languages:
Operating systems and languages ARE software, but ONLY extract them when INDEPENDENTLY used or cited.

✅ EXTRACT these (independent / direct use):
- "analyses were performed using Windows 10" → Windows is the working environment
- "data were analyzed using R version 3.4.1" → R is the analysis tool
- "implemented in Java 1.8" → Java is the development platform
- "the pipeline was written in Python 3.9" → Python is the runtime
- "running Ubuntu 18.04 on the cluster" → Ubuntu is the server OS

❌ DO NOT EXTRACT these (dependent / modifier / comparative use):
- "SPSS Version 18.0 for Windows" → "Windows" is just a platform qualifier for SPSS
- "GraphPad Prism 6.0 for Mac OS X" → "Mac OS X" is a platform qualifier
- "written in R (as opposed to C or C++)" → C/C++ are only mentioned as comparison
- "Programming languages: Perl, C++ (required for BLASR)" → C++ is a dependency of BLASR
- "SAS 9.4 for Windows" → "Windows" is a platform qualifier

KEY RULE: DO NOT extract a language/OS if it appears ONLY as:
  1. A "for [OS]" qualifier after another software name
  2. A comparison/contrast ("as opposed to", "unlike", "instead of", "rather than")
  3. A dependency/prerequisite of another tool without direct use

Do NOT include (these are NOT software):
- Laboratory reagent kits and consumables (e.g., "DNeasy Blood and Tissue Kit", "SYBR Green Master Mix", "BCA Protein Assay Kit", "TRIzol Reagent")
- Laboratory instruments and hardware (e.g., PCR machines, centrifuges, spectrometers, microscopes, NanoDrop)
- Biological constructs (e.g., UAS-GFP, Gr32a-Gal4, plasmid names, cell lines, antibodies, primers)
- Psychometric scales and questionnaires (e.g., "General Health Questionnaire", "Beck Depression Inventory", "SF-36")
- Online databases and data repositories that are purely for browsing data (e.g., "PubMed", "Web of Science", "ISI-WoS", "PsycINFO", "Scopus", "MEDLINE", "Science Citation Index", "Social Sciences Citation Index", "Arts & Humanities Citation Index", "Cochrane Library", "ClinicalTrials.gov", "GenBank", "UniProt")
- Search engines and general web services (e.g., "Google Scholar", "Google Search", "UpToDate")
- Generic device categories (e.g., "iPhone application", "smartphone app", "web browser")
- Hardware (GPUs, CPUs, servers, cameras)

Additional Rules:
1. Extract the name EXACTLY as written in the paper text — do NOT expand abbreviations (e.g., use "SPSS" not "IBM SPSS Statistics", use "Excel" not "Microsoft Excel")
2. If the same software appears multiple times, include it only ONCE with the most complete information
3. Same software with different name should be extracted, for example, R and RStudio, Image J and ImageJ, 
4. If unsure whether something is software or a database/questionnaire, do NOT include it
5. Return your answer as a JSON array. Return [] if no software is found.

Example output:
[
  {
    "name": "SPSS",
    "version": "21",
    "publisher": "IBM",
    "url": "",
    "context": "Statistical analysis was conducted using SPSS 21 (IBM)."
  },
  {
    "name": "R",
    "version": "3.4.1",
    "publisher": "R Foundation",
    "url": "https://www.R-project.org/",
    "context": "All analyses were carried out using R v3.4.1 (R Foundation for Statistical Computing)."
  },
  {
    "name": "Windows",
    "version": "10",
    "publisher": "Microsoft",
    "url": "",
    "context": "Data were collected on a computer running Windows 10."
  },
  {
    "name": "Java",
    "version": "1.8",
    "publisher": "Oracle",
    "url": "",
    "context": "The pipeline was implemented in Java 1.8."
  }
]

NOT software examples (do NOT extract these):
- "DNA was extracted using the DNeasy Blood and Tissue Kit (Qiagen)" → reagent kit
- "PCR was performed on an ABI 7500 Fast Real-Time PCR System" → lab instrument
- "Articles were retrieved from PubMed and Web of Science" → databases/data repositories
- "Expression of UAS-GFP was driven by elav-Gal4" → genetic construct
- "Mental health was assessed using the General Health Questionnaire-12" → questionnaire
- "ISI-Web of Science, Science Citation Index Expanded" → bibliographic databases"""

USER_PROMPT_TEMPLATE = """Extract all software mentions from the following academic paper text.
Return ONLY a JSON array, no other text.

--- Paper Text ---
{text}
--- End of Text ---"""


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------

def extract_software_mentions(
    paper: PaperInfo,
    llm: ChatOpenAI,
    llm_logger=None,
    pub_id: str = "",
) -> list[SoftwareMention]:
    """Extract software mentions from a parsed paper using an LLM.

    For long papers, processes text in chunks and deduplicates results.
    After basic dedup, uses LLM to merge abbreviation/full-name duplicates,
    fix name+version concatenation, and filter obvious non-software.

    Args:
        paper: Parsed paper information.
        llm: LangChain ChatOpenAI instance.
        llm_logger: Optional LLMLogger for recording LLM interactions.
        pub_id: Publication ID for logging.

    Returns:
        Deduplicated and aggregated list of SoftwareMention objects.
    """
    full_text = paper.full_text
    if not full_text.strip():
        logger.warning("Paper has no text content to analyze.")
        return []

    # Decide whether to chunk (rough token estimate: 1 token ≈ 4 chars)
    estimated_tokens = len(full_text) // 4
    max_chunk_tokens = 6000  # conservative limit for context

    if estimated_tokens <= max_chunk_tokens:
        chunks = [full_text]
    else:
        chunks = _split_into_chunks(paper, max_chars=max_chunk_tokens * 4)

    all_mentions: list[SoftwareMention] = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
        mentions = _extract_from_chunk(chunk, llm, llm_logger=llm_logger, pub_id=pub_id, chunk_idx=i)
        all_mentions.extend(mentions)

    # Step 1: Dictionary-based deduplication (fast, free)
    deduplicated = _deduplicate_mentions(all_mentions)
    logger.info(f"{pub_id}: After dict dedup: {len(all_mentions)} → {len(deduplicated)} mentions: "
                f"{[m.name for m in deduplicated]}")

    # Step 2: LLM-based smart aggregation (merge abbreviations, fix names, filter)
    aggregated = _llm_aggregate_mentions(deduplicated, llm, llm_logger=llm_logger, pub_id=pub_id)
    logger.info(f"{pub_id}: After LLM aggregation: {len(deduplicated)} → {len(aggregated)} mentions: "
                f"{[m.name for m in aggregated]}")

    return aggregated


def _extract_from_chunk(text: str, llm: ChatOpenAI, llm_logger=None, pub_id: str = "", chunk_idx: int = 0) -> list[SoftwareMention]:
    """Extract software mentions from a single text chunk."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT_TEMPLATE.format(text=text)),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    # Log LLM interaction
    if llm_logger:
        llm_logger.log("extractor", f"extract_chunk_{chunk_idx}", pub_id, "", messages, content)

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        raw_list = json.loads(content)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM response as JSON:\n{content[:500]}")
        return []

    mentions = []
    for item in raw_list:
        try:
            mention = SoftwareMention(
                name=item.get("name", ""),
                version=item.get("version", ""),
                publisher=item.get("publisher", ""),
                url=item.get("url", ""),
                context=item.get("context", ""),
            )
            if mention.name:
                mentions.append(mention)
        except Exception as e:
            logger.warning(f"Skipping invalid mention: {e}")

    return mentions


def _split_into_chunks(paper: PaperInfo, max_chars: int) -> list[str]:
    """Split paper into chunks, preferring section boundaries."""
    chunks = []
    current_chunk = ""

    # Always include abstract in the first chunk
    if paper.abstract:
        current_chunk = f"Abstract:\n{paper.abstract}\n\n"

    for section in paper.sections:
        section_text = ""
        if section.title:
            section_text += f"{section.title}:\n"
        section_text += section.content + "\n\n"

        if len(current_chunk) + len(section_text) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = section_text
        else:
            current_chunk += section_text

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [paper.full_text]


def _deduplicate_mentions(mentions: list[SoftwareMention]) -> list[SoftwareMention]:
    """Deduplicate mentions by software name, keeping the most complete entry."""
    seen: dict[str, SoftwareMention] = {}

    for m in mentions:
        key = m.name.lower().strip()
        if key not in seen:
            seen[key] = m
        else:
            # Merge: keep the entry with more filled fields
            existing = seen[key]
            merged = SoftwareMention(
                name=m.name if m.name else existing.name,
                version=m.version or existing.version,
                publisher=m.publisher or existing.publisher,
                url=m.url or existing.url,
                context=m.context or existing.context,
            )
            seen[key] = merged

    return list(seen.values())


# ---------------------------------------------------------------------------
# LLM-based smart aggregation prompt
# ---------------------------------------------------------------------------

_LLM_AGGREGATE_SYSTEM = """You are given a list of software names extracted from an academic paper.
Your tasks (be CONSERVATIVE — only act when you are CERTAIN):

1. MERGE duplicates: If two entries clearly refer to the same software
   (e.g., "Statistical Package for Social Sciences" and "SPSS"), merge them
   into one entry keeping the most common/short name and combining all metadata fields.
   Use the shorter/more common name (e.g., keep "SPSS" not "Statistical Package for Social Sciences").

2. FIX name+version: If a software name contains a version number
   (e.g., name="SPSS 17", version=""), split to name="SPSS", version="17".
   Similarly: "Stata SE 11.0" → name="Stata", version="11.0".
   "R 3.4.1" → name="R", version="3.4.1".
   ONLY do this for clear version patterns (digits, x.y.z format).

3. FILTER false positives: Remove entries that are CLEARLY NOT software tools.
   Examples of things to REMOVE:
   - Reagent kits, biological assays, antibodies
   - Questionnaires and psychometric scales
   - Pure databases (PubMed, GenBank) unless used as analysis tools
   - Generic terms that are not specific software
   - OS/language appearing ONLY as a platform qualifier for another software
     (e.g., "Windows" in "SPSS for Windows", "Mac OS X" in "Prism for Mac OS X")
   - Language mentioned only in comparison/contrast ("as opposed to C", "unlike Java")
   - Language mentioned only as a dependency of another tool, not directly used
   ONLY filter if you are VERY confident it is not software.

CRITICAL RULES:
- When in doubt, do NOT merge, do NOT fix, do NOT filter. Keep the entry.
- Operating systems (Windows, Linux, macOS) ARE software WHEN USED INDEPENDENTLY.
  Filter them if they appear only as platform qualifiers (e.g., "SPSS for Windows").
- Programming languages (R, Python, Java) ARE software WHEN USED AS TOOLS.
  Filter them if they appear only in comparisons or as other tools' dependencies.
- All changes must preserve information — when merging, combine all fields.
- Return the COMPLETE list (including unchanged entries).

Return a JSON array with the same fields: name, version, publisher, url, context.
Also return a "changes" array documenting what you changed, each with:
  {"action": "merged"|"fixed_name"|"filtered", "original": "...", "result": "...", "reason": "..."}

Return format:
{
  "software": [...],
  "changes": [...]
}"""

_LLM_AGGREGATE_USER = """Here are the software mentions extracted from a paper.
Review and apply conservative merging, name fixing, and filtering.

Software list:
{mentions_json}

Return a JSON object with "software" and "changes" arrays."""


def _llm_aggregate_mentions(
    mentions: list[SoftwareMention],
    llm: ChatOpenAI,
    llm_logger=None,
    pub_id: str = "",
) -> list[SoftwareMention]:
    """Use LLM to intelligently aggregate software mentions.

    Merges full-name/abbreviation duplicates, fixes name+version concatenation,
    and filters obvious non-software false positives.
    Uses a conservative approach — only acts when certain.

    Args:
        mentions: List of deduplicated SoftwareMention objects.
        llm: LangChain ChatOpenAI instance.
        llm_logger: Optional LLMLogger for recording interactions.
        pub_id: Publication ID for logging.

    Returns:
        Aggregated list of SoftwareMention objects.
    """
    if len(mentions) <= 1:
        return mentions

    # Prepare input for LLM
    mentions_data = []
    for m in mentions:
        mentions_data.append({
            "name": m.name,
            "version": m.version,
            "publisher": m.publisher,
            "url": m.url,
            "context": m.context[:300],  # Truncate for token savings
        })

    messages = [
        SystemMessage(content=_LLM_AGGREGATE_SYSTEM),
        HumanMessage(content=_LLM_AGGREGATE_USER.format(
            mentions_json=json.dumps(mentions_data, ensure_ascii=False, indent=2)
        )),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()

        # Log LLM interaction
        if llm_logger:
            llm_logger.log("extractor", "aggregate", pub_id, "", messages, content)

        # Strip markdown fences
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        data = json.loads(content)

        # Parse changes for logging
        changes = data.get("changes", [])
        if changes:
            merged_names = [c["original"] for c in changes if c.get("action") == "merged"]
            fixed_names = [c["original"] for c in changes if c.get("action") == "fixed_name"]
            filtered_names = [c["original"] for c in changes if c.get("action") == "filtered"]
            logger.info(f"{pub_id}: Aggregation changes: "
                        f"merged={merged_names}, fixed={fixed_names}, filtered={filtered_names}")
            for c in changes:
                logger.info(f"  {c.get('action')}: {c.get('original')} → {c.get('result')} "
                            f"(reason: {c.get('reason')})")
        else:
            logger.info(f"{pub_id}: LLM aggregation: no changes needed")

        # Build result list
        software_list = data.get("software", [])
        result = []
        for item in software_list:
            try:
                # Try to find original context (LLM may have truncated it)
                original_context = ""
                item_name_lower = item.get("name", "").lower().strip()
                for orig in mentions:
                    if orig.name.lower().strip() == item_name_lower:
                        original_context = orig.context
                        break

                mention = SoftwareMention(
                    name=item.get("name", ""),
                    version=str(item.get("version", "")),
                    publisher=item.get("publisher", ""),
                    url=item.get("url", ""),
                    context=item.get("context", "") or original_context,
                )
                if mention.name:
                    result.append(mention)
            except Exception as e:
                logger.warning(f"Skipping invalid aggregated mention: {e}")

        # Safety: if LLM returned empty or suspiciously few results, keep originals
        if len(result) == 0 and len(mentions) > 0:
            logger.warning(f"{pub_id}: LLM aggregation returned empty list. Keeping originals.")
            return mentions

        return result

    except Exception as e:
        logger.warning(f"{pub_id}: LLM aggregation failed: {e}. Keeping originals.")
        return mentions
