#!/usr/bin/env python3
"""GPT-Only Baseline — Single-call software citation without agent tools.

For each paper, sends a single GPT API call with the full paper text and a
comprehensive prompt. The LLM must extract software mentions AND generate
complete FORCE11 citations using only its parametric knowledge (no web search,
no verification, no multi-step pipeline).

Usage:
    python3.11 baseline/run_baseline.py
    python3.11 baseline/run_baseline.py --limit 50
    python3.11 baseline/run_baseline.py --resume --workers 4
    python3.11 baseline/run_baseline.py --model gpt-4o
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

load_dotenv()

console = Console()
logger = logging.getLogger(__name__)

BASELINE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASELINE_DIR / "parser_cache"
OUTPUT_FILE = BASELINE_DIR / "baseline_citations.json"
LOG_FILE = BASELINE_DIR / "baseline.log"
FAILED_FILE = BASELINE_DIR / "baseline_failed.json"

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_WORKERS = 4
MAX_PAPER_CHARS = 80000  # Truncate very long papers
PAPER_TIMEOUT = 300      # 5 minutes per paper

# ──────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at identifying SOFTWARE mentioned in academic papers and generating FORCE11-compliant software citations.

Your task has TWO parts:
1. EXTRACT all software mentions from the paper text
2. For each software, GENERATE a complete citation using your knowledge

## PART 1: Software Extraction Rules

INCLUDE the following types — ALL of these count as software:
- Statistical software (SPSS, SAS, Stata, R, GraphPad Prism, Excel)
- Programming languages and environments (R, Python, Java, C++, Perl, MATLAB, Fortran)
- Operating systems (Windows, Linux, macOS, Ubuntu) — ONLY when independently used, NOT as platform qualifiers
- Bioinformatics tools (BLAST, Bowtie, samtools, BWA, GATK, SPM, FSL)
- Data analysis packages and libraries (NumPy, pandas, scikit-learn, ggplot2)
- Image/signal processing software (ImageJ, Fiji, Photoshop, Praat)
- Computational tools and simulators (GROMACS, CASTEP, Gaussian, BEAST)
- Development platforms (PyTorch, TensorFlow, Bioconductor, Galaxy)
- Specialized research tools (MrBayes, PAUP*, ClustalW, Mothur)

DO NOT EXTRACT operating systems or languages when they appear ONLY as:
1. A "for [OS]" qualifier after another software name (e.g., "SPSS for Windows")
2. A comparison/contrast mention
3. A dependency/prerequisite without direct use

DO NOT include (NOT software):
- Laboratory reagent kits, consumables, instruments, hardware
- Biological constructs, cell lines, antibodies
- Psychometric scales and questionnaires
- Online databases and repositories (PubMed, GenBank, UniProt, Web of Science, Scopus, MEDLINE)
- Search engines (Google Scholar)

## PART 2: Citation Generation Rules

For each extracted software, generate:
- **citation_text**: A FORCE11-compliant citation string: "Author(s). (Year). Name (Version X.Y) [Software]. Publisher. URL_or_DOI"
- **completeness_notes**: Note any missing fields (e.g., "Missing fields: DOI")
- **evidence_sentences**: The sentence(s) from the paper where the software is mentioned
- **enriched_metadata**: Use your training knowledge to fill in as much as possible:
  - name, version, authors, year, publisher, doi, url, license, description
  - If a field is not available, leave it as empty string ""
- **original_mention**: What was literally in the paper text:
  - name, version, publisher, url, context (the sentence)

## IMPORTANT RULES
1. Extract names EXACTLY as written in the paper (e.g., "SPSS" not "IBM SPSS Statistics")
2. Same software mentioned multiple times → include only ONCE with most complete info
3. If unsure whether something is software, do NOT include it
4. For metadata, use your best knowledge — do NOT hallucinate DOIs or URLs you're not confident about
5. If you don't know a field value, use empty string ""

## OUTPUT FORMAT
Return a JSON array. Return [] if no software is found.
Each element must have this exact structure:
{
  "software_name": "...",
  "citation_text": "...",
  "completeness_notes": "...",
  "evidence_sentences": "...",
  "enriched_metadata": {
    "name": "...", "version": "...", "authors": "...", "year": "...",
    "publisher": "...", "doi": "...", "url": "...", "license": "...", "description": "..."
  },
  "original_mention": {
    "name": "...", "version": "...", "publisher": "...", "url": "...", "context": "..."
  }
}"""


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a"),
            logging.StreamHandler(),
        ],
    )
    for name in ("httpx", "httpcore", "urllib3", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


def load_parsed_paper(pmcid: str) -> dict:
    """Load a pre-parsed paper from cache."""
    path = CACHE_DIR / f"{pmcid}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def reconstruct_text(parsed: dict, max_chars: int = MAX_PAPER_CHARS) -> str:
    """Reconstruct readable paper text from parsed JSON."""
    parts = []

    title = parsed.get("title", "")
    if title:
        parts.append(f"Title: {title}")

    authors = parsed.get("authors", "")
    if authors:
        parts.append(f"Authors: {authors}")

    abstract = parsed.get("abstract", "")
    if abstract:
        parts.append(f"\nAbstract:\n{abstract}")

    for section in parsed.get("sections", []):
        sec_title = section.get("title", "")
        sec_content = section.get("content", "")
        if sec_content:
            if sec_title:
                parts.append(f"\n## {sec_title}\n{sec_content}")
            else:
                parts.append(f"\n{sec_content}")

    text = "\n".join(parts)

    # Truncate if too long
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... truncated ...]"

    return text


def parse_llm_response(response_text: str) -> list:
    """Parse the LLM JSON response, handling common formatting issues."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they are code fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            # Sometimes LLM wraps in {"software_citations": [...]}
            for key in ("software_citations", "citations", "software", "results"):
                if key in result and isinstance(result[key], list):
                    return result[key]
            return [result]
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
        # Try to find JSON array in the text
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []


def ensure_schema(citation: dict) -> dict:
    """Ensure a citation dict has all required fields with correct types."""
    em = citation.get("enriched_metadata", {})
    om = citation.get("original_mention", {})

    return {
        "software_name": str(citation.get("software_name", "")),
        "citation_text": str(citation.get("citation_text", "")),
        "completeness_notes": str(citation.get("completeness_notes", "")),
        "evidence_sentences": str(citation.get("evidence_sentences", "")),
        "enriched_metadata": {
            "name": str(em.get("name", "")),
            "version": str(em.get("version", "")),
            "authors": str(em.get("authors", "")),
            "year": str(em.get("year", "")),
            "publisher": str(em.get("publisher", "")),
            "doi": str(em.get("doi", "")),
            "url": str(em.get("url", "")),
            "license": str(em.get("license", "")),
            "description": str(em.get("description", "")),
        },
        "original_mention": {
            "name": str(om.get("name", "")),
            "version": str(om.get("version", "")),
            "publisher": str(om.get("publisher", "")),
            "url": str(om.get("url", "")),
            "context": str(om.get("context", "")),
        },
    }


# ──────────────────────────────────────────────────────────────
# Core: Process one paper
# ──────────────────────────────────────────────────────────────

_write_lock = threading.Lock()


def process_paper(client: OpenAI, model: str, pmcid: str) -> dict:
    """Process a single paper through GPT baseline."""
    t0 = time.time()
    logger.info(f"Processing {pmcid}...")

    parsed = load_parsed_paper(pmcid)
    paper_text = reconstruct_text(parsed)

    user_msg = f"Extract all software mentions from the following academic paper and generate complete FORCE11 citations.\n\n{paper_text}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    citations_raw = parse_llm_response(raw)
    citations = [ensure_schema(c) for c in citations_raw]

    elapsed = time.time() - t0
    sw_names = [c["software_name"] for c in citations]
    logger.info(f"{pmcid}: {len(citations)} citations built in {elapsed:.1f}s — {sw_names}")

    return {
        "publication_id": pmcid,
        "title": parsed.get("title", ""),
        "authors": parsed.get("authors", ""),
        "software_citations": citations,
    }


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

def load_existing_results() -> list:
    """Load existing results for resume."""
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(results: list):
    """Atomically save results to output file."""
    tmp = OUTPUT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp.rename(OUTPUT_FILE)


def run_baseline(model: str, limit: int | None, workers: int, resume: bool):
    """Main baseline runner."""
    setup_logging()

    # Discover all papers from parser cache
    all_pmcids = sorted(
        f.stem for f in CACHE_DIR.glob("PMC*.json")
    )
    logger.info(f"Found {len(all_pmcids)} papers in parser cache")

    # Resume support
    results = []
    done_ids = set()
    if resume:
        results = load_existing_results()
        done_ids = {r["publication_id"] for r in results}
        logger.info(f"Resuming: {len(done_ids)} already done")

    # Filter to pending
    pending = [pid for pid in all_pmcids if pid not in done_ids]
    if limit:
        pending = pending[:limit]

    if not pending:
        console.print("[green]All papers already processed![/green]")
        return

    console.print(f"[bold]Baseline: {len(pending)} papers to process "
                  f"(model={model}, workers={workers})[/bold]")

    # Load failed tracking
    failed = {}
    if FAILED_FILE.exists():
        with open(FAILED_FILE, "r") as f:
            failed = json.load(f)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    processed = 0
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing papers...", total=len(pending))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for pmcid in pending:
                fut = executor.submit(process_paper, client, model, pmcid)
                futures[fut] = pmcid

            for future in as_completed(futures):
                pmcid = futures[future]
                try:
                    result = future.result(timeout=PAPER_TIMEOUT)
                    with _write_lock:
                        results.append(result)
                        processed += 1
                        # Save checkpoint every 10 papers
                        if processed % 10 == 0:
                            save_results(results)
                            logger.info(f"Checkpoint saved: {len(results)} papers total")
                    n_sw = len(result.get("software_citations", []))
                    progress.update(task, advance=1,
                                    description=f"[green]{pmcid}[/green] → {n_sw} software")
                except Exception as e:
                    errors += 1
                    logger.error(f"Failed {pmcid}: {e}")
                    failed[pmcid] = str(e)
                    progress.update(task, advance=1,
                                    description=f"[red]FAIL {pmcid}[/red]")

    # Final save
    save_results(results)

    # Save failed
    if failed:
        with open(FAILED_FILE, "w") as f:
            json.dump(failed, f, indent=2)

    # Summary
    total_sw = sum(len(r.get("software_citations", [])) for r in results)
    console.print(f"\n[bold green]✅ Done![/bold green]")
    console.print(f"  Processed: {processed} papers this run")
    console.print(f"  Total:     {len(results)} papers")
    console.print(f"  Software:  {total_sw} total mentions")
    console.print(f"  Errors:    {errors}")
    console.print(f"  Output:    {OUTPUT_FILE}")


def main():
    parser = argparse.ArgumentParser(description="GPT-Only Baseline for Software Citation")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    args = parser.parse_args()

    run_baseline(args.model, args.limit, args.workers, args.resume)


if __name__ == "__main__":
    main()
