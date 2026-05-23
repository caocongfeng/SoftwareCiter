"""Agent Orchestrator — coordinates the full software citation pipeline."""

import logging
import time

from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.citation_builder import build_citation
from src.config import load_config
from src.extractor import extract_software_mentions
from src.models import PaperInfo, SoftwareCitation, SoftwareMention
from src.parser import parse_document
from src.searcher import search_software_metadata
from src.verifier import verify_citation, VerificationResult

logger = logging.getLogger(__name__)
console = Console()


def run_agent(
    file_path: str,
    config: dict | None = None,
    verbose: bool = True,
    verify: bool = True,
) -> list[SoftwareCitation]:
    """Run the full software citation agent pipeline.

    Steps:
        1. Parse the paper file
        2. Extract software mentions via LLM
        3. For each mention, search the web for metadata
        4. Build FORCE11-compliant citations
        5. Verify and correct citations (optional)

    Args:
        file_path: Path to the academic paper file.
        config: Configuration dict (loads from env if not provided).
        verbose: Whether to print progress to console.
        verify: Whether to run verification & correction (Step 5).

    Returns:
        List of SoftwareCitation objects.
    """
    if config is None:
        config = load_config()

    # Initialize LLM
    llm = ChatOpenAI(
        model=config["openai_model"],
        temperature=config["llm_temperature"],
        api_key=config["openai_api_key"],
    )

    citations: list[SoftwareCitation] = []

    # --- Step 1: Parse paper ---
    if verbose:
        console.print("\n[bold cyan]📄 Step 1: Parsing paper...[/bold cyan]")

    paper = parse_document(file_path)

    if verbose:
        console.print(f"  Title: [green]{paper.title}[/green]")
        console.print(f"  Authors: {paper.authors}")
        console.print(f"  Format: {paper.file_format}")
        console.print(f"  Sections: {len(paper.sections)}")

    # --- Step 2: Extract software mentions ---
    if verbose:
        console.print("\n[bold cyan]🔍 Step 2: Extracting software mentions...[/bold cyan]")

    mentions = extract_software_mentions(paper, llm)

    if verbose:
        if mentions:
            table = Table(title="Software Mentions Found")
            table.add_column("Software", style="green")
            table.add_column("Version", style="yellow")
            table.add_column("Publisher", style="blue")
            table.add_column("URL", style="dim")
            for m in mentions:
                table.add_row(m.name, m.version or "—", m.publisher or "—", m.url or "—")
            console.print(table)
        else:
            console.print("  [yellow]No software mentions found.[/yellow]")
            return citations

    # --- Step 3 & 4: Search metadata and build citations ---
    if verbose:
        console.print(
            f"\n[bold cyan]🌐 Step 3–4: Searching metadata & building citations "
            f"for {len(mentions)} software tools...[/bold cyan]"
        )

    for i, mention in enumerate(mentions):
        if verbose:
            console.print(
                f"\n  [{i + 1}/{len(mentions)}] Processing: [bold green]{mention.name}[/bold green]"
            )

        # Step 3: Web search
        if verbose:
            console.print(f"    🔎 Searching web for metadata...")

        try:
            metadata = search_software_metadata(
                mention,
                llm,
                max_results=config["search_max_results"],
            )
        except Exception as e:
            logger.error(f"Search failed for {mention.name}: {e}")
            if verbose:
                console.print(f"    [red]Search failed: {e}[/red]")
            # Fallback: use what we have from the paper
            from src.models import SoftwareMetadata
            metadata = SoftwareMetadata(
                name=mention.name,
                version=mention.version,
                authors=mention.publisher,
                url=mention.url,
            )

        # Step 4: Build citation
        if verbose:
            console.print(f"    📝 Building citation...")

        try:
            citation = build_citation(metadata, llm)
        except Exception as e:
            logger.error(f"Citation building failed for {mention.name}: {e}")
            if verbose:
                console.print(f"    [red]Citation failed: {e}[/red]")
            from src.citation_builder import build_citation_simple
            citation = build_citation_simple(metadata)

        citations.append(citation)

        if verbose:
            console.print(f"    ✅ {citation.citation_text}")
            if citation.completeness_notes and "Missing" in citation.completeness_notes:
                console.print(f"    [yellow]⚠  {citation.completeness_notes}[/yellow]")

    # --- Step 5: Verify and correct citations ---
    if verify and citations:
        if verbose:
            console.print(
                f"\n[bold cyan]✅ Step 5: Verifying & correcting "
                f"{len(citations)} citations...[/bold cyan]"
            )

        corrected_count = 0
        for citation in citations:
            try:
                result = verify_citation(
                    citation,
                    llm,
                    do_correct=True,
                    context=citation.metadata.description,
                )
                if result.corrections:
                    corrected_count += 1
                    if verbose:
                        console.print(
                            f"    🔧 [yellow]{citation.software_name}[/yellow]: "
                            f"corrected {', '.join(result.corrections.keys())}"
                        )
                elif not result.is_correct and verbose:
                    issue_fields = {i.field for i in result.issues if i.issue_type != "empty_or_placeholder"}
                    if issue_fields:
                        console.print(
                            f"    ⚠  [dim]{citation.software_name}: issues in {', '.join(issue_fields)}[/dim]"
                        )
            except Exception as e:
                logger.warning(f"Verification failed for {citation.software_name}: {e}")

        if verbose:
            console.print(
                f"\n    Verified: {len(citations)} | "
                f"Corrected: [blue]{corrected_count}[/blue] | "
                f"Clean: [green]{len(citations) - corrected_count}[/green]"
            )

    # --- Final output ---
    if verbose:
        _print_final_summary(citations)

    return citations


def _print_final_summary(citations: list[SoftwareCitation]) -> None:
    """Print a nicely formatted summary of all citations."""
    console.print("\n")
    console.print(
        Panel(
            "[bold]Software Citations (FORCE11 Compliant)[/bold]",
            style="bold cyan",
        )
    )

    for i, c in enumerate(citations, 1):
        console.print(f"\n  [bold]{i}. {c.software_name}[/bold]")
        console.print(f"     {c.citation_text}")
        if c.completeness_notes:
            note_style = "green" if "All FORCE11" in c.completeness_notes else "yellow"
            console.print(f"     [{note_style}]{c.completeness_notes}[/{note_style}]")

    console.print(f"\n  [dim]Total: {len(citations)} software citations generated.[/dim]\n")
