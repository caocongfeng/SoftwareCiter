"""CLI entry point for the Software Citation Agent."""

import argparse
import json
import logging
import sys

from rich.console import Console

from src.agent import run_agent
from src.config import load_config

console = Console()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Software Citation Agent — Extract and build FORCE11-compliant "
        "software citations from academic papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --input paper.xml
  python -m src.main --input paper.xml --output citations.json --format json
  python -m src.main --input paper.txt --format bibtex
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the academic paper file (XML, TXT)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (prints to stdout if not specified)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "bibtex"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        config = load_config()
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)

    # Override format from CLI
    config["output_format"] = args.format

    # Run the agent
    try:
        citations = run_agent(
            file_path=args.input,
            config=config,
            verbose=not args.quiet,
        )
    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logging.exception("Agent failed")
        sys.exit(1)

    # Format output
    output = _format_output(citations, args.format)

    # Write or print
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        if not args.quiet:
            console.print(f"\n[green]Citations saved to: {args.output}[/green]")
    elif args.quiet:
        # In quiet mode without output file, print the formatted result
        print(output)


def _format_output(citations: list, fmt: str) -> str:
    """Format citations for output."""
    if fmt == "json":
        data = [
            {
                "software_name": c.software_name,
                "citation": c.citation_text,
                "completeness": c.completeness_notes,
                "metadata": c.metadata.model_dump(),
            }
            for c in citations
        ]
        return json.dumps(data, indent=2, ensure_ascii=False)

    elif fmt == "bibtex":
        entries = []
        for c in citations:
            m = c.metadata
            key = c.software_name.lower().replace(" ", "_").replace("-", "_")
            entry = f"""@software{{{key},
  author    = {{{m.authors or 'Unknown'}}},
  title     = {{{m.name}}},
  year      = {{{m.year or 'n.d.'}}},
  version   = {{{m.version or 'unknown'}}},
  publisher = {{{m.publisher or 'unknown'}}},
  doi       = {{{m.doi or ''}}},
  url       = {{{m.url or ''}}}
}}"""
            entries.append(entry)
        return "\n\n".join(entries)

    else:  # text
        lines = ["Software Citations (FORCE11 Compliant)", "=" * 45, ""]
        for i, c in enumerate(citations, 1):
            lines.append(f"{i}. {c.citation_text}")
            if c.completeness_notes:
                lines.append(f"   Note: {c.completeness_notes}")
            lines.append("")
        return "\n".join(lines)


if __name__ == "__main__":
    main()
