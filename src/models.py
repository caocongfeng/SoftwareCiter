"""Pydantic data models for the Software Citation Agent."""

from pydantic import BaseModel, Field


class SoftwareMention(BaseModel):
    """A software mention extracted from a paper.

    Contains the raw information found in the publication text.
    Fields may be empty if not mentioned in the paper.
    """

    name: str = Field(description="Software name as mentioned in the paper")
    version: str = Field(default="", description="Version number if mentioned")
    publisher: str = Field(default="", description="Publisher, organization, or author(s) if mentioned")
    url: str = Field(default="", description="URL or link to the software if provided")
    context: str = Field(default="", description="The sentence or passage where the software was mentioned")


class SoftwareMetadata(BaseModel):
    """Enriched software metadata after web search.

    Contains the full set of information needed to build a
    FORCE11-compliant citation.
    """

    name: str = Field(description="Official software name")
    version: str = Field(default="", description="Specific version used/cited")
    authors: str = Field(default="", description="Author(s) or creator(s)")
    year: str = Field(default="", description="Year of release/publication")
    publisher: str = Field(default="", description="Publisher or repository (e.g., Zenodo, CRAN, PyPI)")
    doi: str = Field(default="", description="DOI or other persistent identifier")
    url: str = Field(default="", description="URL to the software or its landing page")
    license: str = Field(default="", description="Software license (e.g., MIT, GPL)")
    description: str = Field(default="", description="Brief description of what the software does")


class SoftwareCitation(BaseModel):
    """A fully formatted software citation following FORCE11 principles."""

    software_name: str = Field(description="Name of the software")
    citation_text: str = Field(description="The formatted citation string")
    metadata: SoftwareMetadata = Field(description="Structured metadata behind the citation")
    completeness_notes: str = Field(
        default="",
        description="Notes about any missing fields or assumptions made",
    )


class PaperSection(BaseModel):
    """A section of a parsed paper."""

    title: str = Field(default="", description="Section title (e.g., 'Methods', 'Abstract')")
    content: str = Field(description="Text content of the section")


class PaperInfo(BaseModel):
    """Parsed paper information."""

    title: str = Field(default="", description="Paper title")
    authors: str = Field(default="", description="Paper authors")
    abstract: str = Field(default="", description="Paper abstract")
    sections: list[PaperSection] = Field(
        default_factory=list,
        description="List of paper sections with their text content",
    )
    source_file: str = Field(default="", description="Original file path")
    file_format: str = Field(default="", description="Detected format (xml, txt, pdf)")

    @property
    def full_text(self) -> str:
        """Return the entire paper text as a single string."""
        parts = []
        if self.abstract:
            parts.append(f"Abstract:\n{self.abstract}")
        for section in self.sections:
            header = f"\n{section.title}:\n" if section.title else "\n"
            parts.append(f"{header}{section.content}")
        return "\n".join(parts)
