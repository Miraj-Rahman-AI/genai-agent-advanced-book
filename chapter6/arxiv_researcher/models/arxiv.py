import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ArxivPaper(BaseModel):
    """
    Model representing information about an arXiv paper
    """

    id: str = Field(default="", description="arXiv ID")
    title: str = Field(default="", description="Paper title")
    link: str = Field(default="", description="Paper link")
    pdf_link: str = Field(default="", description="PDF link")
    abstract: str = Field(default="", description="Paper abstract")
    published: datetime.datetime = Field(default=None, description="Publication date")
    updated: datetime.datetime = Field(default=None, description="Last updated date")
    version: int = Field(default=0, description="Version")
    authors: list[str] = Field(default=[], description="Authors")
    categories: list[str] = Field(default=[], description="Categories")
    relevance_score: Optional[float] = Field(default=None, description="Relevance score")

    @property
    def text(self) -> str:
        return f"""\
<paper>
  <id>{self.id}</id>
  <title>{self.title}</title>
  <link>{self.link}</link>
  <pdf_link>{self.pdf_link}</pdf_link>
  <abstract>{self.abstract}</abstract>
  <published>{self.published}</published>
  <updated>{self.updated}</updated>
  <version>{self.version}</version>
  <authors>{', '.join(self.authors)}</authors>
  <categories>{', '.join(self.categories)}</categories>
  {f"<relevance_score>{self.relevance_score}</relevance_score>" if self.relevance_score else ""}
</paper>"""
