from typing import Optional

from pydantic import BaseModel, Field

from arxiv_researcher.models.arxiv import ArxivPaper


class ReadingResult(BaseModel):
    """
    Model representing the result of reading and analyzing a research paper
    """

    id: int = Field(default=0, description="Unique ID")
    task: str = Field(default="", description="Research task")
    paper: ArxivPaper = Field(default=None, description="Paper metadata")
    markdown_path: str = Field(
        default="", description="Relative path to the paper's markdown file"
    )
    answer: str = Field(default="", description="Answer generated for the task")
    is_related: Optional[bool] = Field(
        default=None, description="Whether the paper is related to the task"
    )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ReadingResult):
            return self.id == other.id
        return False
