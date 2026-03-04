from pydantic import BaseModel, Field


class Section(BaseModel):
    """
    Model representing the structure of a Markdown section
    """

    header: str = Field(description="Section header")
    content: str = Field(description="Section content")
    char_count: int = Field(description="Number of characters in the section")
