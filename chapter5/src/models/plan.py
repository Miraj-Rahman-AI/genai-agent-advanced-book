from pydantic import BaseModel, Field


class Task(BaseModel):
    hypothesis: str = Field(
        title="Hypothesis",
        description=(
            "Describe a verifiable hypothesis in detail along with the reasoning behind it. "
            "The hypothesis should clearly and specifically express the causal relationships, trends, "
            "or expected outcomes that you want to validate through data analysis."
        ),
        examples=[
            "Sales increase on weekends because more people go shopping on Saturdays and Sundays.",
            "Promotional campaigns for new users improve the first-time purchase rate.",
        ],
    )
    purpose: str = Field(
        title="Purpose of Hypothesis Validation",
        description=(
            "Clearly describe the issues or objectives that you aim to clarify by validating this hypothesis. "
            "Explain how validating the hypothesis will contribute to decision-making, business improvement, "
            "or the insights you want to obtain."
        ),
        examples=[
            "Validate differences in sales by day of the week.",
            "Identify the impact of promotional campaigns on purchasing behavior by target segment.",
        ],
    )
    description: str = Field(
        title="Detailed Analysis Method",
        description=(
            "Describe which analytical methods will be used (e.g., univariate analysis, multivariate analysis, "
            "regression analysis, clustering, etc.). "
            "Specify which variables will be used, define function arguments and return values if applicable, "
            "and describe in detail what comparisons or visualizations will be performed."
        ),
    )
    chart_type: str = Field(
        title="Expected Chart Type",
        description="Specify the type of visualization expected.",
        examples=[
            "Histogram",
            "Bar chart",
            "Line chart",
            "Pie chart",
            "Scatter plot",
        ],
    )


class Plan(BaseModel):
    purpose: str = Field(description="The interpreted objective derived from the task request")
    archivement: str = Field(description="The inferred success criteria for achieving the task")
    tasks: list[Task]


class SubTask(BaseModel):
    state: bool
    task: Task
