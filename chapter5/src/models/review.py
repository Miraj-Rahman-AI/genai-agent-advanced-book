from pydantic import BaseModel, Field


class Review(BaseModel):
    observation: str = Field(
        title="Objective assessment of execution results",
        description=(
            "First, describe the objective facts regarding the code execution results. "
            "For example: 'Execution completed successfully and produced the result XX.' "
            "or 'An error occurred.' "
            "Then evaluate whether the execution result sufficiently satisfies the user's request at a minimum level. "
            "If the requirements are not met, append a proposed revision or improvement plan."
        ),
    )
    is_completed: bool = Field(
        title="Task completion status",
        description=(
            "Evaluate whether the execution results sufficiently satisfy the user's request at a minimum level. "
            "Set to False if the task requirements are not met. "
            "Set to True if the minimum requirements are satisfied, even if improvements are still possible."
        ),
    )
