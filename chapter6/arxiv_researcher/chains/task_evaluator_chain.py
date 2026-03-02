from datetime import datetime
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from arxiv_researcher.chains.utils import dict_to_xml_str, load_prompt


class TaskEvaluation(BaseModel):
    need_more_information: bool = Field(
        default=True,
        description="Set to False if the required information is sufficient",
    )
    reason: str = Field(
        default="",
        description="Briefly describe the reason for the evaluation in Japanese",
    )
    content: str = Field(
        default="",
        description="Describe in detail (in Japanese) what additional investigation is needed",
    )


class TaskEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def __call__(
        self, state: dict
    ) -> Command[Literal["decompose_query", "generate_report"]]:
        current_retry_count = state.get("retry_count", 0)

        evaluation: TaskEvaluation = self.run(
            context="\n".join(
                [
                    dict_to_xml_str(item.model_dump(), exclude_keys=["markdown_text"])
                    for item in state["reading_results"]
                ]
            ),
            goal_setting=state["goal"],
        )

        if evaluation.need_more_information:
            current_retry_count += 1

        next_node = (
            "decompose_query" if evaluation.need_more_information else "generate_report"
        )
        return Command(
            goto=next_node,
            update={
                "retry_count": current_retry_count,
                "evaluation": evaluation,
            },
        )

    def run(self, context: str, goal_setting: str) -> TaskEvaluation:
        prompt = ChatPromptTemplate.from_template(load_prompt("task_evaluator"))
        chain = prompt | self.llm.with_structured_output(
            TaskEvaluation,
            method="function_calling",
        )
        return chain.invoke(
            {
                "current_date": self.current_date,
                "context": context,
                "goal_setting": goal_setting,
            }
        )
