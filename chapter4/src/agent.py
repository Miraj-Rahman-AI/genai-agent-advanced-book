import operator
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.configs import Settings
from src.custom_logger import setup_logger
from src.models import (
    AgentResult,
    Plan,
    ReflectionResult,
    SearchOutput,
    Subtask,
    ToolResult,
)
from src.prompts import HelpDeskAgentPrompts

# Maximum number of retry/refinement cycles allowed per subtask.
# Each cycle consists of: tool selection -> tool execution -> subtask answer -> reflection.
MAX_CHALLENGE_COUNT = 3

# Initialize a module-level logger to record agent execution details.
logger = setup_logger(__file__)


class AgentState(TypedDict):
    """
    Main (top-level) state for the agent graph.

    Fields:
        question: The original user question.
        plan: A list of subtasks produced by the planning step.
        current_step: The current subtask index being executed.
        subtask_results: Aggregated results across all executed subtasks.
        last_answer: The final composed answer produced at the end of the workflow.
    """

    question: str
    plan: list[str]
    current_step: int
    subtask_results: Annotated[Sequence[Subtask], operator.add]
    last_answer: str


class AgentSubGraphState(TypedDict):
    """
    State used inside the subgraph that executes a single subtask.

    Fields:
        question: The original user question (passed through for context).
        plan: The full subtask plan for reference/context.
        subtask: The specific subtask currently being executed.
        is_completed: Whether this subtask has been satisfactorily solved.
        messages: The chat message history used for tool selection and answer generation.
        challenge_count: How many refinement attempts have been made for this subtask.
        tool_results: Aggregated tool outputs across attempts.
        reflection_results: Aggregated reflection outputs across attempts.
        subtask_answer: The latest natural-language answer for this subtask.
    """

    question: str
    plan: list[str]
    subtask: str
    is_completed: bool
    messages: list[ChatCompletionMessageParam]
    challenge_count: int
    tool_results: Annotated[Sequence[Sequence[SearchOutput]], operator.add]
    reflection_results: Annotated[Sequence[ReflectionResult], operator.add]
    subtask_answer: str


class HelpDeskAgent:
    """
    Help-desk style agent for the XYZ system.

    High-level workflow:
        1) Create a plan (list of subtasks) from the user's question.
        2) For each subtask, run a subgraph loop:
            - Select tools
            - Execute tools
            - Draft a subtask answer
            - Reflect/evaluate the answer and optionally retry
        3) Compose a final answer from all subtask answers.

    Notes:
        - Tools are LangChain tools passed into the agent at initialization.
        - Reflection is used as a quality gate; if reflection returns NG, the agent retries.
    """

    def __init__(
        self,
        settings: Settings,
        tools: list = [],
        prompts: HelpDeskAgentPrompts = HelpDeskAgentPrompts(),
    ) -> None:
        # Store application settings (API keys, model name, etc.).
        self.settings = settings

        # Store tool instances used by the agent for retrieval/search.
        self.tools = tools

        # Build a fast lookup map from tool name -> tool object for execution.
        self.tool_map = {tool.name: tool for tool in tools}

        # Store prompt templates used by the planner/subtask/final answer stages.
        self.prompts = prompts

        # Initialize an OpenAI client for all LLM calls in this agent.
        self.client = OpenAI(api_key=self.settings.openai_api_key)

    def create_plan(self, state: AgentState) -> dict:
        """
        Generate a subtask plan for answering the user's question.

        This step uses an LLM prompt (planner system + planner user prompt) and
        parses the response into a structured Plan object. The output plan is
        returned as a list of subtask strings.

        Args:
            state: The current top-level agent state containing the user question.

        Returns:
            A dict containing the generated plan, e.g. {"plan": [...]}
        """

        logger.info("🚀 Starting plan generation process...")

        # Build the system prompt for planning.
        system_prompt = self.prompts.planner_system_prompt

        # Format the user prompt by injecting the user's question.
        user_prompt = self.prompts.planner_user_prompt.format(
            question=state["question"],
        )

        # Construct the message list for the OpenAI request.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        logger.debug(f"Final prompt messages: {messages}")

        # Call OpenAI with Structured Outputs parsing into the Plan model.
        try:
            logger.info("Sending request to OpenAI...")
            response = self.client.beta.chat.completions.parse(
                model=self.settings.openai_model,
                messages=messages,
                response_format=Plan,
                temperature=0,
                seed=0,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        # Extract the parsed structured output (Plan) from the response.
        plan = response.choices[0].message.parsed

        logger.info("Plan generation complete!")

        # Return the plan subtasks to update the state in the main graph.
        return {"plan": plan.subtasks}

    def select_tools(self, state: AgentSubGraphState) -> dict:
        """
        Select which tools should be executed to solve the current subtask.

        Behavior:
            - Converts the available LangChain tools into OpenAI tool schema.
            - If this is the first attempt (challenge_count == 0), creates a fresh prompt.
            - If this is a retry, reuses previous messages, trims tool output messages
              (to reduce tokens), and appends a retry instruction.

        Args:
            state: The current subgraph state including question/plan/subtask and history.

        Returns:
            A dict containing an updated "messages" list that includes the tool_calls
            produced by the model.
        """

        logger.info("🚀 Starting tool selection process...")

        # Convert LangChain tool definitions to OpenAI-compatible tool definitions.
        logger.debug("Converting tools for OpenAI format...")
        openai_tools = [convert_to_openai_tool(tool) for tool in self.tools]

        # If this is the first attempt, use the standard tool-selection prompt.
        if state["challenge_count"] == 0:
            logger.debug("Creating user prompt for tool selection...")

            user_prompt = self.prompts.subtask_tool_selection_user_prompt.format(
                question=state["question"],
                plan=state["plan"],
                subtask=state["subtask"],
            )

            messages = [
                {"role": "system", "content": self.prompts.subtask_system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        # If this is a retry, reuse message history and append a retry instruction.
        else:
            logger.debug("Creating user prompt for tool retry...")

            # Reuse previous conversation context.
            messages: list = state["messages"]

            # Token optimization:
            # Remove tool-output messages (and any tool_call artifacts) so we keep
            # reasoning context but drop bulky retrieval payloads.
            messages = [
                message
                for message in messages
                if message["role"] != "tool" or "tool_calls" not in message
            ]

            # Add a retry instruction based on the reflection feedback loop design.
            user_retry_prompt = self.prompts.subtask_retry_answer_user_prompt
            messages.append({"role": "user", "content": user_retry_prompt})

        # Ask the model to decide which tool(s) to call for this subtask.
        try:
            logger.info("Sending request to OpenAI...")
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                tools=openai_tools,  # type: ignore
                temperature=0,
                seed=0,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        # Validate tool calls exist; this workflow requires the model to choose tools.
        if response.choices[0].message.tool_calls is None:
            raise ValueError("Tool calls are None")

        # Store the tool calls in a message format compatible with downstream execution.
        ai_message = {
            "role": "assistant",
            "tool_calls": [
                tool_call.model_dump()
                for tool_call in response.choices[0].message.tool_calls
            ],
        }

        logger.info("Tool selection complete!")
        messages.append(ai_message)

        # Return the updated message history (subgraph state will be updated with this).
        return {"messages": messages}

    def execute_tools(self, state: AgentSubGraphState) -> dict:
        """
        Execute the tool calls generated in the tool-selection step.

        This function:
            - Reads tool calls from the last assistant message.
            - Looks up the matching tool in self.tool_map.
            - Executes each tool with the provided arguments.
            - Appends tool responses back into the messages list in "tool" role format.
            - Returns structured ToolResult objects for traceability.

        Args:
            state: The current subgraph state containing tool_calls in messages.

        Raises:
            ValueError: If tool_calls are missing (None).

        Returns:
            A dict containing updated "messages" and a list of ToolResult objects
            under "tool_results".
        """

        logger.info("🚀 Starting tool execution process...")
        messages = state["messages"]

        # Extract tool calls from the last assistant message.
        tool_calls = messages[-1]["tool_calls"]

        # Ensure tool calls exist; otherwise execution cannot proceed.
        if tool_calls is None:
            logger.error("Tool calls are None")
            logger.error(f"Messages: {messages}")
            raise ValueError("Tool calls are None")

        tool_results = []

        # Execute each tool call and append results to message history.
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            # Find the tool implementation by name.
            tool = self.tool_map[tool_name]

            # Execute tool and get the list of SearchOutput results.
            tool_result: list[SearchOutput] = tool.invoke(tool_args)

            # Store a structured record of the tool execution.
            tool_results.append(
                ToolResult(
                    tool_name=tool_name,
                    args=tool_args,
                    results=tool_result,
                )
            )

            # Append tool output to the conversation in OpenAI tool-message format.
            # The tool_call_id links this output back to the specific tool request.
            messages.append(
                {
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call["id"],
                }
            )

        logger.info("Tool execution complete!")

        # Wrap tool_results in a list-of-lists to support multi-attempt aggregation.
        return {"messages": messages, "tool_results": [tool_results]}

    def create_subtask_answer(self, state: AgentSubGraphState) -> dict:
        """
        Draft a natural-language answer for the current subtask.

        This function calls the LLM using the current messages (which include tool outputs),
        then appends the assistant's answer to the message history.

        Args:
            state: The current subgraph state including tool outputs in messages.

        Returns:
            A dict containing updated "messages" and "subtask_answer".
        """

        logger.info("🚀 Starting subtask answer creation process...")
        messages = state["messages"]

        # Request the model to generate a subtask-level answer based on tool outputs.
        try:
            logger.info("Sending request to OpenAI...")
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                temperature=0,
                seed=0,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        # Extract the generated subtask answer text.
        subtask_answer = response.choices[0].message.content

        # Append assistant answer back into the message history.
        messages.append({"role": "assistant", "content": subtask_answer})

        logger.info("Subtask answer creation complete!")

        return {"messages": messages, "subtask_answer": subtask_answer}

    def reflect_subtask(self, state: AgentSubGraphState) -> dict:
        """
        Reflect on (evaluate) the quality and completeness of the subtask answer.

        The reflection step:
            - Adds a reflection prompt to the messages.
            - Calls the model using Structured Outputs into ReflectionResult.
            - Updates state with reflection results and increments challenge_count.
            - Marks is_completed based on reflection_result.is_completed.
            - If max retries are exceeded and still incomplete, sets a fallback answer.

        Args:
            state: The current subgraph state including the latest subtask answer.

        Raises:
            ValueError: If the parsed reflection result is missing (None).

        Returns:
            A dict containing updated fields:
                - messages
                - reflection_results
                - challenge_count
                - is_completed
                - (optional) subtask_answer fallback if retries exhausted
        """

        logger.info("🚀 Starting reflection process...")
        messages = state["messages"]

        # Add reflection instruction to trigger evaluation.
        user_prompt = self.prompts.subtask_reflection_user_prompt
        messages.append({"role": "user", "content": user_prompt})

        # Call OpenAI and parse reflection output into ReflectionResult.
        try:
            logger.info("Sending request to OpenAI...")
            response = self.client.beta.chat.completions.parse(
                model=self.settings.openai_model,
                messages=messages,
                response_format=ReflectionResult,
                temperature=0,
                seed=0,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        reflection_result = response.choices[0].message.parsed
        if reflection_result is None:
            raise ValueError("Reflection result is None")

        # Append the reflection result to message history for traceability/audit.
        messages.append({"role": "assistant", "content": reflection_result.model_dump_json()})

        # Prepare the subgraph state update.
        update_state = {
            "messages": messages,
            "reflection_results": [reflection_result],
            "challenge_count": state["challenge_count"] + 1,
            "is_completed": reflection_result.is_completed,
        }

        # If maximum retries are reached and the subtask is still incomplete,
        # provide a fallback message indicating the answer could not be found.
        if (
            update_state["challenge_count"] >= MAX_CHALLENGE_COUNT
            and not reflection_result.is_completed
        ):
            update_state["subtask_answer"] = f"Could not find an answer for: {state['subtask']}."

        logger.info("Reflection complete!")
        return update_state

    def create_answer(self, state: AgentState) -> dict:
        """
        Compose the final answer to the user using all subtask answers.

        This function:
            - Extracts (subtask_name, subtask_answer) pairs from the subtask results.
            - Calls the LLM with a "final answer" system prompt and a user prompt containing
              the original question, plan, and summarized subtask outputs.
            - Returns the final response text as "last_answer".

        Args:
            state: The top-level agent state containing all completed subtask results.

        Returns:
            A dict containing {"last_answer": "..."} to update the main graph state.
        """

        logger.info("🚀 Starting final answer creation process...")

        # System prompt instructing how to compose a user-facing help-desk answer.
        system_prompt = self.prompts.create_last_answer_system_prompt

        # Extract only the subtask name and its final answer (omit tool payloads).
        subtask_results = [
            (result.task_name, result.subtask_answer)
            for result in state["subtask_results"]
        ]

        # Inject question, plan, and subtask results into the final-answer user prompt.
        user_prompt = self.prompts.create_last_answer_user_prompt.format(
            question=state["question"],
            plan=state["plan"],
            subtask_results=str(subtask_results),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Request the final composed response from the model.
        try:
            logger.info("Sending request to OpenAI...")
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                temperature=0,
                seed=0,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        logger.info("Final answer creation complete!")

        return {"last_answer": response.choices[0].message.content}

    def _execute_subgraph(self, state: AgentState):
        """
        Execute the subgraph for one subtask and return a Subtask object.

        This method:
            - Builds the subgraph workflow.
            - Invokes it with the current subtask.
            - Converts the subgraph outputs into a Subtask model.
            - Returns a dict that can be merged into the main agent state.

        Args:
            state: The current top-level agent state.

        Returns:
            dict: {"subtask_results": [Subtask(...)]}
        """

        # Create a fresh subgraph instance (planner/execute/reflect loop).
        subgraph = self._create_subgraph()

        # Invoke the subgraph for the current subtask.
        result = subgraph.invoke(
            {
                "question": state["question"],
                "plan": state["plan"],
                "subtask": state["plan"][state["current_step"]],
                "current_step": state["current_step"],
                "is_completed": False,
                "challenge_count": 0,
            }
        )

        # Wrap subgraph outputs in a structured Subtask model for storage.
        subtask_result = Subtask(
            task_name=result["subtask"],
            tool_results=result["tool_results"],
            reflection_results=result["reflection_results"],
            is_completed=result["is_completed"],
            subtask_answer=result["subtask_answer"],
            challenge_count=result["challenge_count"],
        )

        # Return as a list to support aggregation across multiple subtasks.
        return {"subtask_results": [subtask_result]}

    def _should_continue_exec_subtasks(self, state: AgentState) -> list:
        """
        Create dispatch instructions for executing each subtask.

        LangGraph uses Send(...) objects to fan out work. This method generates a
        list of Send messages that instruct the graph to run 'execute_subtasks'
        once for each subtask index.

        Args:
            state: The main agent state containing the plan.

        Returns:
            list: A list of Send(...) objects, one per subtask.
        """
        return [
            Send(
                "execute_subtasks",
                {
                    "question": state["question"],
                    "plan": state["plan"],
                    "current_step": idx,
                },
            )
            for idx, _ in enumerate(state["plan"])
        ]

    def _should_continue_exec_subtask_flow(
        self, state: AgentSubGraphState
    ) -> Literal["end", "continue"]:
        """
        Determine whether the subtask loop should continue or terminate.

        Stop conditions:
            - The reflection marked the subtask as completed (is_completed == True), OR
            - The number of attempts reached MAX_CHALLENGE_COUNT.

        Args:
            state: The current subgraph state.

        Returns:
            "end" to stop, "continue" to retry tool-selection and execution.
        """
        if state["is_completed"] or state["challenge_count"] >= MAX_CHALLENGE_COUNT:
            return "end"
        return "continue"

    def _create_subgraph(self) -> Pregel:
        """
        Build and compile the subgraph workflow for executing a single subtask.

        Node sequence:
            START -> select_tools -> execute_tools -> create_subtask_answer -> reflect_subtask
            Then conditional loop:
                - if "continue": go back to select_tools
                - if "end": terminate

        Returns:
            Pregel: A compiled LangGraph application representing the subtask loop.
        """

        # Initialize a LangGraph state machine for the subtask execution workflow.
        workflow = StateGraph(AgentSubGraphState)

        # Register each step as a node in the subgraph.
        workflow.add_node("select_tools", self.select_tools)                # Choose tools
        workflow.add_node("execute_tools", self.execute_tools)              # Run tools
        workflow.add_node("create_subtask_answer", self.create_subtask_answer)  # Draft answer
        workflow.add_node("reflect_subtask", self.reflect_subtask)          # Evaluate answer

        # Define the start edge.
        workflow.add_edge(START, "select_tools")

        # Define the main linear edges.
        workflow.add_edge("select_tools", "execute_tools")
        workflow.add_edge("execute_tools", "create_subtask_answer")
        workflow.add_edge("create_subtask_answer", "reflect_subtask")

        # Add conditional edges to either retry or finish based on reflection.
        workflow.add_conditional_edges(
            "reflect_subtask",
            self._should_continue_exec_subtask_flow,
            {"continue": "select_tools", "end": END},
        )

        # Compile the workflow into an executable Pregel graph.
        return workflow.compile()

    def create_graph(self) -> Pregel:
        """
        Build and compile the main agent graph.

        Main graph flow:
            START -> create_plan -> (fan out execute_subtasks per subtask) -> create_answer -> END

        Returns:
            Pregel: A compiled LangGraph application representing the full agent workflow.
        """

        # Initialize the top-level LangGraph state machine.
        workflow = StateGraph(AgentState)

        # Register top-level nodes.
        workflow.add_node("create_plan", self.create_plan)              # Plan generation
        workflow.add_node("execute_subtasks", self._execute_subgraph)   # Subtask execution
        workflow.add_node("create_answer", self.create_answer)          # Final composition

        # Start by generating the plan.
        workflow.add_edge(START, "create_plan")

        # After planning, dispatch subtask execution across all subtasks.
        workflow.add_conditional_edges(
            "create_plan",
            self._should_continue_exec_subtasks,
        )

        # After subtask executions, create the final user-facing answer.
        workflow.add_edge("execute_subtasks", "create_answer")

        # Declare the finish point of the workflow.
        workflow.set_finish_point("create_answer")

        # Compile into a runnable graph.
        return workflow.compile()

    def run_agent(self, question: str) -> AgentResult:
        """
        Run the full agent workflow end-to-end for a given user question.

        This method:
            - Creates the compiled main graph.
            - Invokes it with the user's question and initial step index.
            - Wraps the output into a structured AgentResult object.

        Args:
            question: The user's input question.

        Returns:
            AgentResult: The structured result containing plan, subtask traces, and final answer.
        """

        # Build the full agent graph (plan -> subtasks -> final answer).
        app = self.create_graph()

        # Invoke the graph with initial state.
        result = app.invoke({"question": question, "current_step": 0})

        # Convert raw outputs into the typed AgentResult model.
        return AgentResult(
            question=question,
            plan=Plan(subtasks=result["plan"]),
            subtasks=result["subtask_results"],
            answer=result["last_answer"],
        )
