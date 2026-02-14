# Chapter 5: Supporting Data Analysts
Source code used in the book "A Practical Introduction to AI Agents for Field Use" (Kodansha)

* This is a paid service, as code execution is performed using the E2B Sandbox and text generation is performed using the OpenAI API.

## Preparation

Select `chapter5` in the VSCode Add Terminal and open a workspace.

* For instructions on how to open a workspace, see [/README.md#How to Use This Repository](/README.md#How to Use This Repository).

### Creating a Python Virtual Environment and Installing Dependencies

Installing Dependencies

```bash
uv sync
```

Activate the virtual environment created after installation.

```bash
source .venv/bin/activate
```

### Setting Environment Variables

Also, set environment variables in the `.env` file.

See the next subsection for instructions on how to obtain each key.

```
E2B_API_KEY=e2b_***
OPENAI_API_KEY=sk-proj-***
```

<details><summary>Setting the E2B_API_KEY (click to expand)</summary>

The code execution environment uses [E2B](https://e2b.dev/), a sandbox cloud environment designed for AI applications.

Access e2b in advance at [https://e2b.dev/sign-in](https://e2b.dev/sign-in) and obtain an API key. After obtaining it, store the value in the `E2B_API_KEY` parameter in your `.env` file.

<img src="https://i.gyazo.com/7a54ad6d72beaa6e47fad1f9e65ab69d.png">

</details>

<details><summary>Setting the OPENAI_API_KEY (click to expand)</summary>

Text generation uses the OpenAI API.

Access the OpenAI Platform at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) in advance to obtain an API key. After obtaining it, store the value in the `OPENAI_API_KEY` parameter in your `.env` file.

<img src="https://i.gyazo.com/bdbe5fd77930add697f134cd153411c7.png">
</details></br>

## Directory Structure

```tree
chapter5
├── data # Sample data
│ └── sample.csv
├── scripts # src Run the sample files below
│ └── ...
├── src
│ ├── models
│ │ ├── data_thread.py # Code generation and execution. Review Data Types
│ │ ├── program.py # 5.4.1. Code Generation
│ │ ├── review.py # 5.4.3. Reviewing Execution Results
│ │ └── plan.py # 5.5.1. Planning
│ ├── modules
│ │ ├── describe_dataframe.py # 5.3.2. Dataset Overview
│ │ ├── generate_code.py # 5.4.1. Code Generation
│ │ ├── set_dataframe.py # 5.4.2. Uploading Data to the Sandbox
│ │ ├── execute_code.py # 5.4.2. Code Execution
│ │ ├── generate_review.py # 5.4.3. Reviewing Execution Results
│ │ ├── generate_plan.py # 5.5.1. Planning
│ │ └── generate_report.py # 5.5.3. Report Generation
│ ├── prompts
│ │ └── ...
│ └── llms
│ ├── apis
│ │ └── openai.py # OpenAI API
│ ├── models
│ │ └── llm_response.py # Output data type from LLM API
│ └── utils
│ └── load_prompt.py # Load template file
├── .env # Environment variable definition
├── main.py # Data analysis agent execution script
└── pyproject.toml # Dependency management
```

## 5.3 Implementation Preparation

### 5.3.1 Setting up the execution environment for the generated code

First, run the code in the E2B Sandbox environment.

```bash
uv run python scripts/01_e2b_sandbox.py
```

Here, the `print` function is executed in the E2B Sandbox environment.

``python
from e2b_code_interpreter import Sandbox

with Sandbox() as sandbox:
execution = sandbox.run_code("print('Hello World!')")
logger.info("\n".join(execution.logs.stdout))
```

### 5.3.2 Overview of the Dataset to be Analyzed

#### How to Use the Jinja2 Template Engine

We will build a prompt using [Jinja2](https://jinja.palletsprojects.com/en/stable/).

```bash
uv run python scripts/02_jinja_template.py
```

Here, you can see how the prompt changes depending on the variable corresponding to the placeholder using a placeholder template.

``python
from jinja2 import Template

source = """{% if message %}Message is here: {{ message }}{% endif %}"""
template = Template(source=source)
print("1.", template.render(message="hello"))
print("2.", template.render())
```

Templates can also be written as `.jinja` files (see [`load_template`](/chapter5/src/llms/utils/load_template.py)).

#### Checking the Data Overview

To provide LLM with the data information to be analyzed, we use [pandas](https://pandas.pydata.org/) to write the data information.

```bash
uv run python scripts/03_describe_dataframe.py
```

```python
# src/prompts/describe_dataframe.jinja

'''python
>>> df.info()
{{ df_info }}

>>> df.sample(5).to_markdown()
{{ df_sample }}

>>> df.describe()
{{ df_describe }}
'''
```

### 5.3.3 Calling LLM

Generate a profile by calling LLM using the OpenAI API and executing a function that returns the data type [`LLMResponse`](/chapter5/src/llms/models/llm_response.py).

```bash
uv run python scripts/04_generate_profile.py
```

This is an excerpt, but it calls the predefined ([`generate_response`](/chapter5/src/llms/apis/openai.py)) function as follows:

```python
prompt_template = Template(source=PROMPT)
message = prompt_template.render(role=role)
response: LLMResponse = openai.generate_response(
[{"role": "user", "content": message}],
model=model,
)
```

## 5.4 Single-Agent Workflow for Program Generation

In Section 5.4, we will build a single-agent workflow that performs (1) code generation, (2) code execution, and (3) review of execution results.

<img src="https://i.gyazo.com/0c2893618ab7513cabc5387073e4d6b6.png">

To manage the coding process, we will manage the results of steps (1) through (3) as [`DataThread`](/chapter5/src/models/data_thread.py).

```python
class DataThread(BaseModel):
process_id: str
thread_id: int
user_request: str | None
code: str | None = None
error: str | None = None
stderr: str | None = None
stdout: str | None = None
is_completed: bool = False
observation: str | None = None
results: list[dict] = Field(default_factory=list)
pathes: dict = Field(default_factory=dict)
```

### 5.4.1 Code Generation (Planning)

Generate code that meets the task requirements and the data to be analyzed. In this example, the request is "Please provide an overview of the data."

``bash
uv run python scripts/05_generate_code.py
```
