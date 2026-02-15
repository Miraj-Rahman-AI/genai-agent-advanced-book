from src.llms.apis import openai
from src.llms.models import LLMResponse
from src.llms.utils import load_template
from src.models import DataThread, Program


def generate_code(
    data_info: str,
    user_request: str,
    remote_save_dir: str = "outputs/process_id/id",
    previous_thread: DataThread | None = None,
    model: str = "gpt-4o-mini-2024-07-18",
    template_file: str = "src/prompts/generate_code.jinja",
) -> LLMResponse:
    template = load_template(template_file)
    system_message = template.render(
        data_info=data_info,
        remote_save_dir=remote_save_dir,
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Task request: {user_request}"},
    ]
    # Self-correction: incorporate review results if available (see Section 5.4.3)
    if previous_thread:
        # Add the code from the previous thread
        messages.append({"role": "assistant", "content": previous_thread.code})
        # Add stdout and stderr from the previous thread
        if previous_thread.stdout and previous_thread.stderr:
            messages.extend(
                [
                    {"role": "system", "content": f"stdout: {previous_thread.stdout}"},
                    {"role": "system", "content": f"stderr: {previous_thread.stderr}"},
                ],
            )
        # Add observation results from the previous thread
        if previous_thread.observation:
            messages.append(
                {
                    "role": "user",
                    "content": f"Refer to the following and regenerate code that satisfies the user requirements: {previous_thread.observation}",
                },
            )
    return openai.generate_response(
        messages,
        model=model,
        response_format=Program,
    )
