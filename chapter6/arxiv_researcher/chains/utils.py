from pathlib import Path


def load_prompt(name: str) -> str:
    """Load a prompt file.

    Args:
        name (str): Prompt file name (without the .prompt extension)

    Returns:
        str: The content of the prompt
    """
    prompt_path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return prompt_path.read_text().strip()


def dict_to_xml_str(data: dict, exclude_keys: list[str] = []) -> str:
    xml_str = "<item>"
    for key, value in data.items():
        if key not in exclude_keys:
            xml_str += f"<{key}>{value}</{key}>"
    xml_str += "</item>"
    return xml_str
