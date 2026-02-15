import io

import pandas as pd

from src.llms.utils import load_template


def describe_dataframe(
    file_object: io.BytesIO,
    template_file: str = "src/prompts/describe_dataframe.jinja",
) -> str:
    # Read the CSV file and create a DataFrame
    df = pd.read_csv(file_object)
    # Retrieve summary information of the DataFrame
    buf = io.StringIO()
    df.info(buf=buf)
    df_info = buf.getvalue()
    # Build and return the DataFrame information
    template = load_template(template_file)
    return template.render(
        df_info=df_info,
        df_sample=df.sample(5).to_markdown(),
        df_describe=df.describe().to_markdown(),
    )
