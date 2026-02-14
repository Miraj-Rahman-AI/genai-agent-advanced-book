#!/bin/bash

# Display the current working directory to confirm where the setup is being executed.
# This helps ensure the script is being run inside the correct project folder.
echo "Working directory: $(pwd)"

# Check whether the 'uv' package manager is already installed on the system.
# 'uv' is used for Python environment and dependency management in this project.
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."

    # Download and install uv using the official installation script.
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # After installation, suggest adding uv to the system PATH if not already available.
    # This ensures the uv command can be executed globally from the terminal.
    echo "uv has been installed. If necessary, run the following command to add it to your PATH:"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"'

    # Automatically add uv to the current session PATH so it can be used immediately.
    export PATH="$HOME/.cargo/bin:$PATH"
else
    # If uv is already installed, display its location.
    echo "uv is already installed: $(which uv)"
fi

# Create a Python 3.12 virtual environment and install all required dependencies.
# 'uv sync' reads project configuration (pyproject.toml / requirements)
# and installs all necessary packages into the virtual environment.
echo "Creating Python 3.12 virtual environment and installing dependencies..."
uv sync

# Register a Jupyter Notebook kernel for this virtual environment.
# This allows the environment to appear as a selectable kernel inside Jupyter.
echo "Registering Jupyter kernel..."
uv run python -m ipykernel install --user --name genai_ch3 --display-name "Python 3.12 (Chapter 3)"

# Create an environment variable configuration file if it does not already exist.
# The .env file typically contains API keys and sensitive configuration values.
if [ ! -f .env ]; then
    echo "Creating environment variable file (.env) from .env.example..."
    cp .env.example .env
    echo "NOTE: Please update the .env file and set the appropriate API keys."
fi

# Final setup completion message and usage instructions.
echo "Environment setup is complete!"
echo "You can activate the virtual environment using: source .venv/bin/activate"
echo "Or run commands directly using uv: uv run python your_script.py"
echo "To start Jupyter Notebook, run: uv run jupyter notebook"
