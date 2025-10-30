# Agent: Python Agent

## Description

This agent provides a general-purpose, sandboxed environment for executing Python code to accomplish user-defined tasks. It leverages a Large Language Model (LLM) to interpret a natural language task, generate Python code, and execute it within a Docker container. The agent operates by creating an interactive session with a [Jupyter kernel](https://docs.jupyter.org/en/latest/projects/kernels.html) running inside the container, allowing it to iteratively write code, execute it, and use the output to inform its next steps until the task is complete.


## Environment

To run this agent, a Docker daemon must be available and running on the host machine. The agent itself is a Python command-line application. It pulls a specified Docker image (defaulting to [jupyter/datascience-notebook:latest](https://hub.docker.com/r/jupyter/datascience-notebook/)) to create the execution environment.

## Tools

- `execute_code`
- `restart_kernel`
- `complete_task`


## Examples

`uv run python_agent/main.py --model "anthropic/claude-haiku-4-5-20251001" --task "please generate an interesting dataset, and visualize it"`
