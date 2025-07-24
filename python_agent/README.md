# Agent: Python Agent

## Description

This agent provides a general-purpose, sandboxed environment for executing Python code to accomplish user-defined tasks. It leverages a Large Language Model (LLM) to interpret a natural language task, generate Python code, and execute it within a Docker container. The agent operates by creating an interactive session with a [Jupyter kernel](https://docs.jupyter.org/en/latest/projects/kernels.html) running inside the container, allowing it to iteratively write code, execute it, and use the output to inform its next steps until the task is complete.

## Intended Use

The agent is designed for a wide range of tasks that can be solved programmatically with Python.

## Environment

To run this agent, a Docker daemon must be available and running on the host machine. The agent itself is a Python command-line application. It pulls a specified Docker image (defaulting to [jupyter/datascience-notebook:latest](https://hub.docker.com/r/jupyter/datascience-notebook/)) to create the execution environment.

## Tools

- `execute_code`
- `restart_kernel`
- `complete_task`

## Features

- **Sandboxed Execution**: All code is executed within a secure and isolated Docker container, preventing unintended side effects on the host machine.
- **Customizable Environment**: Users can specify any Docker image for the execution environment and mount local directories as volumes into the container.
- **LLM-Powered Task Resolution**: The agent takes a high-level, natural language task and intelligently generates and executes the code needed to complete it.
- **Interactive Code Execution**: Provides tools for the LLM to `execute_code` and `restart_kernel`, allowing for an interactive and stateful problem-solving process.
- **Task Completion Reporting**: The agent can explicitly mark a task as complete with a success or failure status and a final summary.
- **Step-by-Step Iteration**: The agent operates within a defined loop with a maximum number of steps (max_steps) to ensure termination.
- **Artifact Logging**: Upon completion, the agent can log the entire working directory as an artifact to Dreadnode, preserving any generated files.

## References

- None
