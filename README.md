# Example Agents

This repo contains a variety of example agents to use with the Dreadnode platform.

## Setup

All examples share the same project and dependencies, you setup the virtual environment with uv:

```bash
uv sync
```

## Python Agent

A basic agent with access to a dockerized Jupyter kernel to execute code safely.

```bash
uv run -m python_agent --help
```

- Provided a task (`--task`), begin a generation loop with access to the Jupyter kernel
- The work directory (`--work-dir`) is mounted into the container, along with any other docker-style volumes (`--volumes`)
- When finished, the agent markes the task as complete with a status and summary
- The work directory is logged as an artifact for the run

## Dangerous Capabilities

Based on [research](https://deepmind.google/research/publications/78150/) from Google DeepMind,
this agent works to solve a variety of CTF challenges given access to execute bash commands on
a network-local Kali linux container.

```bash
uv run -m dangerous_capabilities --help
```

The harness will automatically build all the containers with the supplied flag, and load them
as needed to ensure they are network-isolated from each other. The process is generally:

1. For each challenge, produce P agent tasks where P = parallelism
2. For all agent tasks, run them in parallel capped at your concurrency setting
3. Inside each task, bring up the associated environment
4. Continue requesting the next command from the inference model - execute it in the `env` container
5. If the flag is ever observed in the output, exit
6. Otherwise run until an error, give up, or max-steps is reached

Check out [./dangerous_capabilities/challenges/challenges.json](./dangerous_capabilities/challenges/challenges.json)
to see all the environments and prompts.

## Dotnet Reversing

This agent is provided access to Cecil and ILSpy for use in reversing
and analyzing Dotnet managed binaries for vulnerabilities.

```bash
uv run -m dotnet_reversing --help
```

You can provide a path containing binaries (recursively), and a target vulnerability term
that you would like the agent to search for. The tool suite provided to the agent includes:

- Search for a term in target modules to identify functions of interest
- Decompile individual methods, types, or entire modules
- Collect all call flows which lead to a target method in all supplied binaries
- Report a vulnerability finding with associated path, method, and description
- Mark a task as complete with a summary
- Give up on a task with a reason

You can also specify the path as a Nuget package identifier and pass `--nuget` to the agent. It
will download the package, extract the binaries, and run the same analysis as above.

```bash
# Local
uv run -m dotnet_reversing --model <model> --path /path/to/local/binaries

# Nuget
uv run -m dotnet_reversing --model <model> --path <nuget-package-id> --nuget
```

## Sensitive Data Extraction

This agent is provided access to a filsystem tool based on [fsspec](https://filesystem-spec.readthedocs.io/en/latest/)
for use in extracting sensitive data stored in files.

```bash
uv run -m sensitive_data_extraction --help
```

The agent is granted some maximum step count to operate tools, query and search files, and provide
reports of any sensitive data it finds. With the help of `fsspec`, the agent can operate on
local files, Github repos, S3 buckets, and other cloud storage systems.

```bash
# Local
uv run -m sensitive_data_extraction --model <model> --path /path/to/local/files

# S3
uv run -m sensitive_data_extraction --model <model> --path s3://bucket

# Azure
uv run -m sensitive_data_extraction --model <model> --path azure://container

# GCS
uv run -m sensitive_data_extraction --model <model> --path gcs://bucket

# Github
uv run -m sensitive_data_extraction --model <model> --path github://owner:repo@/
```

Check out the their docs for more options:
- https://filesystem-spec.readthedocs.io/en/latest/api.html#built-in-implementations
- https://filesystem-spec.readthedocs.io/en/latest/api.html#other-known-implementations

## SAST Vulnerability Scanning

This agent is designed to perform static code analysis to identify security vulnerabilities in source code. It uses a combination of direct file access and container-based approaches to analyze code for common security issues.

```bash
uv run -m sast_scanning --help
```

The agent systematically examines codebases using either direct file access or an isolated container environment. It can:

- Execute targeted analysis commands to search through source files
- Report detailed findings with vulnerability location, type, and severity
- Support various programming languages through configurable extensions
- Operate in two modes: "direct" (filesystem access) or "container" (isolated analysis)
- Challenges and vulnerability patterns are defined in YAML configuration files, allowing for flexible targeting of specific security issues across different codebases.

### Metrics and Scoring

The agent tracks several key metrics to evaluate performance:

- **valid_findings**: Count of correctly identified vulnerabilities matching expected issues
- **raw_findings**: Total number of potential vulnerabilities reported by the model
- **coverage**: Percentage of known vulnerabilities successfully identified
- **duplicates**: Count of repeatedly reported vulnerabilities

Findings are scored using a weighted system that prioritizes matching the correct vulnerability name (3x), function (2x), and line location (1x) to balance semantic accuracy with positional precision.

```bash
# Run in direct mode (default)
uv run -m sast_scanning --model <model> --mode direct

# Run in container mode (isolated environment)
uv run -m sast_scanning --model <model> --mode container

# Run a specific challenge
uv run -m sast_scanning --model <model> --mode container --challenge <challenge-name>

# Customize analysis parameters
uv run -m sast_scanning --model <model> --max-steps 50 --timeout 60
```