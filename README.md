# Example Agents

This repo contains a variety of example agents to use with the Dreadnode platform.

## Setup

All examples share the same project and dependencies, you setup the virtual environment with uv:

```bash
export UV_INDEX_DREADNODE_PYPI_PASSWORD=<platform-api-key>

uv sync
```

*`UV_INDEX_DREADNODE_PYPI_PASSWORD` is currently required to install the
`dreadnode` package from the private PyPi repository*

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

# Optional: Pass a GH token
$ gh auth token
uv run -m sensitive_data_extraction --model <model> --path /path/to/local/files --github-token <gh-auth-token>

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
