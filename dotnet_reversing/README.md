# Agent: Dotnet Reversing

## Description

This agent is designed to perform reverse engineering and analysis of .NET binaries. It can decompile .NET assemblies and leverage a large language model (LLM) to analyze the source code based on a user-defined task, such as identifying security vulnerabilities. The agent can process binaries from a local file path or directly fetch them from the NuGet package repository. It operates asynchronously and can run multiple analysis instances in parallel.

## Intended Use

The primary purpose of this agent is to assist security researchers and developers in automating the process of scanning .NET applications for potential security flaws. A user can provide a high-level task, like "Find only critical vulnerabilities," and the agent will use its tools to decompile the code and use an LLM to analyze it, reporting any findings. It can also be used as a simple utility to decompile and view the source code of .NET assemblies.

## Environment

The agent is a command-line application built with Python. It requires a Python environment with the necessary libraries installed, as specified in the script. It interacts with the public NuGet API (api.nuget.org) to fetch packages. For its analysis capabilities, it relies on a configured language model, which can be a remote API (like GPT-4o-mini) or a locally hosted model (e.g., via Ollama). For observability and task tracking, it can be optionally connected to a Dreadnode server.

## Tools

- `decompile_module`
- `decompile_type`
- `decompile_methods`
- `list_namespaces`
- `list_types_in_namespace`
- `list_methods_in_type`
- `list_types`
- `list_methods`
- `search_for_references`
- `get_call_flows_to_method`

## Features

- Multi-Source Analysis: Capable of analyzing .NET binaries from local paths, directories, or directly from NuGet packages.
- LLM-Powered Analysis: Utilizes a configurable language model to intelligently analyze decompiled source code based on a custom task.
- Vulnerability Reporting: Can identify and report findings, classifying them by criticality (critical, high, medium, low, info).
- Concurrent Execution: Supports running multiple agent instances in parallel to speed up the analysis of many binaries.
- Source Code Dumping: Includes a utility to decompile and save the source code of specified binaries to a text file.
- Iterative Analysis: Performs analysis in an iterative loop, with a configurable maximum number of steps to prevent infinite runs.
- Task Completion Summary: Provides a final summary upon task completion, indicating success or failure and a brief markdown report.

## References

- [ILSpy](https://github.com/icsharpcode/ILSpy)
