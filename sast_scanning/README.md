# Agent: Sast Scanning

## Description

This agent is a specialized Static Application Security Testing (SAST) framework designed to evaluate the capabilities of Large Language Models (LLMs) in identifying security vulnerabilities in source code. It operates by presenting the LLM with a "challenge," a codebase containing known, predefined vulnerabilities. The agent then prompts the model to act as a security expert, analyze the files, and report any security issues it discovers. The agent tracks the findings and scores the model's performance by comparing its results against a manifest of the known vulnerabilities, providing metrics like coverage and accuracy.

## Intended Use

The primary purpose of this agent is to benchmark and compare the effectiveness of different LLMs for security code review tasks. It is intended for researchers and security professionals who want to quantitatively measure a model's ability to detect various types of vulnerabilities (e.g., SQL Injection, XSS, Command Injection) in a controlled and reproducible environment.

## Environment

The agent is a Python command-line application. The agent operates on a local collection of code "challenges" located in the challenges directory. For its container mode, a running Docker daemon is required on the host machine.

## Tools

This harness uses the older style tool calling.

- `ReadFile`
- `Finding`
- `CompleteTask`

## Features

- Challenge-Based Evaluation: Runs security analysis on pre-defined coding challenges, each with a manifest of known vulnerabilities.
- Dual Operation Modes:
  - Direct Mode: The LLM is given a list of files and can request to read them one by one. This tests the model's ability to analyze code when the content is provided directly.
  - Container Mode: The LLM is placed in a sandboxed shell environment with the source code mounted. It must use shell commands (ls, cat, grep, etc.) to explore and analyze the files, testing its tool-use and planning capabilities.
- Automated Scoring: Automatically validates the LLM's reported findings against the ground truth from the challenge manifest, tracking metrics for valid findings, duplicates, and overall coverage.
- Structured Vulnerability Reporting: Defines a clear schema for the LLM to report vulnerabilities, including the vulnerability type, description, file, function, and line number.
- Customizable System Prompts: Allows for easy modification of the system prompt and the addition of suffixes to test how different instructions affect model performance.
- Concurrent Execution: Leverages asyncio to run evaluations for multiple challenges in parallel, speeding up the testing process.

## References

- None
