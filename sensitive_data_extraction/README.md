# Agent: Sensitive Data Extraction

## Description

This agent leverages a Large Language Model (LLM) to autonomously explore and analyze file systems for sensitive data. It is designed to navigate through a given path, read the contents of various files, and identify information such as passwords, API keys, personal identifiable information (PII), and other confidential data. A key feature of this agent is its use of the fsspec library, allowing it to operate on a wide variety of storage systems, including local directories, cloud storage like AWS S3 and Google Cloud Storage, and even remote sources like GitHub repositories.

## Intended Use

The Agent is used for performing a thorough search through fileshares and files, then reporting its findings in a structured format, which can then be used for remediation efforts.

## Environment

The environment is simply a filesystem. The Agent must have the necessary credentials to access the target path specified by the user (e.g., AWS credentials configured for S3 access, or a GitHub token for private repositories). For observability, the agent can be connected to a Dreadnode server to log detailed run information, metrics, and findings.

## Tools

- `fsspec`: The underlying library that provides a unified Pythonic interface to various local and remote file systems. This is what enables the agent's versatility in accessing different storage backends like s3://, gs://, and github://.

## Features

- Multi-Filesystem Support: Can analyze files on local disks, AWS S3, Google Cloud Storage, GitHub repositories, and any other backend supported by fsspec.
- LLM-Powered Data Identification: Employs a language model to intelligently parse file contents and identify a broad range of sensitive data types based on context.
- Structured Data Reporting: Uses a dedicated report_sensitive_data tool that forces the LLM to report findings in a structured format, including the file path, location within the file, data type, the sensitive value itself, and a comment.
- Location-Aware Reporting: Can specify the location of findings differently based on the file type (line number for text, seconds for audio/video, or byte offset for binary files).
- Autonomous Exploration: The agent can independently navigate the directory structure of the target path to ensure comprehensive coverage.
- Task Control: Includes tools for the agent to explicitly complete_task with a summary or give_up if it gets stuck, providing better insight into its reasoning process.

## References

- [fsspec](https://github.com/fsspec/fsspec)
