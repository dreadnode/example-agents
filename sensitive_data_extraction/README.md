# Agent: Sensitive Data Extraction

## Description

This agent leverages a Large Language Model (LLM) to autonomously explore and analyze file systems for sensitive data. It is designed to navigate through a given path, read the contents of various files, and identify information such as passwords, API keys, personal identifiable information (PII), and other confidential data. A key feature of this agent is ability to operate on a wide variety of storage systems, including local directories, cloud storage like AWS S3 and Google Cloud Storage, and even remote sources like GitHub repositories (via [fsspec](https://filesystem-spec.readthedocs.io/en/latest/)).


## Environment

The environment is simply a filesystem. The Agent must have the necessary credentials to access the target path specified by the user (e.g., AWS credentials configured for S3 access, or a GitHub token for private repositories).

## Tools

- `fsspec`: The underlying library that provides a unified Pythonic interface to various local and remote file systems. This is what enables the agent's versatility in accessing different storage backends like `s3://`, `gs://`, and `github://`.


## References

- [fsspec](https://github.com/fsspec/fsspec)


## Examples

`uv run main.py --model "" --path ""`