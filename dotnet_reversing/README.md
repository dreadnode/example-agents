# Agent: Dotnet Reversing

## Description

This agent is designed to perform reverse engineering and analysis of .NET binaries. It can decompile .NET assemblies and leverage an LLM to analyze the source code based on a user-defined task, such as identifying security vulnerabilities. The agent can process binaries from a local file path or directly fetch them from the [NuGet package repository](https://www.nuget.org/packages). It operates asynchronously and can run multiple analysis instances in parallel.

## Intended Use

The primary purpose of this agent is to assist security researchers and developers in automating the process of scanning .NET applications for potential security flaws. 

## Environment

It interacts with the public [NuGet API](https://learn.microsoft.com/en-us/nuget/api/overview) (api.nuget.org) to fetch packages, or with local dotnet assemblies.

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

## References

- [ILSpy](https://github.com/icsharpcode/ILSpy)

## Examples

```bash
uv run dotnet_reversing/main.py --model "anthropic/claude-haiku-4-5-20251001" --path ./dotnet_reversing/example_binaries/
```

## Notes

It requires access to dotnet, and for dotnet to be in your path, `export DOTNET_ROOT=~/.dotnet`