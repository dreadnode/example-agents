# Example Agents

This repo contains a variety of example agents to use with the Dreadnode platform.

## Dangerous Capabilities

Based on [research](https://deepmind.google/research/publications/78150/) from Google DeepMind,
this agent works to solve a variety of CTF challenges given access to execute bash commands on
a network-local Kali linux comntainer.

```
uv run -m dangerous_capabilities --help
```

The harness will automatically build all the containers with the supplied flag, and load them
as needed to ensure they are network-isolated from eachother. The process is generally:

1. For each challenge, produce P agent tasks where P = parallelism
2. For all agent tasks, run them in parallel capped at your concurrency setting
3. Inside each task, bring up the associated environment
4. Continue requesting the next command from the inference model - execute it in the `env` container
5. If the flag is ever observed in the output, exit
6. Otherwise run until an error, give up, or max-steps is reached

Check out [./dangerous_capabilities/challenges/challenges.json](./dangerous_capabilities/challenges/challenges.json)
to see all the environments and prompts.

