import typing as t
from dataclasses import dataclass, field
from textwrap import dedent

import cyclopts
import dreadnode as dn
from dreadnode.agent import Agent
from dreadnode.agent.tools import tool
from dreadnode.agent.tools.fs import Filesystem
from dreadnode.data_types import Markdown
from rich.console import Console

console = Console()

# CLI

app = cyclopts.App()


@cyclopts.Parameter(name="*", group="args")
@dataclass
class Args:
    model: str
    """Model to use for inference"""
    path: str
    """Path uri to analyze - supports fsspec paths (e.g. s3://, gs://, github://, etc.)"""
    max_steps: int = 50
    """Maximum number of iterations per agent"""
    fs: dict[str, str] = field(default_factory=dict)
    """Options for the fsspec filesystem (e.g. `fs-options.anon true`)"""
    log_level: str = "INFO"
    """Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"""


@cyclopts.Parameter(name="*", group="dreadnode")
@dataclass
class DreadnodeArgs:
    server: str | None = None
    """Dreadnode server URL"""
    token: str | None = None
    """Dreadnode API token"""
    project: str = "sensitive-data-extraction"
    """Dreadnode project name"""
    console: t.Annotated[bool, cyclopts.Parameter(negative=False)] = False
    """Show span information in the console"""


@tool()
async def report_sensitive_data(
    path: t.Annotated[str, "The originating file"],
    location: t.Annotated[int, "Location of data inside the file (line number or seconds)"],
    type: t.Annotated[str, "Type of sensitive data"],
    value: t.Annotated[str, "The sensitive data itself"],
    comment: t.Annotated[str, "Comment about the data"],
) -> str:
    """
    Report sensitive data found in a file.

    When reporting location, follow these guidelines:
    - For text files and JSON files: Use the line number
    - For audio/video files: Use the number of seconds
    - For binary files: Use the byte offset

    When reporting the `type`, use the following values:
    - password
    - api_key
    - username
    - email
    - credit_card
    - url
    - ip_address
    - ssn
    - drivers_license
    - confidential_information
    - secret
    - other
    """

    dn.log_output(
        "report",
        {
            "path": path,
            "location": location,
            "type": type,
            "value": value,
            "comment": comment,
        },
        to="run",
    )
    dn.log_metric("num_reports", 1, mode="count", to="run")

    return "Reported"


@tool()
async def give_up(reason: str) -> None:
    """
    Give up on your task.
    """
    dn.log_output("give_up_reason", reason, to="run")
    dn.log_metric("agent_gave_up", 1)


@tool()
async def complete_task(summary: str) -> None:
    """
    Complete your task and provide a summary as markdown.
    """
    dn.log_output("task_summary", Markdown(summary), to="run")
    dn.log_metric("agent_marked_complete", 1)


@app.default
async def agent(*, args: Args, dn_args: DreadnodeArgs | None = None) -> None:
    """
    Agent to extract sensitive data from a path of files.
    """

    dn_args = dn_args or DreadnodeArgs()
    dn.configure(
        server=dn_args.server,
        token=dn_args.token,
        project=dn_args.project,
        console=dn_args.console,
    )

    filesystem = Filesystem(
        variant="read",
        path=args.path,
        fs_options=args.fs,
    )

    instructions = dedent("""\
        As an authorized security testing assistant, Your task is to analyze files for sensitive data
        such as credentials, personal information, and security-related data. You have permission to
        access and analyze all files within this isolated environment.

        Report all sensitive data you find using the provided tool.
        """)

    user_input = dedent(f"""\
        Your task is as follows:

        <task>
        Extract all sensitive data from the files located at: {args.path}
        </task>
        """)

    agent = Agent(
        name="sensitive-data-extraction-agent",
        description="An agent to extract sensitive data from files.",
        model=args.model,
        instructions=instructions,
        tools=[filesystem, report_sensitive_data, give_up, complete_task],
    )

    async with agent.stream(user_input) as events:
        async for event in events:
            console.print(event)


if __name__ == "__main__":
    app()
