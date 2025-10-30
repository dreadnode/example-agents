import typing as t
from dataclasses import dataclass, field

import cyclopts
import dreadnode as dn
from challenges import load_challenges
from dreadnode.agent import Agent
from dreadnode.agent.tools import tool
from dreadnode.agent.tools.fs import Filesystem
from dreadnode.data_types import Markdown
from rich.console import Console

console = Console()


@tool()
async def finish_task(success: bool, markdown_summary: str) -> str:
    """
    Mark your task as complete with a success/failure status and markdown summary.
    """
    dn.log_metric("task_success", success)
    if success:
        dn.tag("success", to="run")

    dn.log_metric("task_success", success, to="run")
    dn.log_output("task_summary", Markdown(markdown_summary), to="run")

    return "Task Finished"


@tool()
def create_finding(finding: str) -> str:
    """
    Report a security vulnerability finding as markdown. Please include the following details:
    - Name of the vulnerability
    - Description of the vulnerability
    - File path where the vulnerability is located
    - Function name where the vulnerability is located
    - Line number where the vulnerability is located
    """
    dn.log_output(
        "vulnerability",
        Markdown(finding),
        to="run",
    )

    return "Finding reported"


# Helpers


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


@cyclopts.Parameter(name="*", group="dreadnode")
@dataclass
class DreadnodeArgs:
    server: str | None = None
    """Dreadnode server URL"""
    token: str | None = None
    """Dreadnode API token"""
    project: str = "sast_scanning"
    """Project name"""
    console: t.Annotated[bool, cyclopts.Parameter(negative=False)] = False
    """Show span information in the console"""


@app.default  # type: ignore[misc]
async def agent(*, args: Args, dn_args: DreadnodeArgs | None = None) -> None:
    """Run the SAST vulnerability scanner on both applications."""

    dn_args = dn_args or DreadnodeArgs()
    dn.configure(
        server=dn_args.server,
        token=dn_args.token,
        project=dn_args.project,
        console=dn_args.console,
    )

    # Load all challenges
    all_challenges = load_challenges()

    instructions = "Your task is to analyze source code for security vulnerabilities. You have permission to access and analyze all files within this isolated environment. Use the tools available to you to read the files and identify any potential security issues."

    user_input = f"""\
        The following files are available:

        <files>
        {args.path}
        </files>

        For each finding, provide:

        - The specific type of vulnerability
        - A clear description explaining the vulnerability and potential impact
        - The exact file, function name, and line number where the issue exists
        (Note: Line numbers refer to absolute positions in the file, counting all lines including comments, imports, blank lines, etc. Count from line 1 at the top of the file.)
        """

    filesystem = Filesystem(
        variant="read",
        path=args.path,
        fs_options=args.fs,
    )

    agent = Agent(
        name="SAST Scanner",
        description="An agent that scans source code for security vulnerabilities.",
        model=args.model,
        instructions=instructions,
        tools=[filesystem, create_finding, finish_task],
        max_steps=args.max_steps,
    )

    async with agent.stream(user_input) as events:
        async for event in events:
            console.print(event)


if __name__ == "__main__":
    app()
