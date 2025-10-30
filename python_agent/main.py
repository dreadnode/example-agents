import typing as t
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent

import cyclopts
import dreadnode as dn
from dreadnode.agent import Agent
from dreadnode.agent.events import AgentEnd
from dreadnode.agent.hooks import Hook
from dreadnode.agent.tools import tool
from dreadnode.data_types import Markdown
from kernel import PythonKernel
from rich.console import Console

console = Console()


# CLI

app = cyclopts.App()


@cyclopts.Parameter(name="*", group="args")
@dataclass
class Args:
    model: str
    """Model to use for inference"""
    task: str
    """Task to perform"""
    image: str = "jupyter/datascience-notebook:latest"
    """Docker image to use for the container"""
    volumes: t.Annotated[
        list[str],
        cyclopts.Parameter(
            name=["--volume", "-v"],
            help="Additional volumes to mount in the container (e.g. /path/to/dir:/path/in/container)",
        ),
    ] = field(default_factory=list)
    max_steps: int = 50
    """Maximum number of steps to take"""


@cyclopts.Parameter(name="*", group="dreadnode")
@dataclass
class DreadnodeArgs:
    server: str | None = None
    """Dreadnode server URL"""
    token: str | None = None
    """Dreadnode API token"""
    project: str | None = "python-agent"
    """Dreadnode project name"""
    console: t.Annotated[bool, cyclopts.Parameter(negative=False)] = False
    """Show span information in the console"""


@tool()
async def complete_task(success: bool, markdown_summary: str) -> None:  # noqa: FBT001
    """
    Mark your task as complete with a success/failure status and markdown summary.
    """
    dn.log_metric("task_success", success, to="run")
    dn.log_output("task_summary", Markdown(markdown_summary), to="run")


def upload_work_hook(
    work_dir: Path,
) -> Hook:
    async def upload_work(event: AgentEnd) -> None:
        dn.log_artifact(str(work_dir))

    return upload_work


@app.default
async def agent(*, args: Args, dn_args: DreadnodeArgs | None = None) -> None:
    """
    General agent with access to a dockerized jupyter environment.
    """

    dn_args = dn_args or DreadnodeArgs()
    dn.configure(
        server=dn_args.server,
        token=dn_args.token,
        project=dn_args.project,
        console=dn_args.console,
    )

    instructions = dedent(f"""\
        Work to complete the following task. You have access to a dockerized jupyter environment.
        You can run code in the environment and use the results to help you complete the task.

        Unless otherwise specified, use `~/work` to store files and data. Additional volumes are listed below.

        <volumes>
        {args.volumes}
        </volumes>

        <task>
        {args.task}
        </task>
        """)

    async with PythonKernel(
        image=args.image,
        volumes=args.volumes,
    ) as kernel:
        agent = Agent(
            name="python-agent",
            model=args.model,
            description="An agent with access to a dockerized jupyter environment.",
            instructions=instructions,
            tools=[kernel],
            hooks=[upload_work_hook(work_dir=kernel.work_dir)],
        )

        async with agent.stream(args.task) as events:
            async for event in events:
                console.print(event)


if __name__ == "__main__":
    app()
