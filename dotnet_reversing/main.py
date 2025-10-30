import io
import typing as t
import zipfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import aiohttp
import cyclopts
import dreadnode as dn
from dreadnode.agent import Agent
from dreadnode.agent.events import AgentStart
from dreadnode.agent.hooks import Hook
from dreadnode.agent.tools import tool
from dreadnode.data_types import Markdown
from reversing import DotnetReversingTool
from rich.console import Console

console = Console()

if t.TYPE_CHECKING:
    from loguru import Record as LogRecord

# CLI

app = cyclopts.App()


@cyclopts.Parameter(name="*", group="args")
@dataclass
class Args:
    model: str
    """Model to use for inference (rigging generator identifier like 'gpt-4o-mini' or 'ollama/llama3-70b')"""
    path: str
    """Binary or directory of binaries to analyze or other supported identifier"""
    nuget: bool = False
    """Treat the path as a NuGet package id or path to a list of packages"""
    task: str = "Find all useful vulnerabilities"
    """Task presented to the agent"""
    max_steps: int = 25
    """Maximum number of iterations per agent"""
    concurrency: int = 3
    """Maximum number of agents to run in parallel at any given time"""
    log_level: str = "INFO"
    """Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"""


@cyclopts.Parameter(name="*", group="dreadnode")
@dataclass
class DreadnodeArgs:
    server: str | None = None
    """Dreadnode server URL"""
    token: str | None = None
    """Dreadnode API token"""
    project: str = "dotnet-reversing-example"
    """Project name"""
    console: t.Annotated[bool, cyclopts.Parameter(negative=False)] = False
    """Show span information in the console"""


def log_formatter(record: "LogRecord") -> str:
    return "".join(
        (
            "<green>{time:HH:mm:ss.SSS}</green> | ",
            "<dim>{extra[prefix]}</dim> " if record["extra"].get("prefix") else "",
            "<level>{message}</level>\n",
        ),
    )


@tool()
async def report_finding(file: str, method: str, criticality: str, content: str) -> str:
    """
    Report a finding regarding areas or interest or vulnerabilities.

    for criticality, use:
    - "critical"
    - "high"
    - "medium"
    - "low"
    - "info"
    """

    dn.log_output(
        "finding",
        {
            "file": file,
            "method": method,
            "content": content,
            "criticality": criticality,
        },
    )
    dn.log_metric("num_reports", 1, mode="count")
    dn.tag(criticality)
    dn.tag("reports", to="run")
    return "Reported"


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


async def download_nuget_package(package: str) -> Path:
    """
    Download a NuGet package and return the path to the package.
    """

    package = package.lower()

    async with aiohttp.ClientSession() as client:
        # Get the versions
        async with client.get(
            f"https://api.nuget.org/v3-flatcontainer/{package}/index.json",
        ) as response:
            if response.status != 200:  # noqa: PLR2004
                raise RuntimeError(f"Failed to get package {package} from NuGet")

            data = await response.json()
            versions = data["versions"]
            latest_version = versions[-1]

        # Download the nupkg and extract it
        async with client.get(
            f"https://api.nuget.org/v3-flatcontainer/{package}/{latest_version}/{package}.{latest_version}.nupkg",
        ) as response:
            if response.status != 200:  # noqa: PLR2004
                raise RuntimeError(f"Failed to download package {package} from NuGet")

            output_dir = Path(f".nuget/{package}_{latest_version}")
            output_dir.mkdir(parents=True, exist_ok=True)

            data = await response.read()
            with io.BytesIO(data) as buffer, zipfile.ZipFile(buffer) as zip_file:
                zip_file.extractall(output_dir)

    return output_dir


def upload_package_hook(
    package_path: str,
) -> Hook:
    async def upload_package(event: AgentStart) -> None:
        dn.log_artifact(package_path)

    return upload_package


@app.default
async def main(*, args: Args, dn_args: DreadnodeArgs | None = None) -> None:
    """
    Agent to analyze .NET binaries and report findings.
    """

    dn_args = dn_args or DreadnodeArgs()
    dn.configure(
        server=dn_args.server,
        token=dn_args.token,
        project=dn_args.project,
        console=dn_args.console,
    )

    path = await download_nuget_package(args.path) if args.nuget else Path(args.path)
    reversing = DotnetReversingTool(variant="all", base_path=path)

    binary_list = "\n".join(reversing.binaries)

    instructions = dedent("""\
        You are a .NET reversing expert.

        Use the Dotnet Reversing tool to decompile and analyze the binaries.
        For each finding, use the Report Finding tool to log your discoveries.

        Once you have completed your analysis, use the Finish Task tool to summarize your findings
        and indicate whether you successfully completed the task.
        """)

    user_input = dedent(f"""\
        Here is your task:

        <task>
        {args.task}
        </task>

        <files>
        {binary_list}
        </files>
        """)

    agent = Agent(
        name="Dotnet Reversing Agent",
        description="Agent to analyze .NET binaries and report findings.",
        model=args.model,
        instructions=instructions,
        tools=[reversing, report_finding, finish_task],
        max_steps=args.max_steps,
        hooks=[upload_package_hook(str(path))],
    )

    async with agent.stream(user_input) as events:
        async for event in events:
            console.print(event)


if __name__ == "__main__":
    app()
