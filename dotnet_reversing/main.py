import io
import sys
import typing as t
import zipfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import aiohttp
import cyclopts
import dreadnode as dn
import litellm
import rigging as rg
from loguru import logger

from dotnet_reversing.reversing import DotnetReversing

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
    task: str = "Find only critical vulnerabilities"
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
    project: str = "dotnet-reversing"
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


@dn.task(name="Report finding", log_inputs=False, log_output=False)
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
    logger.success(f"Reporting finding for {file} ({method}) [{criticality}]:")
    logger.info(content)
    logger.info("---")
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


@dn.task(name="Finish task", log_output=False)
async def finish_task(success: bool, markdown_summary: str) -> None:
    """
    Mark your task as complete with a success/failure status and markdown summary.
    """
    dn.log_metric("task_success", success)
    if success:
        dn.tag("success", to="run")

    log_func = logger.success if success else logger.warning
    log_func(f"Agent finished the task (success={success}): {markdown_summary}")

    dn.log_metric("task_success", success, to="run")
    dn.log_output("task_summary", markdown_summary, to="run")


@dn.task(name="Download NuGet package")
async def download_nuget_package(package: str) -> Path:
    """
    Download a NuGet package and return the path to the package.
    """

    package = package.lower()
    logger.info(f"Downloading NuGet package {package}...")

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
            logger.info(f" |- Latest version is {latest_version}")

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

            logger.info(f" |- Extracted to {output_dir}")

    return output_dir


async def agent(args: Args) -> None:
    with (
        dn.run(),
        dn.task_span("Agent"),
        logger.contextualize(prefix=str(args.path)),
    ):
        dn.log_params(
            model=args.model,
            path=str(args.path),
            nuget=args.nuget,
            task=args.task,
            max_steps=args.max_steps,
        )

        path = await download_nuget_package(args.path) if args.nuget else Path(args.path)
        reversing = DotnetReversing.from_path(path)

        logger.info(f"Analyzing the following binaries with the goal: '{args.task}':")
        for binary in reversing.binaries:
            logger.info(f"  |- {binary}")

        dn.log_inputs(
            binaries=[str(b) for b in reversing.binaries],
        )

        binary_list = "\n".join(reversing.binaries)

        prompt = dedent(f"""\
        Analyze the following binaries and resolve the task below using all the tools available to you.
        Provide a report for all interesting findings you discover while performing the task.

        <task>
        {args.task}
        </task>

        <files>
        {binary_list}
        </files>
        """)

        dn.log_input("task", args.task, to="run")
        dn.log_input("binaries", binary_list, to="run")

        generator = rg.get_generator(args.model)
        chat = (
            await generator.chat(prompt)
            .catch(
                *litellm.exceptions.LITELLM_EXCEPTION_TYPES,
                on_failed="include",
            )
            .using(
                reversing.tools,
                report_finding,
                finish_task,
                max_depth=args.max_steps,
            )
            .cache("latest")
            .run()
        )

        if chat.failed and chat.error:
            if isinstance(chat.error, rg.error.MaxDepthError):
                logger.warning(f"Max steps reached ({args.max_steps})")
                dn.log_metric("max_steps_reached", 1)
                dn.log_output("task_summary", f"Max steps ({args.max_steps}) reached", to="run")
            else:
                logger.warning(f"Failed with {chat.error}")
                dn.log_metric("inference_failed", 1)
                dn.log_output("task_summary", f"Inference failed with {chat.error}", to="run")

        elif chat.last.role == "assistant":
            dn.log_output("last_message", chat.last.content)
            logger.info(str(chat.last))


@app.default
async def main(*, args: Args, dn_args: DreadnodeArgs | None = None) -> None:
    """
    Agent to analyze .NET binaries and report findings.
    """

    logger.remove()
    logger.add(sys.stderr, format=log_formatter, level=args.log_level)
    logger.enable("rigging")

    dn_args = dn_args or DreadnodeArgs()
    dn.configure(
        server=dn_args.server,
        token=dn_args.token,
        project=dn_args.project,
        console=dn_args.console,
    )

    await agent(args)

    logger.info("Done.")


@app.command
async def dump(*, args: Args, dn_args: DreadnodeArgs | None = None) -> None:
    """
    Dump the source code of the binaries in the specified path.
    """
    logger.remove()
    logger.add(sys.stderr, format=log_formatter, level=args.log_level)
    logger.enable("rigging")

    dn_args = dn_args or DreadnodeArgs()
    dn.configure(
        server=dn_args.server,
        token=dn_args.token,
        project=dn_args.project,
        console=dn_args.console,
    )

    path = await download_nuget_package(args.path) if args.nuget else Path(args.path)
    reversing = DotnetReversing.from_path(path)

    for binary in reversing.binaries:
        logger.info(f"Dumping source code for {binary}...")
        source_code = reversing.decompile_module(binary)
        output_file = Path(f"{binary}_source.txt")
        output_file.write_text(source_code)
        logger.success(f"Source code dumped to {output_file}")


if __name__ == "__main__":
    app()
