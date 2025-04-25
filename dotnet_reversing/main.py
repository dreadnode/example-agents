import io
import sys
import typing as t
import zipfile
from dataclasses import dataclass
from pathlib import Path

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
    """Model to use for inference"""
    path: str
    """Directory of binaries to analyze or other supported identifier"""
    nuget: bool = False
    """Treat the path as a NuGet package id"""
    task: str = "Find critical vulnerabilities"
    """Task presented to the agent"""
    max_steps: int = 50
    """Maximum number of iterations per agent"""
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
async def report_finding(file: str, method: str, content: str) -> str:
    """
    Report a finding regarding areas or interest or vulnerabilities.
    """
    logger.success(f"Reporting finding for {file} ({method}):")
    logger.info(content)
    logger.info("---")
    dn.log_output(
        "finding",
        {
            "file": file,
            "method": method,
            "content": content,
        },
        to="run",
    )
    dn.log_metric("num_reports", 1, mode="count", to="run")
    return "Reported"


@dn.task(name="Give up", log_output=False)
async def give_up(reason: str) -> None:
    """
    Give up on your task.
    """
    logger.warning(f"Agent gave up: {reason}")
    dn.log_metric("gave_up", 1)


@dn.task(name="Complete task", log_output=False)
async def complete_task(summary: str) -> None:
    """
    Complete the task.
    """
    logger.info(f"Agent completed the task: {summary}")
    dn.log_metric("marked_task_completed", 1)


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


@app.default
async def agent(*, args: Args, dn_args: DreadnodeArgs | None = None) -> None:
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

    with dn.run(tags=[args.task]), dn.task_span("Agent"):
        dn.log_params(
            model=args.model,
            path=str(args.path),
            nuget=args.nuget,
            task=args.task,
            max_steps=args.max_steps,
        )

        path = await download_nuget_package(args.path) if args.nuget else Path(args.path)
        reversing = DotnetReversing.from_path(path)

        logger.info(f"Analayzing the following binaries with the goal: '{args.task}':")
        for binary in reversing.binaries:
            logger.info(f"  |- {binary}")

        dn.log_inputs(
            binaries=[str(b) for b in reversing.binaries],
        )

        binary_list = "\n".join(reversing.binaries)

        generator = rg.get_generator(args.model)
        chat = (
            await generator.chat(
                f"""\
            Analyze the following binaries and resolve the task below using all the tools available to you.
            Provide a report for all interesting findings you discover while performing the task.

            <task>
            {args.task}
            </task>

            <files>
            {binary_list}
            </files>
            """,
            )
            .catch(
                *litellm.exceptions.LITELLM_EXCEPTION_TYPES,
                on_failed="include",
            )
            .using(
                reversing.tools,
                report_finding,
                give_up,
                complete_task,
                max_depth=args.max_steps,
            )
            .run()
        )

        if chat.failed and chat.error:
            if isinstance(chat.error, rg.error.MaxDepthError):
                logger.warning(f"Max steps reached ({args.max_steps})")
                dn.log_metric("max_steps_reached", 1)
            else:
                logger.warning(f"Failed with {chat.error}")
                dn.log_metric("inference_failed", 1)

        elif chat.last.role == "assistant":
            dn.log_output("last_message", chat.last.content)
            logger.info(str(chat.last))

    logger.info("Done.")


if __name__ == "__main__":
    app()
