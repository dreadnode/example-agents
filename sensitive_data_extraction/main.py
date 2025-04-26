import sys
import typing as t
from dataclasses import dataclass, field

import cyclopts
import dreadnode as dn
import litellm
import rigging as rg
from loguru import logger

from sensitive_data_extraction.filesystem import FilesystemTools

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


def log_formatter(record: "LogRecord") -> str:
    return "".join(
        (
            "<green>{time:HH:mm:ss.SSS}</green> | ",
            "<dim>{extra[prefix]}</dim> " if record["extra"].get("prefix") else "",
            "<level>{message}</level>\n",
        ),
    )


@dn.task(name="Report data", log_output=False, log_inputs=False)
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
    logger.success(f"Found data in {path}:{location} ({type})")
    logger.success(f"  |- {comment}")
    logger.success(f"  |= {value}")

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


@dn.task(name="Give up", log_output=False)
async def give_up(reason: str) -> None:
    """
    Give up on your task.
    """
    logger.warning(f"Agent gave up: {reason}")
    dn.log_metric("agent_gave_up", 1)


@dn.task(name="Complete task", log_output=False)
async def complete_task(summary: str) -> None:
    """
    Complete your task.
    """
    logger.info(f"Agent completed the task: {summary}")
    dn.log_metric("agent_marked_complete", 1)


@app.default
async def agent(*, args: Args, dn_args: DreadnodeArgs | None = None) -> None:
    """
    Agent to extract sensitive data from a path of files.
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
        send_to_logfire=False,
    )

    with dn.run(), dn.task_span("Agent"):
        dn.log_params(
            model=args.model,
            path=args.path,
            max_steps=args.max_steps,
        )

        filesystem = FilesystemTools(
            args.path,
            mode="read-only",
            fs_options=args.fs,
        )

        generator = rg.get_generator(args.model)
        chat = (
            await generator.chat(
                """\
                As an authorized security testing assistant, Your task is to analyze files for sensitive data
                such as credentials, personal information, and security-related data. You have permission to
                access and analyze all files within this isolated environment.

                Report all sensitive data you find using the provided tool.
                """,
            )
            .catch(
                *litellm.exceptions.LITELLM_EXCEPTION_TYPES,
                on_failed="include",
            )
            .using(
                filesystem.tools,
                report_sensitive_data,
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
                dn.log_metric("failed_chat", 1, mode="count")

        elif chat.last.role == "assistant":
            logger.info(str(chat.last))
            dn.log_output("last_message", chat.last.content)

    logger.info("Done.")


if __name__ == "__main__":
    app()
