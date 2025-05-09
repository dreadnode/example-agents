import sys
import typing as t
from dataclasses import dataclass, field

import cyclopts
import dreadnode as dn
import litellm
import rigging as rg
from loguru import logger

from python_agent.kernel import PythonKernel

if t.TYPE_CHECKING:
    from loguru import Record as LogRecord

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
    log_level: str = "INFO"
    """Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"""


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


def log_formatter(record: "LogRecord") -> str:
    return "".join(
        (
            "<green>{time:HH:mm:ss.SSS}</green> | ",
            "<dim>{extra[prefix]}</dim> " if record["extra"].get("prefix") else "",
            "<level>{message}</level>\n",
        ),
    )


@dn.task(name="Complete task", log_output=False)
async def complete_task(success: bool, markdown_summary: str) -> None:  # noqa: FBT001
    """
    Mark your task as complete with a success/failure status and markdown summary.
    """
    dn.log_metric("task_success", success, to="run")

    log_func = logger.success if success else logger.warning
    log_func(f"Agent finished the task (success={success}): {markdown_summary}")

    dn.log_output("task_summary", markdown_summary, to="run")


@app.default
async def agent(*, args: Args, dn_args: DreadnodeArgs | None = None) -> None:
    """
    General agent with access to a dockerized jupyter environment.
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

    with dn.run(), dn.task_span("Agent"):
        dn.log_params(
            model=args.model,
            image=args.image,
            max_steps=args.max_steps,
        )
        dn.log_input("task", args.task, to="run")
        dn.log_input("volumes", "\n".join(args.volumes), to="run")
        dn.push_update()

        generator = rg.get_generator(args.model)

        logger.info("Starting agent ...")

        async with PythonKernel(
            image=args.image,
            volumes=args.volumes,
        ) as kernel:

            @dn.task(name="Execute code")
            async def execute_code(code: str) -> str:
                """
                Execute code in the kernel and return the result.
                """
                logger.info(f"Executing:\n{code}")
                result = await kernel.execute_code(code)
                logger.info(f"Result:\n{result}")
                return result

            @dn.task(name="Restart kernel")
            async def restart_kernel() -> None:
                """
                Restart the kernel.
                """
                logger.info("Restarting kernel ...")
                await kernel.restart()

            chat = (
                await generator.chat(
                    f"""\
                    Work to complete the following task. You have access to a dockerized jupyter environment.
                    You can run code in the environment and use the results to help you complete the task.

                    Unless otherwise specified, use `~/work` to store files and data. Additional volumes are listed below.

                    <volumes>
                    {args.volumes}
                    </volumes>

                    <task>
                    {args.task}
                    </task>
                    """,
                )
                .catch(
                    *litellm.exceptions.LITELLM_EXCEPTION_TYPES,
                    on_failed="include",
                )
                .using(
                    execute_code,
                    restart_kernel,
                    complete_task,
                    max_depth=args.max_steps,
                )
                .run()
            )

            dn.log_artifact(kernel.work_dir)

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
            dn.log_output("last_message", chat.last.content, to="run")
            logger.info(str(chat.last))

    logger.info("Done.")


if __name__ == "__main__":
    app()
