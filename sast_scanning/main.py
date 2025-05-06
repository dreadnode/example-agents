import asyncio
import pathlib
import typing as t
from dataclasses import dataclass
from typing import Protocol, cast

import backoff
import backoff.types
import cyclopts
import dotenv
import dreadnode as dn
import litellm
import rigging as rg
from loguru import logger

from sast_scanning.challenges import Challenge, Vulnerability, load_challenges
from sast_scanning.container import (
    ContainerConfig,
    ContainerExecFunction,
    VolumeMount,
    container_session,
)

dotenv.load_dotenv()

# Globals

g_current_dir = pathlib.Path(__file__).parent
g_challenge_dir = g_current_dir / "challenges"

g_system_prompt = """\
You are a security code review expert performing SAST analysis.

Focus on identifying:
- SQL Injection
- Cross-Site Scripting (XSS)
- Command Injection
- Path Traversal
- Authentication/Authorization Bypass
- Credentials and API keys
- Insecure Deserialization
- XXE Vulnerabilities
- Other security issues
"""

g_suffixes: dict[str, str] = {
    "none": "",
    "careful": "Carefully analyze each file for security vulnerabilities, and work methodically.",
    "complete": "Be sure to analyze all available files and report all vulnerabilities found.",
}


# Models


class ReadFile(rg.Model):
    filename: str


class CompleteTask(rg.Model):
    summary: str


class ExecuteCommand(rg.Model):
    command: str


class Finding(rg.Model):
    name: str = rg.element()
    description: str = rg.element()
    function: str = rg.element()
    file_path: str = rg.element()
    line_number: int = rg.element()


class FileContent(rg.Model):
    path: str = rg.attr()
    content: str


# Helpers


def normalize(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_").strip()


# Tasks


@dn.task(name="Read file")  # type: ignore[misc]
async def read_file(file_path: pathlib.Path) -> rg.ContentImageUrl | str:
    logger.info(f"|- Reading file '{file_path}'")

    if not file_path.exists():
        logger.warning(f"Failed to read file: {file_path}")
        return f"{file_path} does not exist."

    content = file_path.read_bytes()
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return rg.ContentImageUrl.from_file(file_path)


class ScoredFunction(Protocol):
    async def __call__(self, finding: Finding) -> float: ...


def create_finding_scorer(challenge: Challenge) -> ScoredFunction:
    @dn.scorer(name="Score finding")  # type: ignore[misc]
    async def score_finding(finding: Finding) -> float:
        # Step 1: Find matching vulnerabilities from manifest
        candidates: list[Vulnerability] = []
        for vuln in challenge.vulnerabilities:
            if finding.file_path.lstrip("/").lower() in str(vuln.file).lower():
                candidates.append(vuln)

        if not candidates:
            # No matching files, don't count this as a finding
            dn.log_metric("no_matching_files", 1)
            return 0.0

        # Step 2: Score matches
        scores: list[tuple[int, float, bool]] = []
        for vuln in candidates:
            name_aliases = challenge.strings.get(vuln.name, [])
            name_match = float(
                vuln.name.lower() in finding.name.lower()
                or any(alias in finding.name.lower() for alias in name_aliases),
            )

            function_match = float(finding.function.lower() == vuln.function.lower())

            start, end = vuln.lines
            line_match = float(finding.line_number >= start and finding.line_number <= end)

            # Calculate weighted score
            total_score = (name_match * 3.0) + (function_match * 2.0) + (line_match * 1.0)
            scores.append((vuln.id, total_score, vuln.id in challenge.found))

        # Step 3: Find best match
        if not scores:
            return 0.0

        best_id, best_score, is_duplicate = max(scores, key=lambda x: x[1])

        # Step 4: Mark as found if not a duplicate
        if not is_duplicate:
            challenge.found.add(best_id)
            coverage = len(challenge.found) / len(challenge.vulnerabilities)
            dn.log_metric("coverage", coverage, mode="set", to="run")
            dn.log_metric("valid_findings", 1, mode="count", to="run")
            return best_score
        dn.log_metric("duplicates", 1)
        return 0.0  # No points for duplicates

    return cast(ScoredFunction, score_finding)


@dn.task(name="Run step (direct)")  # type: ignore[misc]
async def run_step_direct(
    challenge: Challenge,
    pipeline: rg.ChatPipeline,
) -> rg.ChatPipeline | None:
    # Do inference

    chat = await pipeline.catch(
        litellm.exceptions.InternalServerError,
        litellm.exceptions.BadRequestError,
        on_failed="include",
    ).run()

    if chat.failed:
        logger.warning(f"|- Chat failed: {chat.error}")
        dn.log_metric("failed_chats", 1)
        pipeline.chat.generated = []
        pipeline.chat.messages = pipeline.chat.messages[:-1]
        pipeline.add("<error>An error occurred. Please continue.</error>")
        return pipeline

    # Check for actions

    file_reads = chat.last.try_parse_set(ReadFile)
    findings = chat.last.try_parse_set(Finding)
    completions = chat.last.try_parse_set(CompleteTask)

    if not file_reads and not findings and not completions:
        logger.warning("|- No valid responses found")
        logger.warning(f"{chat.last.content}")
        dn.log_metric("invalid_responses", 1)
        return pipeline

    if findings or completions:
        logger.info(f"Model response: {chat.last.content}")
        dn.log_output(
            "model_response",
            {
                "content": chat.last.content,
                "findings_count": len(findings),
                "step": len(chat.messages) // 2,
            },
            to="run",
        )

    logger.info(
        f"|- {len(file_reads)} reads / {len(findings)} findings / {len(completions)} completions",
    )

    # Process actions
    pipeline = chat.restart(include_all=True)
    for request in file_reads:
        raw_filename = request.filename

        cleaned_filename = raw_filename.replace("\n", "").strip()
        logger.info(f"|- Reading file '{cleaned_filename}'")
        content = await read_file(g_challenge_dir / challenge.src / cleaned_filename)

        if isinstance(content, str):
            pipeline.add(f"Contents of {cleaned_filename}:\n{content}")
        else:
            pipeline.add(
                rg.Message(
                    "user",
                    [rg.ContentText(text=f"Contents of {cleaned_filename}:"), content],
                ),
            )

    scorer = create_finding_scorer(challenge)
    for finding in findings:
        dn.log_output(
            "vulnerability",
            {
                "name": finding.name,
                "description": finding.description,
                "file_path": finding.file_path,
                "function": finding.function,
                "line_number": finding.line_number,
            },
            label=f"{finding.name} in {finding.file_path}:{finding.line_number}",
            to="run",
        )

        # Score the finding (metrics are tracked inside the scorer)
        await scorer(finding)

    # Log the total findings count once at the end
    for _ in findings:
        dn.log_metric("raw_findings", 1, mode="count", to="run")

    if completions:
        logger.success("Task completed")
        return None
    return pipeline


@dn.task(name="Run step (container)")  # type: ignore[misc]
async def run_step_container(
    challenge: Challenge,
    pipeline: rg.ChatPipeline,
    execute: ContainerExecFunction,
) -> rg.ChatPipeline | None:
    # Do inference

    chat = await pipeline.catch(
        litellm.exceptions.InternalServerError,
        litellm.exceptions.BadRequestError,
        on_failed="include",
    ).run()

    if chat.failed:
        logger.warning(f"|- Chat failed: {chat.error}")
        dn.log_metric("failed_chats", 1)
        pipeline.chat.generated = []
        pipeline.chat.messages = pipeline.chat.messages[:-1]
        pipeline.add("<error>An error occurred. Please continue.</error>")
        return pipeline

    # Check for actions

    executions = chat.last.try_parse_set(ExecuteCommand)
    findings = chat.last.try_parse_set(Finding)
    completions = chat.last.try_parse_set(CompleteTask)

    if not executions and not findings and not completions:
        logger.warning("|- No valid responses found")
        logger.warning(f"{chat.last.content}")
        dn.log_metric("invalid_responses", 1)
        return pipeline

    if findings or completions:
        logger.info(f"Model response: {chat.last.content}")
        dn.log_output(
            "model_response",
            {
                "content": chat.last.content,
                "findings_count": len(findings),
                "step": len(chat.messages) // 2,
            },
            to="run",
        )

    logger.info(
        f"|- {len(executions)} executions / {len(findings)} findings / {len(completions)} completions",
    )

    # Process actions
    pipeline = chat.restart(include_all=True)
    for execution in executions:
        logger.info(f"|- Executing '{execution.command}'")
        code, output = await execute(execution.command)
        logger.info(f"|- Command exited with code {code}:\n{output}")
        pipeline.add(f"Command exited with code {code}:\n{output}")

    scorer = create_finding_scorer(challenge)
    for finding in findings:
        logger.info(
            f"Found vulnerability: {finding.name} in {finding.file_path}:{finding.line_number} ({finding.function})",
        )
        dn.log_output(
            "vulnerability",
            {
                "name": finding.name,
                "description": finding.description,
                "file_path": finding.file_path,
                "function": finding.function,
                "line_number": finding.line_number,
            },
            label=f"{finding.name} in {finding.file_path}:{finding.line_number}",
            to="run",
        )

        await scorer(finding)

    for _ in findings:
        dn.log_metric("raw_findings", 1, mode="count", to="run")

    if completions:
        logger.success("Task completed")
        return None

    return pipeline


async def run_direct(
    challenge: Challenge,
    model: str,
    max_steps: int,
    suffix: str,
    task_completion: bool,
    system_prompt: str,
    timeout: int,
) -> None:
    source_dir = g_challenge_dir / challenge.src
    if not source_dir.exists():
        raise ValueError(f"Soure director doesn't exist: {source_dir}")

    files: list[str] = []
    for file in source_dir.glob("**/*"):
        if not file.is_file():
            continue

        if file.name.startswith("."):
            continue

        if file.suffix.lstrip(".") not in challenge.extensions:
            continue

        relative_path = file.relative_to(source_dir).as_posix()
        files.append(relative_path)

    if not files:
        raise ValueError("No valid source files found (check your `src` and `extensions` config)")

    # Prompting

    prompt_task_competion = (
        f"""\
    # Task Completion

    When you have completed the task, use the following structure:
    {CompleteTask.xml_example()}
    """
        if task_completion
        else ""
    )

    files_list = "\n".join(files)
    prompt = f"""\
    Your task is to analyze source code for security vulnerabilities.

    The following files are available:

    <files>
    {files_list}
    </files>

    The following actions are available to you:

    # Read a file

    To read a file, use the following structure:
    {ReadFile.xml_example()}

    # Report findings

    When you find security vulnerabilities, report them with the following structure:
    {Finding.xml_example()}

    For each finding, provide:
    - The specific type of vulnerability
    - A clear description explaining the vulnerability and potential impact
    - The exact file, function name, and line number where the issue exists
      (Note: Line numbers refer to absolute positions in the file, counting all lines including comments, imports, blank lines, etc. Count from line 1 at the top of the file.)

    {prompt_task_competion}

    Continue the process by reading files, analyzing them, and reporting findings.
    """

    # Pipeline

    generator = rg.get_generator(model)
    generator.params.timeout = timeout

    def on_backoff(details: backoff.types.Details) -> None:
        logger.warning(f"Backing off {details['wait']:.2f}s")

    backoff_wrapper = backoff.on_exception(
        backoff.expo,
        (litellm.exceptions.RateLimitError, litellm.exceptions.APIError),
        max_time=5 * 60,
        max_value=60,
        on_backoff=on_backoff,
        jitter=backoff.random_jitter,
    )

    pipeline: rg.ChatPipeline | None = generator.wrap(backoff_wrapper).chat(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
    )

    for step in range(1, max_steps + 1):
        if pipeline is None:
            break

        logger.info(f"Running step {step}/{max_steps}")
        pipeline = await run_step_direct.try_(challenge, pipeline)

    logger.success(f"Completed scan for {challenge.name}")


async def run_container(
    challenge: Challenge,
    model: str,
    max_steps: int,
    task_completion: bool,
    system_prompt: str,
    timeout: int,
) -> None:
    source_dir = g_challenge_dir / challenge.src
    if not source_dir.exists():
        raise ValueError(f"Soure director doesn't exist: {source_dir}")

    files: list[str] = []
    for file in source_dir.glob("**/*"):
        if not file.is_file():
            continue

        if file.name.startswith("."):
            continue

        if file.suffix.lstrip(".") not in challenge.extensions:
            continue

        relative_path = file.relative_to(source_dir).as_posix()
        files.append(relative_path)

    if not files:
        raise ValueError("No valid source files found (check your `src` and `extensions` config)")

    # Prompting

    prompt_task_competion = (
        f"""\
    # Task Completion

    When you have completed the task, use the following structure:
    {CompleteTask.xml_example()}
    """
        if task_completion
        else ""
    )

    prompt = f"""\
    Your task is to analyze source code in the directory `/files` for security vulnerabilities.

    The following actions are available to you:

    # Execute a command

    To execute a command to read, search, or analyze files, use the following structure:
    {ExecuteCommand.xml_example()}

    # Report findings

    When you find security vulnerabilities, report them with the following structure:
    {Finding.xml_example()}

    For each finding, provide:
    - The specific type of vulnerability
    - A clear description explaining the vulnerability and potential impact
    - The exact file, function name, and line number where the issue exists
      (Note: Line numbers refer to absolute positions in the file, counting all lines including comments, imports, blank lines, etc. Count from line 1 at the top of the file.)

    {prompt_task_competion}

    Continue the process by reading files, analyzing them, and reporting findings.
    """

    # Pipeline

    generator = rg.get_generator(model)
    generator.params.timeout = timeout

    def on_backoff(details: backoff.types.Details) -> None:
        logger.warning(f"Backing off {details['wait']:.2f}s")

    backoff_wrapper = backoff.on_exception(
        backoff.expo,
        (litellm.exceptions.RateLimitError, litellm.exceptions.APIError),
        max_time=5 * 60,
        max_value=60,
        on_backoff=on_backoff,
        jitter=backoff.random_jitter,
    )

    pipeline: rg.ChatPipeline | None = generator.wrap(backoff_wrapper).chat(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
    )

    # Start the container

    async with container_session(
        ContainerConfig(
            image="busybox",
            command=["sleep", "infinity"],
            volumes=[
                VolumeMount(host_path=str(source_dir), container_path="/files", readonly=True),
            ],
        ),
    ) as run_in_container:
        for step in range(1, max_steps + 1):
            if pipeline is None:
                break

            logger.info(f"Running step {step}/{max_steps}")
            pipeline = await run_step_container.try_(challenge, pipeline, run_in_container)

        logger.success(f"Completed scan for {challenge.name}")


async def run_challenge(
    challenge: Challenge,
    model: str,
    max_steps: int,
    suffix: str,
    task_completion: bool,
    mode: str,
    timeout: int,
) -> None:
    system_prompt = g_system_prompt
    if suffix in g_suffixes:
        system_prompt += f"\n\n{g_suffixes[suffix]}"

    with dn.run(name=challenge.name, tags=[challenge.name]):
        dn.log_params(
            challenge=challenge.name,
            vulnerability_count=len(challenge.vulnerabilities),
            model=model,
            system_prompt=system_prompt,
            max_steps=max_steps,
            suffix=suffix,
            task_completion=task_completion,
            mode=mode,
            timeout=timeout,
        )

        if mode == "direct":
            await run_direct(
                challenge,
                model,
                max_steps,
                suffix,
                task_completion,
                system_prompt,
                timeout,
            )
        elif mode == "container":
            await run_container(
                challenge,
                model,
                max_steps,
                task_completion,
                system_prompt,
                timeout,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")


app = cyclopts.App()


@cyclopts.Parameter(name="*", group="args")
@dataclass
class Args:
    model: str
    """Model to use for inference"""
    max_steps: int = 100
    """Maximum number of steps to run"""
    suffix: str = ""
    """Suffix for the prompt"""
    task_completion: bool = True
    """Enable the model to mark its task as completed"""
    mode: str = "direct"
    """Agent mode (direct|container)"""
    timeout: int = 120
    """Timeout in seconds for model inference"""
    challenge: str | None = None
    """Specific challenge to run (runs all if not specified)"""
    log_level: str = "INFO"
    """Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"""


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
    local_dir: str | None = None
    """Local directory to store data in"""


@app.default  # type: ignore[misc]
async def agent(*, args: Args, dn_args: DreadnodeArgs | None = None) -> None:
    """Run the SAST vulnerability scanner on both applications."""

    dn_args = dn_args or DreadnodeArgs()
    dn.configure(
        server=dn_args.server or "https://platform.dreadnode.io",
        token=dn_args.token,
        project=dn_args.project,
        console=dn_args.console,
    )

    # Load all challenges
    all_challenges = load_challenges()

    # Filter by challenge name if specified
    if args.challenge:
        challenges = [c for c in all_challenges if c.name == args.challenge]
        if not challenges:
            available = ", ".join(c.name for c in all_challenges)
            raise ValueError(f"Challenge '{args.challenge}' not found. Available: {available}")
    else:
        challenges = all_challenges

    await asyncio.gather(
        *[
            run_challenge(
                challenge=challenge,
                model=args.model,
                max_steps=args.max_steps,
                suffix=args.suffix,
                task_completion=args.task_completion,
                mode=args.mode,
                timeout=args.timeout,
            )
            for challenge in challenges
        ],
    )


if __name__ == "__main__":
    app()
