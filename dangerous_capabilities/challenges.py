import asyncio
import contextlib
import json
import pathlib
import typing as t
import uuid
from contextlib import asynccontextmanager

import aiodocker
import aiodocker.containers
import aiodocker.networks
import docker  # type: ignore [import-untyped]
import dreadnode as dn
import rich
from loguru import logger
from pydantic import BaseModel

# Constants

current_dir = pathlib.Path(__file__).parent
challenges_dir = current_dir / "challenges"

Difficulty = t.Literal["easy", "medium", "hard"]

NETWORK_PREFIX = "dangerous-capabilities"
COMMAND_TIMEOUT_EXIT_CODE = 137

# Models


class ContainerDef(BaseModel):
    name: str
    path: pathlib.Path


class Challenge(BaseModel):
    name: str
    containers: list[ContainerDef]
    prompts: dict[str, str]


class ContainerExecFunction(t.Protocol):
    async def __call__(
        self,
        cmd: str,
        *,
        timeout: int = 10,
        workdir: str | None = None,
        shell: str = "/bin/bash",
    ) -> tuple[int, str]: ...


def _parse_memory_limit(limit: str) -> int:
    """Convert memory limit string to bytes integer."""
    if limit.lower().endswith("g"):
        return int(float(limit[:-1]) * 1024 * 1024 * 1024)
    if limit.lower().endswith("m"):
        return int(float(limit[:-1]) * 1024 * 1024)
    if limit.lower().endswith("k"):
        return int(float(limit[:-1]) * 1024)
    # Assume bytes if no unit specified
    return int(float(limit))


@dn.task(name="Start container")
async def start_container(
    client: aiodocker.Docker,
    container: ContainerDef,
    network: aiodocker.networks.DockerNetwork,
    *,
    hostname: str = "linux",
    memory_limit: str | None = None,
) -> aiodocker.containers.DockerContainer:
    config: dict[str, t.Any] = {
        "Image": f"{container.path}:latest",
        "Hostname": hostname,
    }

    if memory_limit:
        config["HostConfig"] = {
            "Memory": _parse_memory_limit(memory_limit),
        }

    docker_container = await client.containers.create(config)
    await docker_container.start()

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(
            docker_container.wait(),
            timeout=1,
        )

    # Check for non-zero exit code
    info = await docker_container.show()
    if info["State"]["ExitCode"] != 0:
        logs = await docker_container.log(stdout=True, stderr=True)
        log_str = "\n".join(logs)
        raise RuntimeError(
            f"Container {container.name} failed to start:\n{log_str}",
        )

    await network.connect(
        {
            "Container": docker_container.id,
            "EndpointConfig": {
                "Aliases": [container.name],
            },
        },
    )

    return docker_container


@asynccontextmanager
async def start_containers(
    challenge: Challenge,
    *,
    memory_limit: str | None = None,
    isolated: bool = True,
) -> t.AsyncGenerator[ContainerExecFunction, None]:
    docker_client = aiodocker.Docker()

    try:
        network_name = f"{NETWORK_PREFIX}-{uuid.uuid4().hex[:8]}"
        network = await docker_client.networks.create(
            {
                "Name": network_name,
                "Driver": "bridge",
                "Internal": isolated,
            },
        )
        logger.info(f"Created network '{network_name}' for '{challenge.name}'")

        containers = await asyncio.gather(
            *[
                start_container(docker_client, container, network, memory_limit=memory_limit)
                for container in challenge.containers
            ],
        )
    except Exception:
        await docker_client.close()
        raise

    # Check if any of the container
    logger.info(f"Started {len(containers)} containers for '{challenge.name}'")
    for container in containers:
        logger.info(f'  |- {container["Config"]["Image"]} ({container["Name"]})')

    async def container_exec(
        cmd: str,
        *,
        timeout: int = 10,
        workdir: str | None = None,
        shell: str = "/bin/bash",
    ) -> tuple[int, str]:
        exec_ = await containers[0].exec(
            [
                "timeout",
                "--kill-after=1",
                "--signal=SIGTERM",
                str(timeout),
                shell,
                "-c",
                cmd,
            ],
            privileged=True,
            workdir=workdir,
        )

        output = ""
        async with exec_.start() as stream:
            while True:
                message = await stream.read_out()
                if message is None:
                    break
                output += message.data.decode(errors="replace")

        inspection = await exec_.inspect()

        exit_code = inspection.get("ExitCode", None) or 0
        if exit_code == COMMAND_TIMEOUT_EXIT_CODE:
            logger.warning(f"Command timed out after {timeout}s")

        return exit_code, output

    try:
        yield container_exec
    finally:
        for container in containers:
            await container.stop(signal="SIGKILL")
            await container.delete()
        await network.delete()
        await docker_client.close()


async def build_challenges(flag: str, *, rebuild: bool = False) -> list[Challenge]:
    with (challenges_dir / "challenges.json").open() as f:
        challenges = [Challenge(**challenge) for challenge in json.load(f)]

    container_paths = {
        container.path for challenge in challenges for container in challenge.containers
    }
    docker_client = docker.DockerClient()

    logger.info("Pruning networks ...")
    docker_client.networks.prune()

    logger.success(
        f"Building {len(container_paths)} containers ...",
    )
    for path in container_paths:
        full_path = challenges_dir / path
        tag = f"{path}:latest"

        if not rebuild and docker_client.images.list(name=tag):
            logger.info(f" |- Found {tag}, skipping build")
            continue

        logger.info(f" |- Building {tag} ({full_path})")

        for item in docker_client.api.build(
            path=str(full_path),
            tag=tag,
            buildargs={"FLAG": flag},
            decode=True,
        ):
            if "error" in item:
                rich.print()
                raise RuntimeError(item["error"])
            if "stream" in item:
                rich.print("[dim]" + item["stream"].strip() + "[/]")

    logger.success("Containers built.")

    return challenges
