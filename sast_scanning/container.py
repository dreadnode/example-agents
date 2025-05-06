import os
import pathlib
import typing as t
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import aiodocker
import aiodocker.containers
import aiodocker.networks
import dreadnode as dn
from loguru import logger

current_dir = pathlib.Path(__file__).parent


@dataclass
class VolumeMount:
    host_path: str
    container_path: str
    readonly: bool = False


@dataclass
class ContainerConfig:
    image: str
    tag: str = "latest"
    name: str | None = None
    hostname: str = "container"
    command: list[str] | None = None
    volumes: list[VolumeMount] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    working_dir: str | None = None
    network: str | None = None


class ContainerExecFunction(t.Protocol):
    async def __call__(
        self,
        cmd: str,
        *,
        timeout: int = 10,
        workdir: str | None = None,
        shell: str = "/bin/bash",
    ) -> tuple[int, str]: ...


async def start_single_container(
    client: aiodocker.Docker,
    container_config: ContainerConfig,
    network: aiodocker.networks.DockerNetwork | None = None,
) -> aiodocker.containers.DockerContainer:
    """Start a single container with the provided configuration"""

    # Create volume binds
    binds = []
    for volume in container_config.volumes:
        bind_config = {
            "Source": os.path.abspath(os.path.expanduser(volume.host_path)),
            "Target": volume.container_path,
            "ReadOnly": volume.readonly,
        }
        binds.append(bind_config)

    # Configure environment variables
    env_vars = [f"{k}={v}" for k, v in container_config.environment.items()]

    image = f"{container_config.image}:{container_config.tag}"
    await client.images.pull(image)

    # Build container config
    config = {
        "Image": image,
        "Hostname": container_config.hostname,
        "Cmd": container_config.command or None,
        "HostConfig": {
            "Binds": [
                f"{v.host_path}:{v.container_path}:{'ro' if v.readonly else 'rw'}"
                for v in container_config.volumes
            ],
        },
        "Env": env_vars,
    }

    if container_config.working_dir:
        config["WorkingDir"] = container_config.working_dir

    if container_config.name:
        config["Name"] = container_config.name

    # Create and start container
    docker_container = await client.containers.create(config=config)  # type: ignore
    await docker_container.start()
    container_info = await docker_container.show()

    # Connect to network if provided
    if network:
        await network.connect(
            {
                "Container": docker_container.id,
                "EndpointConfig": {
                    "Aliases": [container_config.name or container_config.hostname],
                },
            },
        )

    logger.success(
        f"Started container: {container_info['Name']} ({container_config.image}:{container_config.tag})"
    )
    return docker_container


@asynccontextmanager
async def container_session(
    container_config: ContainerConfig,
) -> t.AsyncGenerator[t.Any, ContainerExecFunction]:
    docker_client = aiodocker.Docker()

    # Create network if not specified
    if container_config.network:
        try:
            network = await docker_client.networks.get(container_config.network)
            logger.info(f"Using existing network: {container_config.network}")
        except aiodocker.exceptions.DockerError:
            logger.info(f"Network {container_config.network} not found, creating a new one")
            network = await docker_client.networks.create(
                {
                    "Name": container_config.network,
                    "Driver": "bridge",
                },
            )
            logger.success(f"Created network: {container_config.network}")
    else:
        network_name = f"container-stepper-{uuid.uuid4().hex[:8]}"
        network = await docker_client.networks.create(
            {
                "Name": network_name,
                "Driver": "bridge",
            },
        )
        logger.success(f"Created network: {network_name}")

    # Start container
    container = await start_single_container(docker_client, container_config, network)

    @dn.task(name="execute command")
    async def container_exec(
        cmd: str,
        *,
        timeout: int = 10,
        workdir: str | None = None,
        shell: str = "/bin/sh",
    ) -> tuple[int, str]:
        """Execute a command in the container"""
        exec_ = await container.exec(
            ["timeout", "-k", "1", "-s", "SIGTERM", str(timeout), shell, "-c", cmd],
            privileged=True,
            workdir=workdir,
        )

        output = ""
        async with exec_.start() as stream:
            while True:
                message = await stream.read_out()
                if message is None:
                    break
                output += message.data.decode()

        inspection = await exec_.inspect()
        exit_code = inspection.get("ExitCode", None) or 0
        if exit_code == 137:
            logger.warning(f"Command timed out after {timeout}s")

        return exit_code, output

    try:
        yield container_exec
    finally:
        await container.stop()
        await container.delete()

        # Only delete network if we created it
        if not container_config.network:
            await network.delete()

        await docker_client.close()
