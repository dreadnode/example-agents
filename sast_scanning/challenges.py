import pathlib

import yaml  # type: ignore
from pydantic import BaseModel, Field

current_dir = pathlib.Path(__file__).parent
challenges_dir = current_dir / "challenges"


class Vulnerability(BaseModel):
    id: int
    name: str
    file: pathlib.Path
    function: str
    lines: tuple[int, int]


class Challenge(BaseModel):
    name: str
    src: pathlib.Path
    extensions: list[str]
    strings: dict[str, list[str]]
    vulnerabilities: list[Vulnerability]

    # Track the vulnerability ids that we've already seen
    # (it's here because we already have this object where we need it)
    found: set[int] = Field(default_factory=set)


def load_challenges() -> list[Challenge]:
    with (challenges_dir / "challenges.yml").open() as f:
        return [Challenge(**challenge) for challenge in yaml.safe_load(f)]
