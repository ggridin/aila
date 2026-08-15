from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACT_VERSION = "1.0"

WorkerId = Literal["mic", "camera", "filesystem", "speaker", "display"]
ObservationKind = Literal[
    "speech.segment",
    "scene.caption",
    "scene.motion",
    "file.changed",
    "file.created",
    "file.deleted",
    "sensor.status",
]
Verb = Literal["snapshot", "speak", "render", "clear"]
DisplayRenderKind = Literal["text", "markdown", "image"]
FileChange = Literal["changed", "created", "deleted"]

VALID_WORKERS: frozenset[str] = frozenset(
    {"mic", "camera", "filesystem", "speaker", "display"}
)
VALID_OBSERVATION_KINDS: frozenset[str] = frozenset(
    {
        "speech.segment",
        "scene.caption",
        "scene.motion",
        "file.changed",
        "file.created",
        "file.deleted",
        "sensor.status",
    }
)
VALID_VERBS: frozenset[str] = frozenset({"snapshot", "speak", "render", "clear"})

OBSERVATION_KINDS_BY_WORKER: dict[str, frozenset[str]] = {
    "mic": frozenset({"speech.segment", "sensor.status"}),
    "camera": frozenset({"scene.caption", "scene.motion", "sensor.status"}),
    "filesystem": frozenset({"file.changed", "file.created", "file.deleted"}),
    "speaker": frozenset(),
    "display": frozenset(),
}

VERBS_BY_WORKER: dict[str, frozenset[str]] = {
    "mic": frozenset(),
    "camera": frozenset({"snapshot"}),
    "filesystem": frozenset(),
    "speaker": frozenset({"speak"}),
    "display": frozenset({"render", "clear"}),
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Severity(str, Enum):
    info = "info"
    notice = "notice"
    warning = "warning"
    alert = "alert"


class SpeechSegmentPayload(StrictModel):
    text: str = Field(min_length=1)
    lang: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def end_cannot_precede_start(self) -> SpeechSegmentPayload:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        return self


class SensorStatusPayload(StrictModel):
    """Health/status signal from a sensor (e.g. its inference backend is down).

    Lets the agent distinguish a degraded sense ("my ears are offline") from a
    genuine quiet observation.
    """

    component: str = Field(min_length=1)
    state: Literal["ok", "degraded", "unavailable"]
    detail: str = Field(default="", max_length=500)


class SceneBox(StrictModel):
    label: str = Field(min_length=1)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(ge=0)
    h: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)


class SceneCaptionPayload(StrictModel):
    caption: str = Field(min_length=1)
    labels: list[str] = Field(default_factory=list)
    boxes: list[SceneBox] = Field(default_factory=list)


class SceneMotionPayload(StrictModel):
    level: float = Field(ge=0.0, le=1.0)
    region: str = Field(min_length=1)


class FileEventPayload(StrictModel):
    path: str = Field(min_length=1)
    change: FileChange
    size: int = Field(ge=0)
    mtime: datetime

    @field_validator("path")
    @classmethod
    def path_must_be_absolute(cls, value: str) -> str:
        if not (
            PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
        ):
            raise ValueError("path must be absolute")
        return value


class FileChangedPayload(FileEventPayload):
    change: Literal["changed"]


class FileCreatedPayload(FileEventPayload):
    change: Literal["created"]


class FileDeletedPayload(FileEventPayload):
    change: Literal["deleted"]

    @model_validator(mode="after")
    def deleted_files_report_zero_size(self) -> FileDeletedPayload:
        if self.size != 0:
            raise ValueError("deleted file events must report size 0")
        return self


class SnapshotArgs(StrictModel):
    pass


class SnapshotResult(StrictModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    scene_caption: SceneCaptionPayload = Field(alias="scene.caption")


class SpeakArgs(StrictModel):
    text: str = Field(min_length=1)
    voice: str | None = None
    rate: float = Field(default=1.0, gt=0.0)


class SpeakResult(StrictModel):
    duration_ms: int = Field(ge=0)


class RenderArgs(StrictModel):
    kind: DisplayRenderKind
    content: str
    region: str | None = None


class RenderResult(StrictModel):
    rendered: Literal[True]


class ClearArgs(StrictModel):
    pass


class ClearResult(StrictModel):
    cleared: Literal[True]


OBSERVATION_PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    "speech.segment": SpeechSegmentPayload,
    "scene.caption": SceneCaptionPayload,
    "scene.motion": SceneMotionPayload,
    "file.changed": FileChangedPayload,
    "file.created": FileCreatedPayload,
    "file.deleted": FileDeletedPayload,
    "sensor.status": SensorStatusPayload,
}

COMMAND_ARG_MODELS: dict[tuple[str, str], type[BaseModel]] = {
    ("camera", "snapshot"): SnapshotArgs,
    ("speaker", "speak"): SpeakArgs,
    ("display", "render"): RenderArgs,
    ("display", "clear"): ClearArgs,
}

RESULT_DATA_MODELS: tuple[type[BaseModel], ...] = (
    SnapshotResult,
    SpeakResult,
    RenderResult,
    ClearResult,
)
