"""Public request contracts for the constrained synthetic-data service."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Identifier = Annotated[str, Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]
Description = Annotated[str, Field(min_length=1, max_length=6000)]


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class TaskSpec(StrictBody):
    schema_version: Literal[1] = 1
    robot: Literal["franka_panda"] = "franka_panda"
    template: Literal["pick_place"] = "pick_place"
    object_color: Literal["red", "blue", "green"] = "red"
    destination: Literal["right"] = "right"
    scene: Literal["table"] = "table"
    camera: Literal["fixed"] = "fixed"
    width: Literal[864] = 864
    height: Literal[480] = 480
    fps: Literal[24] = 24
    duration_seconds: Literal[5] = 5
    seed: Literal[7] = 7

    @field_validator("schema_version", "width", "height", "fps", "duration_seconds", "seed", mode="before")
    @classmethod
    def exact_integer(cls, value):
        if type(value) is not int:
            raise ValueError("Must be an integer, not a string, boolean or float")
        return value


class SpecBody(StrictBody):
    description: Description
    appearance_description: str = Field(default="", max_length=6000)


class AssetBody(StrictBody):
    upload_id: Identifier
    description: str = Field(default="", max_length=6000)


class ReferenceBody(StrictBody):
    """Exactly one frozen source specification or an existing media asset."""

    spec_id: Identifier | None = None
    spec_revision: int | None = Field(default=None, ge=1)
    source_asset_id: Identifier | None = None
    appearance_description: str = Field(default="", max_length=6000)

    @model_validator(mode="after")
    def one_source(self):
        if bool(self.spec_id) == bool(self.source_asset_id):
            raise ValueError("Provide exactly one of spec_id or source_asset_id")
        if self.spec_id and self.spec_revision is None:
            raise ValueError("spec_revision is required with spec_id")
        if self.source_asset_id and self.spec_revision is not None:
            raise ValueError("spec_revision is only allowed with spec_id")
        return self


class PreflightBody(ReferenceBody):
    check_provider: bool = False
    known_task_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[0-9]+$")


class QuoteBody(ReferenceBody):
    model: Literal["MiniMax-H3"] = "MiniMax-H3"
    duration_seconds: int | None = Field(default=None, ge=4, le=15)
    resolution: Literal["768P"] = "768P"
    variants: Literal[1] = 1

    @field_validator("duration_seconds", "variants", mode="before")
    @classmethod
    def exact_integer(cls, value):
        if value is None:
            return value
        if type(value) is not int:
            raise ValueError("Must be an integer")
        return value


class RenderOptions(StrictBody):
    model: Literal["MiniMax-H3"] = "MiniMax-H3"
    duration_seconds: int | None = Field(default=None, ge=4, le=15)
    resolution: Literal["768P"] = "768P"
    variants: Literal[1] = 1
    appearance_description: Description
    quote_id: Identifier
    max_cost_minor_units: int = Field(gt=0, le=1_000_000)
    currency: Literal["CNY"] = "CNY"
    automatic_paid_retries: Literal[False] = False

    @field_validator("duration_seconds", "variants", mode="before")
    @classmethod
    def exact_integer(cls, value):
        if value is None:
            return value
        if type(value) is not int:
            raise ValueError("Must be an integer")
        return value

    @field_validator("automatic_paid_retries", mode="before")
    @classmethod
    def false_only(cls, value):
        if value is not False:
            raise ValueError("Automatic paid retries are disabled")
        return value


class JobBody(StrictBody):
    mode: Literal["scene_only", "scene_and_render", "uploaded_video_render"]
    spec_id: Identifier | None = None
    spec_revision: int | None = Field(default=None, ge=1)
    source_asset_id: Identifier | None = None
    render: RenderOptions | None = None

    @model_validator(mode="after")
    def fields_for_mode(self):
        if self.mode in {"scene_only", "scene_and_render"}:
            if not self.spec_id or self.spec_revision is None:
                raise ValueError("Scene jobs require spec_id and spec_revision")
            if self.source_asset_id is not None:
                raise ValueError("Scene jobs cannot use source_asset_id")
        else:
            if not self.source_asset_id or self.spec_id is not None or self.spec_revision is not None:
                raise ValueError("Upload render jobs require only source_asset_id")
        if self.mode == "scene_only" and self.render is not None:
            raise ValueError("scene_only must not contain a paid render request")
        if self.mode != "scene_only" and self.render is None:
            raise ValueError("Render settings and an accepted quote are required")
        return self


class ExportBody(StrictBody):
    asset_id: Identifier
    kind: Literal["available", "semantic_gt", "media"] = "available"
    purpose: Literal["research", "preview", "action_supervised_training"] = "research"


class ReconcileBody(StrictBody):
    task_id: str = Field(min_length=1, max_length=128, pattern=r"^[0-9]+$")
    note: str = Field(min_length=1, max_length=1000)


class MediaLinkBody(StrictBody):
    filename: str = Field(min_length=1, max_length=256)
    expires_seconds: int = Field(default=300, ge=30, le=900)
