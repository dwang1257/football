from datetime import datetime
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Conf = Annotated[int, Field(strict=True, ge=1, le=5)]
Prob = Annotated[float, Field(gt=0.0, lt=1.0)]
Points = Annotated[float, Field(ge=-3.0, le=3.0)]


class GameOpinion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game: str
    pick: str | None = None
    conf: Conf | None = None
    prob: Prob | None = None
    why: str | None = None
    kickoff: datetime | str | None = None
    model_prob: float | None = None

    @model_validator(mode="after")
    def _one_strength(self) -> Self:
        given = (self.conf is not None) + (self.prob is not None)
        if self.pick is None and given:
            raise ValueError("conf or prob given without a pick")
        if self.pick is not None and given != 1:
            raise ValueError("a pick needs exactly one of conf (1-5) or prob (0-1)")
        return self


class TeamOpinion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team: str
    points: Points
    why: str | None = None


class OpinionFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    season: int | None = None
    week: int | None = None
    games: list[GameOpinion] = Field(default_factory=list[GameOpinion])
    teams: list[TeamOpinion] = Field(default_factory=list[TeamOpinion])

    @field_validator("games", "teams", mode="before")
    @classmethod
    def _none_is_empty(cls, value: Any) -> Any:
        return [] if value is None else value
