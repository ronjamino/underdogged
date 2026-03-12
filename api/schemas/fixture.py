from pydantic import BaseModel


class FixtureOut(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    league: str
    match_date: str         # ISO-8601
    matchweek: int | None   # not available in current CSV; always None
    status: str             # "upcoming" | "live" | "completed"
