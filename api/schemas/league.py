from pydantic import BaseModel


class LeagueOut(BaseModel):
    id: str             # canonical code: "PL", "ELC", "BL1", "SA", "PD"
    name: str           # display name
    country: str
    match_count: int    # number of upcoming predicted fixtures
    last_updated: str   # ISO-8601 timestamp of latest CSV file
