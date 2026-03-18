from pydantic import BaseModel, model_validator


class PredictionOut(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    league: str
    match_date: str             # ISO-8601: "YYYY-MM-DDTHH:MM:SSZ"
    prob_home: float            # model probability for home win
    prob_draw: float            # model probability for draw
    prob_away: float            # model probability for away win
    predicted_outcome: str      # "H", "D", or "A"
    confidence: float           # max(prob_home, prob_draw, prob_away)
    odds_home: float | None     # bookmaker decimal odds (B365 / Odds API)
    odds_draw: float | None
    odds_away: float | None
    value_bet: str | None       # "H", "D", "A", or None (positive EV outcome)

    # Actual result (populated after match is played)
    actual_result: str | None = None   # 'H', 'D', 'A' or None
    home_score: int | None = None
    away_score: int | None = None

    # Form
    home_form_winrate: float | None = None
    away_form_winrate: float | None = None
    home_momentum: float | None = None
    away_momentum: float | None = None
    home_venue_draw_rate: float | None = None
    away_venue_draw_rate: float | None = None
    # Form sequences: comma-separated W/D/L oldest→newest, team's own perspective
    home_form: str | None = None
    away_form: str | None = None
    h2h_form: str | None = None

    # Head to head
    h2h_home_winrate: float | None = None
    h2h_draw_rate: float | None = None
    h2h_total_goals: float | None = None

    # Goals
    home_avg_goals_scored: float | None = None
    home_avg_goals_conceded: float | None = None
    away_avg_goals_scored: float | None = None
    away_avg_goals_conceded: float | None = None
    expected_total_goals: float | None = None

    @model_validator(mode="after")
    def check_probs_sum(self) -> "PredictionOut":
        total = self.prob_home + self.prob_draw + self.prob_away
        if not (0.90 <= total <= 1.10):
            raise ValueError(
                f"Probabilities must sum to ~1.0 (got {total:.3f}): "
                f"H={self.prob_home}, D={self.prob_draw}, A={self.prob_away}"
            )
        return self
