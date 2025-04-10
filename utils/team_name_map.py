TEAM_NAME_MAP = {
    "Manchester City FC": "Man City",
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Manchester United FC": "Man United",
    "Tottenham Hotspur FC": "Tottenham",
    "Brighton & Hove Albion FC": "Brighton",
    "Newcastle United FC": "Newcastle",
    "Leicester City FC": "Leicester",
    "Aston Villa FC": "Aston Villa",
    "West Ham United FC": "West Ham",
    "Crystal Palace FC": "Crystal Palace",
    "Brentford FC": "Brentford",
    "Southampton FC": "Southampton",
    "Everton FC": "Everton",
    "Nottingham Forest FC": "Nottm Forest",
    "Wolverhampton Wanderers FC": "Wolves",
    "AFC Bournemouth": "Bournemouth",
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich",
    # Add more as needed
}

def normalize_team(name):
    return TEAM_NAME_MAP.get(name, name)
