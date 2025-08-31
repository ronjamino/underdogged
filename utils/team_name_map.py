import re

TEAM_NAME_MAP = {
    # Premier League teams
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United", 
    "Newcastle United FC": "Newcastle",
    "Tottenham Hotspur FC": "Tottenham",
    "Brighton & Hove Albion FC": "Brighton",
    "Brentford FC": "Brentford",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Leeds United FC": "Leeds",
    "Nottingham Forest FC": "Nottm Forest",
    "AFC Bournemouth": "Bournemouth",
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea", 
    "Liverpool FC": "Liverpool",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Burnley FC": "Burnley",
    "Leicester City FC": "Leicester",
    "Aston Villa FC": "Aston Villa",
    "Crystal Palace FC": "Crystal Palace",
    "Southampton FC": "Southampton",
    "Ipswich Town FC": "Ipswich",
    
    # Championship teams (current 2024-25 season)
    "Burnley FC": "Burnley",  # relegated from PL
    "Sheffield United FC": "Sheffield Utd", 
    "Luton Town FC": "Luton",  # relegated from PL
    "Leicester City FC": "Leicester",  # relegated from PL
    "Leeds United FC": "Leeds",
    "West Bromwich Albion FC": "West Brom",
    "Middlesbrough FC": "Middlesbrough", 
    "Hull City FC": "Hull",
    "Preston North End FC": "Preston",
    "Bristol City FC": "Bristol City",
    "Norwich City FC": "Norwich",
    "Swansea City AFC": "Swansea",
    "Stoke City FC": "Stoke",
    "Derby County FC": "Derby",
    "Queens Park Rangers FC": "QPR",
    "Blackburn Rovers FC": "Blackburn",
    "Cardiff City FC": "Cardiff",
    "Coventry City FC": "Coventry",
    "Millwall FC": "Millwall",
    "Plymouth Argyle FC": "Plymouth",
    "Sheffield Wednesday FC": "Sheffield Wed",
    "Watford FC": "Watford",
    "Portsmouth FC": "Portsmouth",
    "Oxford United FC": "Oxford",
    
    # Common variations
    "Sheff Utd": "Sheffield Utd",
    "Sheffield Utd": "Sheffield Utd",
    "Sheff Wed": "Sheffield Wed", 
    "Sheffield Wed": "Sheffield Wed",
    "West Brom": "West Brom",
    "WBA": "West Brom",
    "QPR": "QPR",
    "Queens Park Rangers": "QPR",
}

_STRIP_WORDS = {"fc", "afc", "cfc", "city", "united", "town", "rovers", "albion"}
_SUFFIXES = [
    r"\s+fc$", r"\s+afc$", r"\s+cf$", r"\s+cfc$", r"\s+football club$",
    r"\s+united fc$", r"\s+city fc$", r"\s+town fc$", r"\s+athletic fc$",
    r"\s+rovers fc$", r"\s+albion fc$", r"\s+county fc$", r"\s+wednesday fc$",
]

def _strip_suffixes(name: str) -> str:
    """Remove common football club suffixes"""
    n = name
    for pat in _SUFFIXES:
        n = re.sub(pat, "", n, flags=re.I)
    return n

def _squash_spaces(name: str) -> str:
    """Normalize whitespace"""
    return re.sub(r"\s+", " ", name).strip()

def _handle_common_abbreviations(name: str) -> str:
    """Handle common abbreviations manually"""
    name_lower = name.lower()
    
    # Sheffield teams
    if "sheffield u" in name_lower or "sheff u" in name_lower:
        return "Sheffield Utd"
    if "sheffield w" in name_lower or "sheff w" in name_lower:
        return "Sheffield Wed"
        
    # West Bromwich
    if "west brom" in name_lower or "wba" in name_lower:
        return "West Brom" 
        
    # QPR
    if "qpr" in name_lower or "queens park" in name_lower:
        return "QPR"
        
    return name

def canonicalize_team(name: str) -> str:
    """Convert team name to canonical form"""
    if not name:
        return name
        
    # Explicit mapping first
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    
    # Handle common abbreviations
    abbrev_result = _handle_common_abbreviations(name)
    if abbrev_result != name:
        return abbrev_result
        
    # Generic cleanup
    n = _strip_suffixes(name)
    
    # Remove standalone noise words but be careful not to remove important parts
    words = n.split()
    if len(words) > 1:  # Only remove if multiple words
        n = " ".join(w for w in words if w.lower() not in _STRIP_WORDS or len(words) <= 2)
    
    n = _squash_spaces(n)
    
    # Final check in map
    return TEAM_NAME_MAP.get(n, n)

def normalize_team(name: str) -> str:
    """Main function to normalize team names (keeps old API)"""
    return canonicalize_team(name)

# Test function to help identify unmapped teams
def find_unmapped_teams(df):
    """Find teams that might need manual mapping"""
    all_teams = set()
    all_teams.update(df['home_team'].unique())
    all_teams.update(df['away_team'].unique()) 
    
    unmapped = []
    for team in sorted(all_teams):
        normalized = normalize_team(team)
        if normalized == team and team not in TEAM_NAME_MAP:
            unmapped.append(team)
    
    return unmapped

if __name__ == "__main__":
    # Test some Championship team names
    test_names = [
        "Sheffield United FC",
        "Sheffield Wednesday FC", 
        "West Bromwich Albion FC",
        "Queens Park Rangers FC",
        "Leicester City FC",
        "Luton Town FC"
    ]
    
    print("Testing Championship team name normalization:")
    for name in test_names:
        normalized = normalize_team(name)
        print(f"{name:25} -> {normalized}")