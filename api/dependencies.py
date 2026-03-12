"""
Shared FastAPI dependencies.
"""
from typing import Annotated
from fastapi import Depends
from api.database import DataStore, get_db

DB = Annotated[DataStore, Depends(get_db)]

VALID_LEAGUES = {"PL", "ELC", "BL1", "SA", "PD"}
