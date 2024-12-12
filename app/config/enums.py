import enum

class Environment(enum.Enum):
    PROD: str = "PROD"
    DEV: str = "DEV"

class Milestone(enum.Enum):
    ONE: str = "Milestone 1"
    TWO: str = "Milestone 2"
    THREE: str = "Milestone 3"
    FINAL: str = "Final Project"
