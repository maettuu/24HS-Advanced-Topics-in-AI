import os
import sys
from app.config.enums import Environment

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.enums import Milestone
from typing import Optional


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=os.path.join(os.path.dirname(__file__), '..', '..', '.env'), env_file_encoding='utf-8')

    bot_username: Optional[str] = None
    bot_password: Optional[str] = None
    default_host_url: str = "https://speakeasy.ifi.uzh.ch"
    listen_freq: int = 2
    utils_path: str = os.path.dirname(__file__).rpartition("app")[0] + "utils"
    too_large_dataset_path: str = os.path.join(utils_path, "too_large_dataset")
    useful_dataset_path: str = os.path.join(utils_path, "useful_dataset")
    milestone: Milestone = Milestone.THREE
    environment: Environment = Environment.DEV if "dev" in sys.argv or "--reload" in sys.argv else Environment.PROD
    use_llm: bool = False

settings = Config()
