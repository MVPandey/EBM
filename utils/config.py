"""Configuration for the application."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration for the application."""

    model_config = SettingsConfigDict(env_file='.env')

    huggingface_api_token: str = Field(description='The API token for the Hugging Face API.')
    kaggle_api_token: str = Field(description='The API token for the Kaggle API.')


config = Config()  # type: ignore[missing-argument]
