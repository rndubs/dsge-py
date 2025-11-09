"""
Configuration and Settings for DSGE Package

This module provides centralized configuration management using pydantic-settings.
Settings can be loaded from environment variables or a .env file.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.

    Environment variables can be set directly or loaded from a .env file
    in the project root directory.

    Example .env file:
        FRED_API_KEY=your_api_key_here

    Attributes
    ----------
    fred_api_key : str, optional
        FRED (Federal Reserve Economic Data) API key for downloading economic data.
        Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
    """

    fred_api_key: Optional[str] = Field(
        default=None,
        description="FRED API key for downloading economic data",
        alias="FRED_API_KEY"
    )

    model_config = SettingsConfigDict(
        # Look for .env file in project root
        env_file=".env",
        # Don't error if .env file doesn't exist
        env_file_encoding="utf-8",
        # Allow both FRED_API_KEY and fred_api_key
        populate_by_name=True,
        # Make validation case-insensitive
        case_sensitive=False,
        # Allow extra fields (for future expansion)
        extra="ignore"
    )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.

    This function implements a singleton pattern to ensure settings
    are only loaded once per application run.

    Returns
    -------
    Settings
        The application settings instance

    Examples
    --------
    >>> from dsge.config import get_settings
    >>> settings = get_settings()
    >>> api_key = settings.fred_api_key
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment variables and .env file.

    This is useful for testing or when settings need to be refreshed
    during runtime.

    Returns
    -------
    Settings
        The newly loaded settings instance

    Examples
    --------
    >>> from dsge.config import reload_settings
    >>> settings = reload_settings()
    """
    global _settings
    _settings = Settings()
    return _settings


def get_fred_api_key() -> Optional[str]:
    """
    Get the FRED API key from settings.

    This is a convenience function that returns the FRED API key
    if it's configured in the environment or .env file.

    Returns
    -------
    str or None
        The FRED API key if configured, None otherwise

    Examples
    --------
    >>> from dsge.config import get_fred_api_key
    >>> api_key = get_fred_api_key()
    >>> if api_key:
    ...     print("API key is configured")
    ... else:
    ...     print("No API key found")
    """
    return get_settings().fred_api_key


__all__ = [
    'Settings',
    'get_settings',
    'reload_settings',
    'get_fred_api_key',
]
