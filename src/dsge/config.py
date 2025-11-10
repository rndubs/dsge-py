"""
Configuration and Settings for DSGE Package.

This module provides centralized configuration management using pydantic-settings.
Settings can be loaded from environment variables or a .env file.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DSGESettings(BaseSettings):
    """Base settings class with common fields."""

    fred_api_key: str | None = Field(
        default=None, description="FRED API key for downloading economic data", alias="FRED_API_KEY"
    )


class Settings(DSGESettings):
    """
    Application settings loaded from environment variables or .env file.

    Environment variables can be set directly or loaded from a .env file
    in the project root directory.

    Example .env file:
        FRED_API_KEY=your_api_key_here

    Attributes:
    ----------
    fred_api_key : str, optional
        FRED (Federal Reserve Economic Data) API key for downloading economic data.
        Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
    """

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
        extra="ignore",
    )


# Global settings instance
_settings: DSGESettings | None = None


def get_settings() -> DSGESettings:
    """
    Get the global settings instance.

    This function implements a singleton pattern to ensure settings
    are only loaded once per application run.

    Returns:
    -------
    Settings
        The application settings instance

    Examples:
    --------
    >>> from dsge.config import get_settings
    >>> settings = get_settings()
    >>> api_key = settings.fred_api_key
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings(env_file: str | None = ".env") -> DSGESettings:
    """
    Reload settings from environment variables and .env file.

    This is useful for testing or when settings need to be refreshed
    during runtime.

    Parameters
    ----------
    env_file : str or None, optional
        Path to .env file to load. If None, will not load from .env file.
        Defaults to ".env"

    Returns:
    -------
    Settings
        The newly loaded settings instance

    Examples:
    --------
    >>> from dsge.config import reload_settings
    >>> settings = reload_settings()
    >>> # For testing without .env file:
    >>> settings = reload_settings(env_file=None)
    """
    global _settings

    # Create settings with custom env_file parameter
    if env_file is None:
        # Disable .env file loading by creating a custom Settings class
        # that explicitly sets env_file to a non-existent file
        class SettingsNoEnv(DSGESettings):
            """Settings without .env file loading."""

            model_config = SettingsConfigDict(
                # Don't load from .env file
                env_file_encoding="utf-8",
                populate_by_name=True,
                case_sensitive=False,
                extra="ignore",
            )

        _settings = SettingsNoEnv()
    else:
        _settings = Settings()

    return _settings


def get_fred_api_key() -> str | None:
    """
    Get the FRED API key from settings.

    This is a convenience function that returns the FRED API key
    if it's configured in the environment or .env file.

    Returns:
    -------
    str or None
        The FRED API key if configured, None otherwise

    Examples:
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
    "Settings",
    "get_fred_api_key",
    "get_settings",
    "reload_settings",
]
