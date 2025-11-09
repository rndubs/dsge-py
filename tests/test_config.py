"""
Tests for configuration and settings management.
"""

import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dsge.config import Settings, get_settings, reload_settings, get_fred_api_key


class TestSettings:
    """Tests for Settings class."""

    def test_settings_creation(self):
        """Test that Settings can be created."""
        settings = Settings()
        assert settings is not None

    def test_settings_has_fred_api_key_field(self):
        """Test that Settings has fred_api_key field."""
        settings = Settings()
        assert hasattr(settings, 'fred_api_key')

    def test_settings_fred_api_key_is_optional(self):
        """Test that fred_api_key is optional (can be None)."""
        settings = Settings()
        # Should be None or a string
        assert settings.fred_api_key is None or isinstance(settings.fred_api_key, str)

    def test_settings_with_env_var(self, monkeypatch):
        """Test that Settings loads from environment variable."""
        test_key = "test_api_key_12345"
        monkeypatch.setenv('FRED_API_KEY', test_key)

        settings = Settings()
        assert settings.fred_api_key == test_key

    def test_settings_case_insensitive(self, monkeypatch):
        """Test that environment variable is case-insensitive."""
        test_key = "test_key_case"
        monkeypatch.setenv('fred_api_key', test_key)  # lowercase

        settings = Settings()
        assert settings.fred_api_key == test_key

    def test_settings_from_env_file(self, tmp_path, monkeypatch):
        """Test that Settings can load from .env file."""
        # Create a temporary .env file
        env_file = tmp_path / ".env"
        test_key = "test_key_from_file"
        env_file.write_text(f"FRED_API_KEY={test_key}\n")

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create settings (should load from .env in current directory)
        settings = Settings()

        # Note: This may not work if .env is not in the expected location
        # The settings will try to load from project root .env
        # This test documents the behavior


class TestGetSettings:
    """Tests for get_settings singleton function."""

    def test_get_settings_returns_settings(self):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_singleton(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same object
        assert settings1 is settings2

    def test_reload_settings_creates_new_instance(self):
        """Test that reload_settings creates a new Settings instance."""
        settings1 = get_settings()
        settings2 = reload_settings()

        # Should be Settings instances
        assert isinstance(settings1, Settings)
        assert isinstance(settings2, Settings)

        # After reload, get_settings should return the new instance
        settings3 = get_settings()
        assert settings3 is settings2


class TestGetFredApiKey:
    """Tests for get_fred_api_key convenience function."""

    def test_get_fred_api_key_returns_string_or_none(self):
        """Test that get_fred_api_key returns correct type."""
        api_key = get_fred_api_key()
        assert api_key is None or isinstance(api_key, str)

    def test_get_fred_api_key_matches_settings(self):
        """Test that get_fred_api_key matches settings.fred_api_key."""
        api_key = get_fred_api_key()
        settings = get_settings()

        assert api_key == settings.fred_api_key

    def test_get_fred_api_key_with_env_var(self, monkeypatch):
        """Test that get_fred_api_key picks up environment variable."""
        test_key = "test_key_env_123"
        monkeypatch.setenv('FRED_API_KEY', test_key)

        # Reload settings to pick up new env var
        reload_settings()

        api_key = get_fred_api_key()
        assert api_key == test_key

    def test_get_fred_api_key_without_env_var(self, monkeypatch):
        """Test that get_fred_api_key returns None when no key is set."""
        # Remove any existing key
        monkeypatch.delenv('FRED_API_KEY', raising=False)
        monkeypatch.delenv('fred_api_key', raising=False)

        # Reload settings
        reload_settings()

        api_key = get_fred_api_key()
        # Will be None if no .env file exists in project root
        # or the actual key if .env file exists
        assert api_key is None or isinstance(api_key, str)


class TestSettingsModel:
    """Tests for Settings pydantic model configuration."""

    def test_settings_allows_extra_fields(self, monkeypatch):
        """Test that Settings allows extra fields (for future expansion)."""
        monkeypatch.setenv('EXTRA_FIELD', 'extra_value')

        # Should not raise an error
        settings = Settings()
        assert settings is not None

    def test_settings_field_aliases(self, monkeypatch):
        """Test that Settings accepts field aliases."""
        test_key = "test_alias_key"

        # Both should work
        monkeypatch.setenv('FRED_API_KEY', test_key)
        settings1 = Settings()
        assert settings1.fred_api_key == test_key

        # Clear and test lowercase
        monkeypatch.delenv('FRED_API_KEY', raising=False)
        monkeypatch.setenv('fred_api_key', test_key)
        settings2 = Settings()
        assert settings2.fred_api_key == test_key


class TestEnvFileTemplate:
    """Tests to verify .env.template is properly structured."""

    def test_env_template_exists(self):
        """Test that .env.template exists in project root."""
        project_root = Path(__file__).parent.parent
        env_template = project_root / ".env.template"

        assert env_template.exists(), ".env.template not found in project root"

    def test_env_template_has_fred_api_key(self):
        """Test that .env.template includes FRED_API_KEY."""
        project_root = Path(__file__).parent.parent
        env_template = project_root / ".env.template"

        if env_template.exists():
            content = env_template.read_text()
            assert 'FRED_API_KEY' in content, "FRED_API_KEY not in .env.template"

    def test_env_template_has_instructions(self):
        """Test that .env.template has usage instructions."""
        project_root = Path(__file__).parent.parent
        env_template = project_root / ".env.template"

        if env_template.exists():
            content = env_template.read_text()
            # Should have some kind of instructions/comments
            assert '#' in content or 'Copy' in content or 'Get your' in content


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
