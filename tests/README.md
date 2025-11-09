# DSGE Package Tests

This directory contains the test suite for the DSGE estimation framework.

## Test Organization

### Unit Tests
- `test_models.py` - Model specification and parameter tests
- `test_solvers.py` - Linear solver tests
- `test_filters.py` - Kalman filter and smoother tests
- `test_forecasting.py` - Forecasting functionality tests
- `test_occbin.py` - OccBin solver tests
- `test_occbin_filter.py` - OccBin filtering tests
- `test_occbin_estimation.py` - OccBin estimation tests
- `test_data_loading.py` - Data transformation and validation tests
- `test_config.py` - Configuration and settings tests

### Model Tests
- `test_simple_nk_model.py` - Simple New Keynesian model tests
- `test_nyfed_model.py` - NYFed DSGE Model 1002 tests

### Integration Tests
- `test_fred_integration.py` - FRED API integration tests (requires API key)

## Running Tests

### Run All Tests

```bash
uv run pytest tests/
```

### Run Specific Test File

```bash
uv run pytest tests/test_config.py
```

### Run with Verbose Output

```bash
uv run pytest tests/ -v
```

### Run Tests with Coverage

```bash
uv run pytest tests/ --cov=src/dsge --cov-report=html
```

### Skip Slow Tests

Some tests (especially FRED integration tests) can be slow. Skip them with:

```bash
uv run pytest tests/ -m "not slow"
```

## FRED API Integration Tests

Several tests require a valid FRED API key to run. These tests will **automatically skip** if no API key is found.

### Getting a FRED API Key

1. Visit https://fred.stlouisfed.org/docs/api/api_key.html
2. Create a free account
3. Generate an API key

### Setting Up Your API Key

There are two ways to provide your FRED API key:

#### Option 1: Environment Variable

```bash
export FRED_API_KEY="your_api_key_here"
uv run pytest tests/
```

#### Option 2: .env File (Recommended)

1. Copy the template file:
   ```bash
   cp .env.template .env
   ```

2. Edit `.env` and add your key:
   ```
   FRED_API_KEY=your_api_key_here
   ```

3. Run tests:
   ```bash
   uv run pytest tests/
   ```

### Skipped Tests

When no API key is available, you'll see output like this:

```
tests/test_fred_integration.py::TestFREDDownloadIntegration::test_download_gdp_series SKIPPED [  X%]
...
SKIPPED [X] tests/test_fred_integration.py:93: FRED API key not found in environment. Set FRED_API_KEY to run this test.
```

This is expected behavior and means the test infrastructure is working correctly.

## Test Configuration

Test configuration is in `pyproject.toml` under the `[tool.pytest.ini_options]` section.

### Test Markers

- `@pytest.mark.slow` - Tests that take a long time to run
- `@pytest.mark.integration` - Integration tests requiring external resources
- `@pytest.mark.unit` - Unit tests (fast, no external dependencies)

### Running Only Integration Tests

```bash
uv run pytest tests/ -m integration
```

### Running Only Unit Tests

```bash
uv run pytest tests/ -m unit
```

## Current Test Status

As of 2025-11-09:

- **Total Tests**: 118
- **Passing**: 106
- **Skipped**: 12 (FRED API tests without key)
- **Test Coverage**: >80% (target)
- **Test Duration**: ~3.8 minutes (full suite with all dependencies)

### Test Breakdown by Module

| Module | Tests | Status |
|--------|-------|--------|
| Configuration | 18 | ✅ All passing |
| Data Loading | 25 | ✅ All passing |
| Filters | 3 | ✅ All passing |
| Forecasting | 12 | ✅ All passing |
| FRED Integration | 18 | 6 passing, 12 skipped (no API key) |
| Models | 7 | ✅ All passing |
| NYFed Model | 7 | ✅ All passing |
| OccBin Solver | 5 | ✅ All passing |
| OccBin Estimation | 8 | ✅ All passing |
| OccBin Filter | 5 | ✅ All passing |
| Simple NK Model | 6 | ✅ All passing |
| Solvers | 4 | ✅ All passing |

## Writing New Tests

### Test Structure

Follow pytest conventions:

```python
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dsge.module import function_to_test


class TestFeatureName:
    """Tests for specific feature."""

    def test_basic_functionality(self):
        """Test basic use case."""
        result = function_to_test()
        assert result == expected_value

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Using Fixtures

```python
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return create_test_data()

def test_with_fixture(sample_data):
    """Test using fixture."""
    result = process(sample_data)
    assert result is not None
```

### Skipping Tests Conditionally

```python
skip_if_no_resource = pytest.mark.skipif(
    resource_not_available(),
    reason="Resource not available"
)

@skip_if_no_resource
class TestWithResource:
    def test_something(self):
        # This will skip if resource is not available
        pass
```

## Continuous Integration

The test suite is designed to work in CI environments where external resources (like FRED API keys) may not be available. Tests requiring external resources will automatically skip.

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Run tests
        run: uv run pytest tests/
```

## Troubleshooting

### Import Errors

If you see import errors, ensure the project is installed:

```bash
uv sync
```

### FRED API Tests Failing

If FRED API tests are failing (not skipping), check:

1. Is your API key valid?
2. Do you have internet connectivity?
3. Is the FRED API accessible?

You can test your API key manually:

```python
from fredapi import Fred
fred = Fred(api_key='your_key')
data = fred.get_series('GDPC1')
print(data.head())
```

### Tests Running Slowly

The full test suite includes numerical computation tests that can be slow. To run faster:

1. Skip slow tests: `pytest tests/ -m "not slow"`
2. Run specific test files instead of full suite
3. Use pytest-xdist for parallel execution: `pytest tests/ -n auto`

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure tests pass: `pytest tests/`
3. Add appropriate markers (@pytest.mark.slow, etc.)
4. Update this README if adding new test categories
5. Maintain >80% code coverage

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [FRED API documentation](https://fred.stlouisfed.org/docs/api/fred/)
