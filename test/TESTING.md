# Testing Guide for LLM Sentinel

## Overview

This project includes comprehensive smoke tests that verify core functionality without calling external APIs (Vertex AI, Datadog). All external dependencies are mocked using `unittest.mock`.

## Running Smoke Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
cd sentinel-ai
pytest test_smoke.py -v
```

### Run with Coverage Report

```bash
pytest test_smoke.py -v --cov=sentinel --cov-report=term-missing
```

### Run Specific Test Classes

```bash
# Test only retriever
pytest test_smoke.py::TestRetriever -v

# Test only hallucination detection
pytest test_smoke.py::TestHallucination -v

# Test integration
pytest test_smoke.py::TestIntegration -v
```

## Test Coverage

### What's Tested

1. **Retriever (`TestRetriever`)**
   - Initialization with valid/invalid data
   - Top-K similarity search
   - Empty corpus handling

2. **Hallucination Detection (`TestHallucination`)**
   - Sentence splitting logic
   - Grounding check with mocked embedder
   - Empty answer handling

3. **Document Indexing (`TestIndexing`)**
   - Text chunking with overlap
   - File loading and chunk creation
   - Edge cases (empty text, short text)

4. **Telemetry (`TestTelemetry`)**
   - Token estimation
   - Cost calculation
   - Severity classification
   - Telemetry object building
   - Datadog log formatting

5. **Datadog Client (`TestDatadogClient`)**
   - Metric sending (mocked HTTP)
   - Log sending (mocked HTTP)

6. **Vertex AI Embedder (`TestVertexEmbedder`)**
   - Initialization (mocked Vertex AI)
   - Embedding generation
   - Empty input handling

7. **Vertex AI Gemini Client (`TestVertexGeminiClient`)**
   - Initialization (mocked Vertex AI)
   - Answer generation
   - Latency tracking

8. **Integration (`TestIntegration`)**
   - Full pipeline: embed → retrieve → generate → grounding → telemetry → Datadog
   - All external dependencies mocked

### What's NOT Tested (Requires Live APIs)

These smoke tests do **NOT** test:
- Actual Vertex AI API calls
- Actual Datadog API calls
- Network error handling
- Rate limiting
- Real document corpus loading from `data/docs.txt`

For end-to-end integration testing with live APIs, use a separate integration test suite.

## Test Structure

```
test_smoke.py
├── TestRetriever         - Pure logic, no mocks needed
├── TestHallucination     - Uses mocked embedder
├── TestIndexing          - Uses pytest tmp_path fixture
├── TestTelemetry         - Pure logic, no mocks needed
├── TestDatadogClient     - Mocked requests.post
├── TestVertexEmbedder    - Mocked vertexai SDK
├── TestVertexGeminiClient - Mocked vertexai SDK
└── TestIntegration       - All components mocked
```

## Adding New Tests

When adding new functionality:

1. Add unit tests for pure logic (no mocks)
2. Mock external dependencies (Vertex AI, Datadog, file I/O)
3. Test edge cases (empty inputs, errors)
4. Update integration test if pipeline changes

Example:
```python
@patch('sentinel.module.external_dependency')
def test_new_feature(self, mock_dependency):
    """Test description"""
    mock_dependency.return_value = expected_value

    result = function_under_test()

    assert result == expected_result
    assert mock_dependency.called
```

## Continuous Integration

To run tests in CI/CD:

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests with coverage threshold
pytest test_smoke.py -v --cov=sentinel --cov-fail-under=80

# Generate HTML coverage report
pytest test_smoke.py --cov=sentinel --cov-report=html
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'sentinel'`:
```bash
# Make sure you're in the sentinel-ai directory
cd sentinel-ai
export PYTHONPATH=.
pytest test_smoke.py -v
```

### Mock Errors

If mocks aren't working:
- Check the patch path matches the import location in the module
- Use `@patch('module.where.used')` not `@patch('module.where.defined')`

### Vertex AI Mock Issues

The Vertex AI SDK requires specific mock setups:
```python
@patch('sentinel.embedder.vertexai.init')  # Mock init call
@patch('sentinel.embedder.TextEmbeddingModel.from_pretrained')  # Mock model
```

## Next Steps

After smoke tests pass:
1. Run manual end-to-end test with real APIs
2. Set up integration test environment
3. Configure CI/CD pipeline
4. Add performance/load tests for production readiness
