import pytest
import asyncio
from unittest.mock import MagicMock, patch
from src.aggregator import Aggregator
from src.pipeline import run_pipeline
from src.registry import REGISTRY

# --- Test Aggregator ---
def test_aggregator_weighted_average():
    scores = {"metric_a": 1.0, "metric_b": 0.5}
    weights = {"metric_a": 0.5, "metric_b": 0.5} # Total weight 1.0 -> (0.5 + 0.25) = 0.75
    result = Aggregator.aggregate(scores, weights)
    assert result == 0.75

def test_aggregator_no_weights():
    scores = {"metric_a": 1.0, "metric_b": 0.0}
    weights = {} # Defaults to 1.0
    result = Aggregator.aggregate(scores, weights)
    assert result == 0.5

def test_aggregator_empty():
    assert Aggregator.aggregate({}, {}) == 0.0

# --- Test Pipeline ---
@pytest.mark.asyncio
async def test_run_pipeline_mock():
    # Mock configs
    mock_base_config = {
        "evaluators": {
            "test_category": {
                "mock_eval": 1.0
            }
        }
    }
    
    # Register a mock evaluator
    mock_evaluator = MagicMock()
    # Make evaluate an async method (return a future or be an AsyncMock)
    from unittest.mock import AsyncMock
    mock_evaluator.evaluate = AsyncMock(return_value={"score": 0.9, "comment": "good"})
    
    with patch("src.pipeline.yaml.safe_load", return_value=mock_base_config), \
         patch("src.pipeline.load_runtime_config", return_value={}), \
         patch.object(REGISTRY, "get", return_value=mock_evaluator), \
         patch("builtins.open", new_callable=MagicMock), \
         patch("src.utils.cache.CacheManager") as MockCache:
        
        # Mock CacheManager instance
        mock_cache_instance = MockCache.return_value
        mock_cache_instance.get.return_value = None # No cache hit
        
        results = await run_pipeline("gt", "cand", "dummy_config.yaml", "dummy_runtime.yaml", use_cache=False)
        
        assert "test_category" in results
        assert results["test_category"]["final_score"] == 0.9
        assert results["test_category"]["evaluators"]["mock_eval"]["score"] == 0.9

# --- Test Orchestrator instantiation (PydanticAI check) ---
def test_orchestrator_init():
    # Only check if we can import and instantiate without basic errors
    # (requires setting dummy prompts to avoid file read error if mocked incompletely)
    with patch("builtins.open", create=True) as mock_open, \
         patch("yaml.safe_load", return_value={"orchestrator": {"system_prompt": "sys", "user_prompt_template": "usr"}}):
        
        from src.orchestrator import Orchestrator
        # Mocking Agent if needed, but for init check just ensuring no crash
        try:
             orch = Orchestrator(base_url="http://test", api_key="test", model_name="openai:gpt-4")
             assert orch.agent is not None
        except Exception as e:
             pytest.fail(f"Orchestrator init failed: {e}")
