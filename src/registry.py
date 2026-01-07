from collections import defaultdict
from typing import Callable, Dict

class EvaluatorRegistry:
    """Registry for evaluators, allowing them to be looked up by name."""

    def __init__(self):
        self._registry: Dict[str, Callable] = defaultdict(dict)
        

    def register(self, category: str, name: str, factory: Callable):
        """Decorator to register an evaluator."""
        if name in self._registry:
            raise ValueError(f"Duplicate Evaluator '{category}'.'{name}' already registered.")
        self._registry[category][name] = factory

    def get(self, category: str, name: str, **kwargs) -> Callable:
        """Get an evaluator factory by name."""
        factory = self._registry.get(category, {}).get(name)
        if not factory:
            print(f"[WARN] Evaluator '{category}'.'{name}' not registered - Skipping")
        
        try:
            return factory(**kwargs)
        except Exception as e:
            print(f"[ERRRO] Failed to instantiate evaluator '{category}'.'{name}': Error: {str(e)} - Skipping")
    
    def available(self):
        return dict(self._registry)
        
        
REGISTRY = EvaluatorRegistry()