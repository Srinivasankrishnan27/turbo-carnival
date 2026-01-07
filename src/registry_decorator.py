from src.registry import REGISTRY

def register_evaluator(category: str, name: str):
    def wrapper(reg_cls):
        REGISTRY.register(category, name, reg_cls)
        return reg_cls
    return wrapper