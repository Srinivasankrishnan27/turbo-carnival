import json
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError

class EvaluationItem(BaseModel):
    ground_truth: str = Field(..., min_length=1, description="The reference text.")
    candidate: str = Field(..., min_length=1, description="The generated text to evaluate.")
    id: Optional[str] = Field(None, description="Optional unique identifier for the item.")
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary metadata.")

class Dataset:
    """
    Handles loading and validation of evaluation datasets.
    """
    def __init__(self, items: List[EvaluationItem]):
        self.items = items

    @classmethod
    def load(cls, file_path: str) -> 'Dataset':
        """
        Loads a dataset from a file (.json or .jsonl), handling both 
        JSON Arrays (List[Dict]) and JSON Lines (Stream of Dicts).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        valid_items = []
        errors = []

        # Strategy 1: Try reading as entire JSON Array (preferred for .json)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    for i, data in enumerate(content):
                        try:
                            item = EvaluationItem(**data)
                            valid_items.append(item)
                        except ValidationError as e:
                            errors.append(f"Item {i}: Schema Validation Error - {e}")
                    
                    if valid_items or errors:
                        return cls._handle_results(valid_items, errors)
        except json.JSONDecodeError:
            # Not a valid JSON Array, fall back to Line-by-Line (JSONL)
            pass

        # Strategy 2: Line-by-Line (JSONL)
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Handle case where file is a single JSON object not in list
                    if not isinstance(data, dict):
                         errors.append(f"Line {i}: Expected dict, got {type(data)}")
                         continue
                    
                    item = EvaluationItem(**data)
                    valid_items.append(item)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: Invalid JSON - {e}")
                except ValidationError as e:
                    errors.append(f"Line {i}: Schema Validation Error - {e}")

        return cls._handle_results(valid_items, errors)

    @classmethod
    def _handle_results(cls, valid_items, errors) -> 'Dataset':
        if errors:
            print(f"[WARN] {len(errors)} items failed validation and were skipped:")
            for err in errors[:5]:
                print(f"  - {err}")
            if len(errors) > 5:
                print(f"  - ... and {len(errors) - 5} more.")

        if not valid_items and errors:
            print("[ERROR] No valid items found in dataset.")
        
        return cls(valid_items)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
