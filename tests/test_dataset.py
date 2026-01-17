import pytest
import os
import json
from src.dataset import Dataset, EvaluationItem

def test_dataset_valid_jsonl(tmp_path):
    # Create a dummy valid JSONL
    data = [
        {"ground_truth": "GT 1", "candidate": "Cand 1", "id": "1"},
        {"ground_truth": "GT 2", "candidate": "Cand 2"}
    ]
    p = tmp_path / "valid.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    
    ds = Dataset.load(str(p))
    assert len(ds) == 2
    assert ds.items[0].ground_truth == "GT 1"
    assert ds.items[1].id is None

def test_dataset_invalid_schema(tmp_path):
    # Create invalid JSONL (missing candidate)
    p = tmp_path / "invalid.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write('{"ground_truth": "ok"}\n') # Missing candidate
        f.write('{"ground_truth": "ok", "candidate": "ok"}\n') # Valid
    
    # Should load only the valid one
    ds = Dataset.load(str(p))
    assert len(ds) == 1
    assert ds.items[0].ground_truth == "ok"

def test_dataset_bad_json(tmp_path):
    p = tmp_path / "bad_json.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write('Not valid JSON\n')
        f.write('{"ground_truth": "ok", "candidate": "ok"}\n')
    
    ds = Dataset.load(str(p))
    assert len(ds) == 1
