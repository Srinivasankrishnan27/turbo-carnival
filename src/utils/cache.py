import sqlite3
import json
import hashlib
import os
from typing import Optional, Dict, Any

class CacheManager:
    """
    Manages a local SQLite cache for evaluation results to avoid re-computation.
    """
    def __init__(self, project_name: str = "default", enabled: bool = True):
        self.enabled = enabled
        self.project_name = project_name
        # Secure filename: ensure no path traversal or weird chars
        safe_name = "".join([c for c in project_name if c.isalnum() or c in ('-', '_')])
        if not safe_name: 
            safe_name = "default"
            
        self.db_path = f".cache_{safe_name}.db"
        if self.enabled:
            self._init_db()

    def _init_db(self):
        """Initialize the SQLite database and create table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def _generate_key(self, method_name: str, ground_truth: str, candidate: str, kwargs: Dict[str, Any]) -> str:
        """Generate a unique SHA256 hash key based on inputs."""
        # Normalize inputs to ensure consistent keys
        payload = {
            "method": method_name,
            "gt": ground_truth,
            "cand": candidate,
            "kwargs": kwargs # Includes model params, etc.
        }
        # Sort keys to ensure deterministic JSON
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()

    def get(self, method_name: str, ground_truth: str, candidate: str, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve a result from the cache if it exists."""
        if not self.enabled:
            return None

        key = self._generate_key(method_name, ground_truth, candidate, kwargs)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM evaluation_cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return json.loads(row[0])
        except Exception as e:
            print(f"[WARN] Cache read failed: {e}")
        
        return None

    def set(self, method_name: str, ground_truth: str, candidate: str, kwargs: Dict[str, Any], result: Dict[str, Any]):
        """Store a result in the cache."""
        if not self.enabled or result is None:
            return

        key = self._generate_key(method_name, ground_truth, candidate, kwargs)
        value = json.dumps(result)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO evaluation_cache (key, value) VALUES (?, ?)", (key, value))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[WARN] Cache write failed: {e}")
