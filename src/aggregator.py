from typing import Dict, Optional


class Aggregator:
    @staticmethod
    def aggregate(scores: dict, weights: dict) -> Optional[float]:
        if not scores:
            return 0.0

        total, total_weight = 0.0, 0.0
        for name, score in scores.items():
            weight = weights.get(name, 1.0)
            total += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0


        return round(total / total_weight , 4)
