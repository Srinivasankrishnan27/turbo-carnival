


class Aggregator:
    @staticmethod
    def aggregate(scores: dict, weights: dict) -> float:
        if not scores:
            return 0.0

        total, wsum = 0.0, 0.0
        for k, v in scores.items():
            w = weights.get(k, 0.0)
            total += v * w
            wsum += w

        return round(total / wsum, 4) if wsum else 0.0