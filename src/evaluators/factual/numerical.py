"""
Numerical Accuracy Evaluator - Deterministic number extraction and comparison.

This evaluator extracts numerical values from both texts and compares them
with configurable tolerance. Useful for detecting factual errors in numbers,
dates, percentages, and monetary values.
"""

import asyncio
import re
from typing import List, Tuple

from src.base import BaseEvaluator
from src.registry_decorator import register_evaluator
from src.utils.timeit import timeit


@register_evaluator(category="factual", name="numerical_accuracy")
class NumericalAccuracyEvaluator(BaseEvaluator):
    """
    Extracts and compares numerical values between ground truth and candidate.

    Score = number of matched numbers / total numbers in ground truth

    Numbers are matched if they are within the configured tolerance.
    """

    # Regex patterns for different number formats
    PATTERNS = {
        'scientific': r'-?\d+\.?\d*[eE][+-]?\d+',  # 1.23e-4, 6.022E23
        'decimal': r'-?\d+\.\d+',  # 123.456, -0.5
        'integer': r'-?\d+',  # 123, -456
        'percentage': r'\d+\.?\d*\s*%',  # 45%, 12.5%
        'monetary': r'[\$€£¥]\s*[\d,]+\.?\d*',  # $1,234.56, €500
    }

    def __init__(
        self,
        tolerance: float = 0.01,
        match_order: bool = False,
        extract_percentages: bool = True,
        extract_monetary: bool = True,
        **kwargs
    ):
        """
        Initialize the numerical accuracy evaluator.

        Args:
            tolerance: Relative tolerance for number matching (default 1%)
            match_order: If True, numbers must appear in same order
            extract_percentages: Include percentage values
            extract_monetary: Include monetary values
        """
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.match_order = match_order
        self.extract_percentages = extract_percentages
        self.extract_monetary = extract_monetary

        # Build combined pattern
        self._build_pattern()

    def _build_pattern(self):
        """Build the combined regex pattern based on configuration."""
        patterns = [
            self.PATTERNS['scientific'],
            self.PATTERNS['decimal'],
        ]

        if self.extract_percentages:
            patterns.insert(0, self.PATTERNS['percentage'])

        if self.extract_monetary:
            patterns.insert(0, self.PATTERNS['monetary'])

        patterns.append(self.PATTERNS['integer'])

        self._pattern = '|'.join(f'({p})' for p in patterns)

    @property
    def method_name(self):
        return "numerical_accuracy"

    def _clean_number(self, match: str) -> float:
        """Clean and convert a matched string to float."""
        # Remove currency symbols and commas
        cleaned = re.sub(r'[\$€£¥,\s]', '', match)
        # Remove percentage sign but remember it
        is_percentage = '%' in cleaned
        cleaned = cleaned.replace('%', '')

        try:
            value = float(cleaned)
            # Convert percentage to decimal if needed (optional)
            # if is_percentage:
            #     value = value / 100
            return value
        except ValueError:
            return None

    def _extract_numbers(self, text: str) -> List[Tuple[float, str]]:
        """
        Extract all numerical values from text.

        Returns:
            List of (value, original_string) tuples
        """
        numbers = []
        matches = re.finditer(self._pattern, text)

        for match in matches:
            original = match.group(0)
            value = self._clean_number(original)
            if value is not None:
                numbers.append((value, original))

        return numbers

    def _numbers_match(self, gt_val: float, cand_val: float) -> bool:
        """Check if two numbers match within tolerance."""
        if gt_val == cand_val:
            return True

        if gt_val == 0:
            return abs(cand_val) <= self.tolerance

        relative_diff = abs(gt_val - cand_val) / abs(gt_val)
        return relative_diff <= self.tolerance

    def _compute_accuracy(
        self,
        gt_numbers: List[Tuple[float, str]],
        cand_numbers: List[Tuple[float, str]]
    ) -> dict:
        """Compute numerical accuracy metrics."""
        if not gt_numbers:
            # No numbers in ground truth
            if not cand_numbers:
                return {
                    "accuracy": 1.0,
                    "matched": 0,
                    "total": 0,
                    "extra": 0,
                    "details": "No numbers in either text"
                }
            return {
                "accuracy": 1.0,
                "matched": 0,
                "total": 0,
                "extra": len(cand_numbers),
                "details": f"No numbers in GT, {len(cand_numbers)} in candidate"
            }

        if not cand_numbers:
            return {
                "accuracy": 0.0,
                "matched": 0,
                "total": len(gt_numbers),
                "extra": 0,
                "details": f"All {len(gt_numbers)} numbers missing from candidate"
            }

        matched = 0
        matched_details = []
        unmatched_gt = []
        used_cand_indices = set()

        if self.match_order:
            # Order-sensitive matching
            cand_idx = 0
            for gt_val, gt_str in gt_numbers:
                found = False
                while cand_idx < len(cand_numbers):
                    cand_val, cand_str = cand_numbers[cand_idx]
                    cand_idx += 1
                    if self._numbers_match(gt_val, cand_val):
                        matched += 1
                        matched_details.append(f"{gt_str}≈{cand_str}")
                        found = True
                        break
                if not found:
                    unmatched_gt.append(gt_str)
        else:
            # Order-insensitive matching (greedy)
            for gt_val, gt_str in gt_numbers:
                found = False
                for idx, (cand_val, cand_str) in enumerate(cand_numbers):
                    if idx in used_cand_indices:
                        continue
                    if self._numbers_match(gt_val, cand_val):
                        matched += 1
                        matched_details.append(f"{gt_str}≈{cand_str}")
                        used_cand_indices.add(idx)
                        found = True
                        break
                if not found:
                    unmatched_gt.append(gt_str)

        accuracy = matched / len(gt_numbers)
        extra = len(cand_numbers) - len(used_cand_indices)

        details = f"Matched {matched}/{len(gt_numbers)}"
        if unmatched_gt:
            details += f", Missing: {unmatched_gt[:3]}"
            if len(unmatched_gt) > 3:
                details += f"... (+{len(unmatched_gt)-3} more)"

        return {
            "accuracy": accuracy,
            "matched": matched,
            "total": len(gt_numbers),
            "extra": extra,
            "details": details
        }

    @timeit
    async def evaluate(self, ground_truth: str, candidate: str, **kwargs):
        """
        Evaluate numerical accuracy between ground truth and candidate.

        Returns:
            dict with 'score' (accuracy) and 'comment' with details
        """
        try:
            # Extract numbers from both texts
            gt_numbers = await asyncio.to_thread(
                self._extract_numbers, ground_truth
            )
            cand_numbers = await asyncio.to_thread(
                self._extract_numbers, candidate
            )

            # Compute accuracy
            metrics = self._compute_accuracy(gt_numbers, cand_numbers)

            comment = (
                f"Numerical Accuracy: {metrics['accuracy']:.4f} - "
                f"{metrics['details']}"
            )

            return {"score": metrics['accuracy'], "comment": comment}

        except Exception as e:
            return {
                "score": 0.0,
                "comment": f"Numerical accuracy evaluation failed: {str(e)}"
            }
