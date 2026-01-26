"""
Entity Matching Evaluator - Named Entity Recognition and comparison.

This evaluator extracts named entities (people, organizations, locations, etc.)
from both texts and computes precision, recall, and F1 score of entity overlap.
"""

import asyncio
from typing import Set, Tuple, Dict, Optional

from src.base import BaseEvaluator
from src.registry_decorator import register_evaluator
from src.utils.timeit import timeit

# Try to import spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@register_evaluator(category="factual", name="entity_matching")
class EntityMatchingEvaluator(BaseEvaluator):
    """
    Extracts and compares named entities between ground truth and candidate.

    Uses spaCy for Named Entity Recognition (NER).
    Computes F1 score of entity overlap.
    """

    # Entity type weights for weighted scoring
    DEFAULT_WEIGHTS = {
        "PERSON": 1.0,      # People, including fictional
        "ORG": 1.0,         # Companies, agencies, institutions
        "GPE": 1.0,         # Countries, cities, states
        "LOC": 0.8,         # Non-GPE locations
        "DATE": 0.7,        # Absolute or relative dates
        "TIME": 0.7,        # Times
        "MONEY": 0.9,       # Monetary values
        "PERCENT": 0.7,     # Percentages
        "PRODUCT": 0.8,     # Objects, vehicles, foods
        "EVENT": 0.8,       # Named events
        "WORK_OF_ART": 0.7, # Titles of works
        "LAW": 0.8,         # Named laws
        "LANGUAGE": 0.6,    # Languages
        "FAC": 0.7,         # Facilities
        "NORP": 0.7,        # Nationalities, religious, political groups
        "QUANTITY": 0.6,    # Measurements
        "ORDINAL": 0.5,     # "first", "second"
        "CARDINAL": 0.5,    # Numerals that aren't other types
    }

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        match_strategy: str = "exact",
        case_sensitive: bool = False,
        use_weighted_scoring: bool = False,
        entity_types: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize the entity matching evaluator.

        Args:
            spacy_model: Name of spaCy model to use
            match_strategy: "exact" or "fuzzy" (fuzzy uses substring matching)
            case_sensitive: Whether entity matching is case-sensitive
            use_weighted_scoring: Use entity type weights for scoring
            entity_types: List of entity types to consider (None = all)
        """
        super().__init__(**kwargs)
        self.spacy_model = spacy_model
        self.match_strategy = match_strategy
        self.case_sensitive = case_sensitive
        self.use_weighted_scoring = use_weighted_scoring
        self.entity_types = set(entity_types) if entity_types else None

        # Lazy load spaCy model
        self._nlp = None

    def _load_model(self):
        """Load spaCy model (lazy loading)."""
        if self._nlp is None:
            if not SPACY_AVAILABLE:
                raise ImportError(
                    "spaCy is not installed. Install with: pip install spacy && "
                    f"python -m spacy download {self.spacy_model}"
                )
            try:
                self._nlp = spacy.load(self.spacy_model)
            except OSError:
                raise ImportError(
                    f"spaCy model '{self.spacy_model}' not found. "
                    f"Download with: python -m spacy download {self.spacy_model}"
                )
        return self._nlp

    @property
    def method_name(self):
        return "entity_matching"

    def _extract_entities(self, text: str) -> Set[Tuple[str, str]]:
        """
        Extract named entities from text.

        Returns:
            Set of (entity_text, entity_label) tuples
        """
        nlp = self._load_model()
        doc = nlp(text)

        entities = set()
        for ent in doc.ents:
            # Filter by entity type if specified
            if self.entity_types and ent.label_ not in self.entity_types:
                continue

            ent_text = ent.text if self.case_sensitive else ent.text.lower()
            # Clean whitespace
            ent_text = ' '.join(ent_text.split())

            entities.add((ent_text, ent.label_))

        return entities

    def _entities_match(
        self,
        gt_ent: Tuple[str, str],
        cand_ent: Tuple[str, str]
    ) -> bool:
        """Check if two entities match based on strategy."""
        gt_text, gt_label = gt_ent
        cand_text, cand_label = cand_ent

        # Labels must match
        if gt_label != cand_label:
            return False

        if self.match_strategy == "exact":
            return gt_text == cand_text
        elif self.match_strategy == "fuzzy":
            # Substring matching
            return gt_text in cand_text or cand_text in gt_text
        else:
            return gt_text == cand_text

    def _compute_metrics(
        self,
        gt_entities: Set[Tuple[str, str]],
        cand_entities: Set[Tuple[str, str]]
    ) -> Dict:
        """Compute precision, recall, and F1 for entity matching."""
        if not gt_entities and not cand_entities:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "matched": 0,
                "gt_count": 0,
                "cand_count": 0,
                "details": "No entities in either text"
            }

        if not gt_entities:
            return {
                "precision": 0.0 if cand_entities else 1.0,
                "recall": 1.0,
                "f1": 0.0 if cand_entities else 1.0,
                "matched": 0,
                "gt_count": 0,
                "cand_count": len(cand_entities),
                "details": f"No entities in GT, {len(cand_entities)} in candidate"
            }

        if not cand_entities:
            return {
                "precision": 1.0,
                "recall": 0.0,
                "f1": 0.0,
                "matched": 0,
                "gt_count": len(gt_entities),
                "cand_count": 0,
                "details": f"All {len(gt_entities)} entities missing from candidate"
            }

        # Find matches
        matched_gt = set()
        matched_cand = set()
        matched_pairs = []

        for gt_ent in gt_entities:
            for cand_ent in cand_entities:
                if cand_ent in matched_cand:
                    continue
                if self._entities_match(gt_ent, cand_ent):
                    matched_gt.add(gt_ent)
                    matched_cand.add(cand_ent)
                    matched_pairs.append((gt_ent[0], cand_ent[0]))
                    break

        if self.use_weighted_scoring:
            # Weighted scoring
            gt_weight = sum(
                self.DEFAULT_WEIGHTS.get(label, 0.5)
                for _, label in gt_entities
            )
            cand_weight = sum(
                self.DEFAULT_WEIGHTS.get(label, 0.5)
                for _, label in cand_entities
            )
            matched_weight = sum(
                self.DEFAULT_WEIGHTS.get(label, 0.5)
                for _, label in matched_gt
            )

            precision = matched_weight / cand_weight if cand_weight > 0 else 0.0
            recall = matched_weight / gt_weight if gt_weight > 0 else 0.0
        else:
            # Simple counting
            precision = len(matched_cand) / len(cand_entities)
            recall = len(matched_gt) / len(gt_entities)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        # Build details
        unmatched_gt = gt_entities - matched_gt
        details = f"Matched {len(matched_gt)}/{len(gt_entities)} entities"
        if unmatched_gt:
            missing = [f"{text}({label})" for text, label in list(unmatched_gt)[:3]]
            details += f", Missing: {missing}"
            if len(unmatched_gt) > 3:
                details += f"... (+{len(unmatched_gt)-3} more)"

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched": len(matched_gt),
            "gt_count": len(gt_entities),
            "cand_count": len(cand_entities),
            "details": details
        }

    @timeit
    async def evaluate(self, ground_truth: str, candidate: str, **kwargs):
        """
        Evaluate entity matching between ground truth and candidate.

        Returns:
            dict with 'score' (F1) and 'comment' with details
        """
        try:
            if not SPACY_AVAILABLE:
                return {
                    "score": 0.0,
                    "comment": "spaCy not installed. Run: pip install spacy"
                }

            # Extract entities from both texts
            gt_entities = await asyncio.to_thread(
                self._extract_entities, ground_truth
            )
            cand_entities = await asyncio.to_thread(
                self._extract_entities, candidate
            )

            # Compute metrics
            metrics = self._compute_metrics(gt_entities, cand_entities)

            comment = (
                f"Entity F1: {metrics['f1']:.4f} "
                f"(P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}) - "
                f"{metrics['details']}"
            )

            return {"score": metrics['f1'], "comment": comment}

        except ImportError as e:
            return {"score": 0.0, "comment": str(e)}
        except Exception as e:
            return {
                "score": 0.0,
                "comment": f"Entity matching evaluation failed: {str(e)}"
            }
