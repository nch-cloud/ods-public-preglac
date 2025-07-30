"""Context rules for breastfeeding NLP pipeline."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from medspacy.context import ConTextRule


# class ConTextRule(BaseModel):
#     """Rule definition for medspaCy ConText."""

#     literal: str = Field(..., description="The literal string to match")
#     category: str = Field(..., description="The category of the modifier")
#     pattern: str = Field(None, description="Regular expression pattern for matching")
#     direction: str = Field("FORWARD", description="Direction of the context (FORWARD, BACKWARD, BIDIRECTIONAL)")
#     max_scope: int = Field(5, description="max_scope size for the context")
#     terminated_by: List[str] = Field(default_factory=list, description="Categories that terminate this modifier")
#     on_match: str = Field(None, description="Name of function to call when matched")


def get_context_rules() -> List[ConTextRule]:
    """
    Get context rules for the ConText algorithm.
    
    Returns:
        List of ConTextRule objects for the ConText algorithm
    """
    # Historical context rules
    historical_rules = [
        ConTextRule(
            literal="previously",
            category="HISTORICAL",
            pattern=r"(?i)\bpreviously\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["CURRENT"],
        ),
        ConTextRule(
            literal="prior",
            category="HISTORICAL",
            pattern=r"(?i)\bprior\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["CURRENT"],
        ),
        ConTextRule(
            literal="history of",
            category="HISTORICAL",
            pattern=r"(?i)\bhistory\s+of\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["CURRENT"],
        ),
        ConTextRule(
            literal="in the past",
            category="HISTORICAL",
            pattern=r"(?i)\bin\s+the\s+past\b",
            direction="BIDIRECTIONAL",
            max_scope=5,
            terminated_by=["CURRENT"],
        ),
        ConTextRule(
            literal="after",
            category="HISTORICAL",
            pattern=r"(?i)\bafter\b",
            direction="FORWARD",
        ),
        ConTextRule(
            literal="used to",
            category="HISTORICAL",
            pattern=r"(?i)\bused\s+to\b",
            direction="BIDIRECTIONAL",
            max_scope=5,
            terminated_by=["CURRENT"],
        ),
        ConTextRule(
            literal="no longer",
            category="HISTORICAL",
            pattern=r"(?i)\bno\s+longer\b",
            direction="BIDIRECTIONAL", # Can probably change to FORWARD?
            max_scope=5,
            terminated_by=["CURRENT"],
        ),
        ConTextRule(
            literal="home on",
            category="HISTORICAL",
            pattern=r"(?i)\bhome\s+on\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["CURRENT"],
        ),
    ]
    # TODO: Consider "Breast fed for 3 months, now on formula" --> should be historical BF.

    # Uncertainty rules
    # question mark?
    uncertainty_rules = [
        ConTextRule(
            literal="possible",
            category="UNCERTAIN",
            pattern=r"(?i)\bpossible\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="may",
            category="UNCERTAIN",
            pattern=r"(?i)\bmay\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="consider",
            category="UNCERTAIN",
            pattern=r"(?i)\bconsider\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="might",
            category="UNCERTAIN",
            pattern=r"(?i)\bmight\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="could",
            category="UNCERTAIN",
            pattern=r"(?i)\bcould\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="unsure if",
            category="UNCERTAIN",
            pattern=r"(?i)\bunsure\s+if\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
    ]

    # Intent rules (future tense)
    intent_rules = [
        ConTextRule(
            literal="plans to",
            category="INTENT",
            pattern=r"(?i)\bplans\s+to\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="will", # too broad? has to be within 5 tokens before entity... probably fine.
            category="INTENT",
            pattern=r"(?i)\bwill\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="is going to",
            category="INTENT",
            pattern=r"(?i)\bis\s+going\s+to\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="tried",
            category="INTENT",
            pattern=r"(?i)\btried\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="intends to",
            category="INTENT",
            pattern=r"(?i)\bintends\s+to\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="wants to",
            category="INTENT",
            pattern=r"(?i)\bwants\s+to\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="talked about",
            category="INTENT",
            pattern=r"(?i)\btalked\s+about\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="recommend",
            category="INTENT",
            pattern=r"(?i)\brecommend(ed)?(ations)?\b",
            direction="FORWARD",
            max_scope=10, # does this account for sentence boundaries?
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
    ]
    
    # Hypothetical context rules
    hypothetical_rules = [
        ConTextRule(
            literal="question",
            category="HYPOTHETICAL",
            pattern=r"(?i)\bquestions?\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="concerns",
            category="HYPOTHETICAL",
            pattern=r"(?i)\bconcerns\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="advice",
            category="HYPOTHETICAL",
            pattern=r"(?i)\badvice\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        ConTextRule(
            literal="wonder",
            category="HYPOTHETICAL",
            pattern=r"(?i)\bwonder(ing|s|ed)?\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
        # may want to try; may want to; might want to try; might want to
        ConTextRule(
            literal="may want to try",
            category="HYPOTHETICAL",
            pattern=r"(?i)\bmay|might\s+want\s+to(\s+try)?\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
    ]

    # difficulty?
    difficulty_rules = [
        ConTextRule(
            literal="difficult",
            category="DIFFICULTY",
            pattern=r"(?i)\bdifficult(y|ies)?\b",
            direction="FORWARD",
            max_scope=5,
            terminated_by=["DEFINITE_EXISTENCE"],
        ),
    ]

    # Combine all rules
    all_rules = (
        historical_rules
        + uncertainty_rules
        + intent_rules
        + hypothetical_rules
        + difficulty_rules
    )

    return all_rules 
