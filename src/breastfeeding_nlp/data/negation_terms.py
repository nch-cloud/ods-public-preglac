"""Negation termsets for the breastfeeding NLP pipeline."""

from typing import Dict, List, Tuple


def get_negation_termsets() -> Tuple[
    List[str], List[str], List[str], List[str], List[str]
]:
    """
    Get custom negation termsets for negspaCy.
    
    Returns:
        Tuple of lists containing:
        1. Pseudo-negation terms
        2. Preceding negation terms
        3. Following negation terms
        4. Termination terms
        5. Conjunction terms
    """
    # Pseudo-negation terms: terms that look like negations but don't negate
    pseudo_negations = [
        # "no evidence of",
        # "no increase",
        # "no interval change",
        # "no obvious",
        # "no significant change",
        # "no suspicious",
        # "not cause",
        # "not certain if",
        # "not certain whether",
        # "not necessarily",
        # "not only",
        # "without any further",
        # "without apparent",
        # "without change",
        # "without difficulty",
        # "without obvious",
    ]

    # Preceding negation terms: negation terms that precede what they modify
    preceding_negations = [
        "absence of",
        "absent",
        "cannot see",
        "checked for",
        "declined",
        "declined to",
        # "declined to breastfeed",
        "denied",
        "denies",
        "denying",
        "didn't",
        "does not exhibit",
        "doesn't exhibit",
        "excluded",
        "fails to reveal",
        "free of",
        "negative for",
        # "never breast fed",
        # "never breastfed",
        # "never developed",
        # "never had",
        "never",
        "no",
        "no abnormal",
        "no cause of",
        "no complaints of",
        "no evidence",
        "no evidence to suggest",
        "no findings of",
        "no findings to indicate",
        # "no mammographic evidence of",
        "no new",
        "no new evidence",
        "no other evidence",
        # "no radiographic evidence of",
        "no sign of",
        "no significant",
        "no signs of",
        "no suggestion of",
        # "no suspicious",
        "not",
        "not appear",
        "not appreciate",
        "not associated with",
        # "not complain of",
        "not demonstrate",
        "not exhibit",
        "not feel",
        "not had",
        "not have",
        "not know of",
        "not known to have",
        "not reveal",
        "not see",
        "not to be",
        "patient was not",
        "rather than",
        "resolved",
        "ruled out",
        "rules out",
        "test for",
        "to exclude",
        "unable to"
        "unremarkable for",
        "with no",
        "without",
        "without any",
        "without evidence",
        "without indication of",
        "without sign of",
        "stopped",
        "discontinued",
        "avoids",
        "avoiding",
        "unable to",
    ]

    # Following negation terms: negation terms that follow what they modify
    following_negations = [
        "declined",
        "denied",
        "denies",
        "didn't",
        "free",
        "negative",
        "not seen",
        "resolved",
        "ruled out",
        # "not"
    ]

    # Termination terms: terms that stop the scope of a negation
    termination_terms = [
        # All of these terms are being treated as negation terms instead of termination terms. Consider removing them.
        # Affects terms before and after the negation term.
        # ^^ This is because of the custom check that happens after negex.
        # "although",
        # "after",
        # "apart from",
        # "as a cause for",
        # "as the cause of",
        # "as the etiology for",
        # "as the origin for",
        # "as the reason for",
        # "as the source for",
        # "aside from",
        # "but",
        # "cause for",
        # "cause of",
        # "causes for",
        # "causes of",
        # "etiology for",
        # "etiology of",
        "except",
        # "however",
        # "nevertheless",
        # "origin for",
        # "origin of",
        # "origins for",
        # "origins of",
        # "other than",
        # "primary factor",
        # "reason for",
        # "reason of",
        # "reasons for",
        # "reasons of",
        # "require",
        # "required",
        # "source for",
        # "source of",
        # "sources for",
        # "sources of",
        # "still",
        # "though",
        # "unless",
        # "without",
        # "yet",
        # "¿", # this is being treated as a negation term instead of a termination. Need to check the others.
        # r"\s{2,}",  # Two or more consecutive spaces,
        # "instead",
        # "alternatively",

    ]

    # Conjunctions: terms that join phrases together
    # conjunctions = [
    #     "but",
    #     "however",
    #     "nevertheless",
    #     "yet",
    #     "though",
    #     "although",
    #     "still",
    #     "aside from",
    #     "except",
    #     "apart from",
    #     "other than",
    #     "as the",
    #     "as a",
    #     "as",
    #     "whereas",
    #     "while",
    #     "unless",
    #     "without",
    # ]

    return {
        "pseudo_negations": pseudo_negations,
        "preceding_negations": preceding_negations,
        "following_negations": following_negations,
        "termination": termination_terms,
        # "conjunctions": conjunctions,
    }


def get_chunk_prefixes() -> List[str]:
    """
    Get common chunk prefixes relevant to biomedical text.
    
    Returns:
        List of chunk prefixes
    """
    # Common biomedical chunk prefixes
    # chunk_prefixes = [
    #     "breast feeding",
    #     "breastfeeding",
    #     "formula feeding",
    #     "infant feeding",
    #     "bottle feeding",
    #     "breast milk",
    #     "breastmilk",
    #     "pumping",
    #     "nursing",
    #     "maternal feeding",
    #     "mixed feeding",
    #     "nutrition",
    #     "lactation",
    #     "latch",
    #     "nipple",
    #     "baby formula",
    #     "infant formula",
    # ]

    chunk_prefixes = ["no evidence of", "no signs of", "no suggestion of"]
    
    return chunk_prefixes 