"""Target rules for breastfeeding NLP pipeline."""

import re
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
from medspacy.ner import TargetRule
from spacy.tokens import Span


def create_keyword_checker(keywords: Dict[str, list]=None,
                           amount_keywords: List[str]=None,
                           frequency_keywords: List[str]=None,
                           nursing_keywords: List[str]=None):
    """
    Creates a function that checks if any of the specified keywords appear near (+/- 3 tokens) a matched entity
    
    Args:
        keywords: List of strings to check for in the surrounding text
        amount_keywords: List of strings to check for in the surrounding text
        frequency_keywords: List of strings to check for in the surrounding text
        nursing_keywords: List of strings to check for in the surrounding text
    Returns:
        A function to use as an on_match callback
    """
    if keywords is None and amount_keywords is None and frequency_keywords is None:
        return None

    if keywords is None:
        keywords = {}
    
    if amount_keywords is None:
        # default amount keywords
        amount_keywords = ['ml', 'oz', 'ounce', 'ounces', 'cup', 'cups', 'bottle', 'bottles']
    
    if frequency_keywords is None:
        # default frequency keywords
        frequency_keywords = ['hour', 'hours']
    
    if nursing_keywords is None:
        # default nursing keywords that mean the nursing mention is NOT related to breast feeding.
        # Set to surrounding 2 tokens.
        nursing_keywords = ['home', 'facility', 'staff', 'assistant', 'assistants', 'team', "student", "students"]
        nursing_keywords.extend(['agency', 'agencies', 'intervention', 'interventions', 'judgment', 'note', 'notes', 'notereviewed'])
        nursing_keywords.extend(['the', 'reports', 'report', 'reports to', 'line', 'care', 'cares', 'routine', 'called', 'presence', 'clinical'])

    # TODO: add flag for donor milk, milk bank, etc.
    
    def check_for_keywords(matcher, doc, i, matches):
        match_id, start, end = matches[i]  # Get the matched entity's position
        ent = doc[start:end]  # Get the entity span
        
        # Get surrounding tokens, but don't cross sentence boundaries
        sent = doc[start].sent  # Get the sentence containing the entity
        surrounding_start = max(start-5, sent.start)  # Don't go before sentence start
        surrounding_end = min(end+5, sent.end)  # Don't go past sentence end
        # surrounding_start = sent.start
        # surrounding_end = sent.end
        surrounding_text = doc[surrounding_start:surrounding_end].text.lower()  # Get surrounding text
        
        # Check if any of the keywords are in the surrounding text
        for extension_name, keyword_list in keywords.items():
            # Check if any keyword in the list is found in surrounding text
            found = any(keyword.lower() in surrounding_text for keyword in keyword_list)
            # Set the extension for THIS SPECIFIC entity instance
            try:
                Span.set_extension(extension_name.upper(), default=False, force=False)
            except ValueError: # The extension already exists
                pass
            ent._.set(extension_name.upper(), found)
        
        # Check if there's any information about the amount (ml, oz, cup, etc.)
        # amount_pattern = r'\d+\s*(ml|oz|ounces?|cup|tsp|tbsp|fl oz|gal|l|pt|qt)'
        amount_pattern = r'(\d+(?:-\d+)?\s*(?:' + '|'.join(amount_keywords) + '))'

        # Expand window size to include the entire sentence
        # surrounding_text = doc[sent.start:sent.end].text.lower() # Full sentence is too long when it's a bullet point list with ¿
        surrounding_start = max(start-12, sent.start)  # Don't go before sentence start
        surrounding_end = min(end+10, sent.end)  # Don't go past sentence end
        surrounding_text = doc[surrounding_start:surrounding_end].text.lower()
        amount_match = re.findall(amount_pattern, surrounding_text)
        if amount_match:
            # set the amount extension
            try:
                Span.set_extension("AMOUNT", default=None, force=False)
            except ValueError: # The extension already exists
                pass
            ent._.set("AMOUNT", '; '.join(amount_match))
        
        # Check if there's any information about the frequency of feeding
        frequency_pattern = r'(\d+(?:-\d+)?\s*(?:' + '|'.join(frequency_keywords) + '))'
        surrounding_start = max(surrounding_start-5, sent.start)
        surrounding_end = min(surrounding_end+5, sent.end)
        surrounding_text = doc[surrounding_start:surrounding_end].text.lower()
        frequency_match = re.findall(frequency_pattern, surrounding_text)
        if frequency_match:
            # set the frequency extension
            try:
                Span.set_extension("FREQUENCY", default=None, force=False)
            except ValueError: # The extension already exists
                pass
            ent._.set("FREQUENCY", '; '.join(frequency_match))
        
        # Check if there's any information about the nursing term
        if "nurs" in ent.text.lower():
            # Set nursing to True if the entity is "nursing"
            try:
                Span.set_extension("NURSING", default=None, force=False)
            except ValueError: # The extension already exists
                pass
            ent._.set("NURSING", True)

            nursing_pattern = r'\b(?:' + '|'.join(nursing_keywords) + r')\b'
            surrounding_start = max(start-2, sent.start)
            surrounding_end = min(end+2, sent.end)
            surrounding_text = doc[surrounding_start:surrounding_end].text.lower()
            nursing_match = re.findall(nursing_pattern, surrounding_text)

            # Nursing values:
            ## True: a valid nursing mention that indicates breastfeeding
            ## False: a nursing term that does not indicate breastfeeding
            ## NaN: an entity that is not a nursing term
            if nursing_match:
                # set the nursing extension to False
                ent._.set("NURSING", False)

        # If the entity is 'breast'
        # TODO: Test to make sure that this won't negatively impact terms like "breast feeding" or "breast milk"
        if ent.text.lower() == 'breast':
            try:
                Span.set_extension("BREAST", default=None, force=False)
            except ValueError: # The extension already exists
                pass
            ent._.set("BREAST", False)

            # set surrounding text boundaries
            surrounding_start = max(start-3, sent.start)  # Don't go before sentence start
            surrounding_end = min(end+3, sent.end)  # Don't go past sentence end
            surrounding_text = doc[surrounding_start:surrounding_end].text.lower()

            # check to see if 'left', 'right', or 'both' is in the surrounding text
            breast_laterality_pattern = r'\b(left|right|both)\b'
            breast_laterality_match = re.findall(breast_laterality_pattern, surrounding_text)

            # check to see if 'fed at', or 'at' is in the surrounding text
            breast_feeding_pattern = r'\b(fed\s+at|at)\b'
            breast_feeding_match = re.findall(breast_feeding_pattern, surrounding_text)

            if breast_laterality_match or breast_feeding_match:
                ent._.set("BREAST", True)

    return check_for_keywords



def get_target_rules(matching_function: Callable=None) -> List[TargetRule]:
    """
    Get target rules for breastfeeding-related entities.
    
    Returns:
        List of TargetRule objects defining breastfeeding-related entities
    """
    breast_feeding_rules = [
        #  - Breast Feedings
        #  - Breast Fed
        #  - Breastfed
        #  - Breastfeeding
        #  - Breast Feeding, Exclusive
        #  - Exclusive Breast Feeding
        #  - Breastfeeding, Exclusive
        #  - Exclusive Breastfeeding
        TargetRule(
            literal="breast feedings",
            category="BREAST_FEEDING",
            pattern=r"(?i)breast[-\s]?fee?d(ing)?s?",
            on_match=matching_function, # exclusive
        ),
        #  - Breastmilk
        #  - Breast Milk
        TargetRule(
            literal="breastmilk",
            category="BREAST_FEEDING",
            pattern=r"(?i)breast\s?milk",
            on_match=matching_function,
        ),
        #  - EBM
        #  - EBF
        #  - BF
        #  - BM (consider removing). 
            # Decision: remove. Only 1 valid bm mention and that note has other valid breast feeding mentions. Rest are bowel related.
        TargetRule(
            literal="ebm",
            category="BREAST_FEEDING",
            pattern=r"(?i)\b(eb[fm]|bf)\b",
            on_match=matching_function,
        ),
        #  - Chestfeeding
        #  - Chestfeedings
        TargetRule(
            literal="chest feedings",
            category="BREAST_FEEDING",
            pattern=r"(?i)chest[-\s]?fee?d(ing)?s?",
            on_match=matching_function, # exclusive
        ),
        #  - Wet Nursing
        TargetRule(
            literal="wet nursing",
            category="BREAST_FEEDING",
            pattern=r"(?i)\b(is\s+)?(wet\s)?nurs(ing)?(ed)?\b",
            on_match=matching_function,
        ),
        #  - nursed
        TargetRule(
            literal="nursed",
            category="BREAST_FEEDING",
            pattern=r"(?i)\bnursed\b",
            on_match=matching_function,
        ),
        # - able to nurse
        TargetRule(
            literal="able to nurse",
            category="BREAST_FEEDING",
            pattern=r"(?i)\bable to nurse\b",
            on_match=matching_function,
        ),
        
        #  - Milk Sharing
        #  - Sharing, Milk
        TargetRule(
            literal="milk sharing",
            category="BREAST_FEEDING",
            pattern=r"(?i)(sharing\s+milk|milk\s+sharing)",
            on_match=matching_function,
        ),
        # - Expressed Breastmilk
        TargetRule(
            literal="expressed breastmilk",
            category="BREAST_FEEDING",
            pattern=r"(?i)expressed\s?breast\s?milk",
            on_match=matching_function,
        ),
        # - Breast Milk Expressions
        # - Breastmilk Expression
        # - Breastmilk Expressions
        TargetRule(
            literal="breast milk expression",
            category="BREAST_FEEDING",
            pattern=r"(?i)breast\s?milk\s+expressions?",
            on_match=matching_function,
        ),
        # Just breast milk
        TargetRule(
            literal="breast milk",
            category="BREAST_FEEDING",
            pattern=r"(?i)breast\s?milk",
            on_match=matching_function,
        ),
        # fortified breast milk
        TargetRule(
            literal="fortified breast milk",
            category="BREAST_FEEDING",
            pattern=r"(?i)fortified\sbreast\s?milk",
            on_match=matching_function,
        ),
        # - Expression, Breast Milk
        # - Expressions, Breast Milk
        # - Expression, Breastmilk
        # - Expressions, Breastmilk
        TargetRule(
            literal="expression, breast milk",
            category="BREAST_FEEDING",
            pattern=r"(?i)expressions?\s+breast\s?milk",
            on_match=matching_function,
        ),
        # - Milk Expression, Breast
        # - Milk Expressions, Breast
        TargetRule(
            literal="milk expression, breast",
            category="BREAST_FEEDING",
            pattern=r"(?i)milk\s+expressions?\s+breast",
            on_match=matching_function,
        ),
        # - breast and bottle
        # - breast and formula fed
        # - breast and X
        TargetRule(
            literal="breast and bottle",
            category="BREAST_FEEDING",
            # pattern=r"(?i)breast\s+(and|&)\s+(bottle|formula)\s*(feed(ing)?|fed)?",
            pattern=r"(?i)\bbreast\s+(and|&)\s+\w+\s*(fed|feeding|feeds)?\b",
            on_match=matching_function,
        ),
        # - breast/bottle
        # - breast/bottle feeds
        # - breast/formula
        # - breast/formula-fed
        # - breast/formula feeding
        TargetRule(
            literal="breast/bottle",
            category="BREAST_FEEDING",
            pattern=r"(?i)\bbreast/\w+(?:[-\s]?(fed|feeding|feeds))?\b",
            on_match=matching_function
        ),
        # - difficulty latching on
        # - problems included difficulty latching on
        TargetRule(
            literal="latching on",
            category="BREAST_FEEDING",
            pattern=r"(?i)\b(difficult(y|ies)|problem(s)?)\s+(included\s+)?latch(ing)?\s+on\b",
            on_match=matching_function,
        ),
        # - latching on
        TargetRule(
            literal="latching on",
            category="BREAST_FEEDING",
            pattern=r"(?i)\blatch(ing)?\s+on\b",
            on_match=matching_function,
        ),
        # TODO: Consider "to (the)? breast" --> "put him to breast", "back to the breast"
        # TODO: Consider "Fed for 30 minutes on both sides"
    ]

    formula_feeding_rules = [
        # - Formula, Infant
        # - Formulas, Infant
        # - Infant Formulas
        # - Baby Formula
        # - Baby Formulas
        # - Formula, Baby
        # - Formulas, Baby
        TargetRule(
            literal="formula",
            category="FORMULA_FEEDING",
            pattern=r"(?i)(baby\s)?(infant\s)?formulas?",
            on_match=matching_function,
        ),
        # brand names
        # - Similac
        # - Enfamil
        # - Gerber
        # - Earth's Best
        # - sim advance
        # - NeoSure
        # - EnfaCare
        # - Isomil
        # - Alimentum
        TargetRule(
            literal="similac",
            category="FORMULA_FEEDING",
            pattern=r"(?i)\bsim(ilac)?\b",
            on_match=matching_function,
        ),
        TargetRule(
            literal="sim advance",
            category="FORMULA_FEEDING",
            pattern=r"(?i)sim\s+advance",
            on_match=matching_function,
        ),
        TargetRule(
            literal="enfamil",
            category="FORMULA_FEEDING",
            pattern=r"(?i)enfamil",
            on_match=matching_function,
        ),
        TargetRule(
            literal="gerber",
            category="FORMULA_FEEDING",
            pattern=r"(?i)gerber",
            on_match=matching_function,
        ),
        TargetRule(
            literal="earth's best",
            category="FORMULA_FEEDING",
            pattern=r"(?i)earth's\s+best",
            on_match=matching_function,
        ),
        TargetRule(
            literal="neosure",
            category="FORMULA_FEEDING",
            pattern=r"(?i)neosure",
            on_match=matching_function,
        ),
        TargetRule(
            literal="enfacare",
            category="FORMULA_FEEDING",
            pattern=r"(?i)enfacare",
            on_match=matching_function,
        ),
        TargetRule(
            literal='isomil',
            category="FORMULA_FEEDING",
            pattern=r"(?i)isomil",
            on_match=matching_function,
        ),
        TargetRule(
            literal='alimentum',
            category="FORMULA_FEEDING",
            pattern=r"(?i)all?imentum", # allowing for "alimentum" and "allimentum"
            on_match=matching_function,
        ),
        # soy?
        # pediasure? Not exclusively breastfed if pediasure, right?
        # pediatlyte? pedialite? -- used for hydration. Probably not important to include... unless it is?
    ]

    ambiguous_rules = [
        # - Bottlefeeding
        # - Bottlefed
        TargetRule(
            literal="bottlefeeding",
            category="AMBIGUOUS",
            pattern=r"(?i)bottle[-\s]?fee?d(ing)?s?",
            on_match=matching_function,
        ),
        # - Breast Milk Collection
        # - Breast Milk Collections
        # - Breastmilk Collection
        # - Breastmilk Collections
        TargetRule(
            literal="breast milk collection",
            category="AMBIGUOUS",
            pattern=r"(?i)breast\s?milk\s+collections?",
            on_match=matching_function,
        ),
        # - Milk Collection, Breast
        # - Milk Collections, Breast
        TargetRule(
            literal="milk collection, breast",
            category="AMBIGUOUS",
            pattern=r"(?i)milk\s+collections?\s+breast",
            on_match=matching_function,
        ),
        # - Milk -- too broad. Removing for now.
        # TargetRule(
        #     literal="milk",
        #     category="AMBIGUOUS",
        #     pattern=r"(?i)milk",
        #     on_match=matching_function,
        # ),
        # - Collection, Breast Milk
        # - Collections, Breast Milk
        # - Collection, Breastmilk
        # - Collections, Breastmilk
        TargetRule(
            literal="collection, breast milk",
            category="AMBIGUOUS",
            pattern=r"(?i)collections?\s+breast\s?milk",
            on_match=matching_function,
        ),
        # - Breast Pumping
        # - Breast Pumpings
        TargetRule(
            literal="breast pumping",
            category="AMBIGUOUS",
            pattern=r"(?i)breast\s?pumpings?",
            on_match=matching_function,
        ),
        # - Pumping, Breast
        # - Pumpings, Breast
        TargetRule(
            literal="pumping, breast",
            category="AMBIGUOUS",
            pattern=r"(?i)pump(s|ing)(\s+breast)?",
            on_match=matching_function,
        ),
        # - Pumping -- Just because pumping doesn't mean consumption.
        TargetRule(
            literal="pumping",
            category="AMBIGUOUS",
            pattern=r"(?i)pumping",
            on_match=matching_function,
        ),
        # human milk fortifier?
        TargetRule(
            literal="human milk fortifier",
            category="BREAST_FEEDING", # Probably mixed. Should definitely be considered as breast milk to some degree.
            # pattern=r"(?i)human\s+milk\s(?:fortifier)",
            on_match=matching_function,
        ),
        TargetRule(
            literal="human milk",
            category="AMBIGUOUS", # not enough information to be specific
            pattern=r"(?i)human\s+milk", # case insensitive.
            on_match=matching_function,
        ),
        TargetRule(
            literal="fortifier",
            category="BREAST_FEEDING",
            on_match=matching_function,
        ),
        TargetRule(
            literal="hmf", # human milk fortifier
            category="BREAST_FEEDING",
            on_match=matching_function,
        ),
        TargetRule(
            literal="elecare",
            category="AMBIGUOUS",
            pattern=r"(?i)el(ea)care",
            on_match=matching_function,
        ),
    ]
    
    # Feeding related but not definitive terms
    feeding_related_rules = [
        TargetRule(
            literal="tender nipples",
            category="FEEDING_RELATED",
            pattern=r"(?i)tender\s+nipples?",
            on_match=matching_function,
        ),
        TargetRule(
            literal="cracked nipples",
            category="FEEDING_RELATED",
            pattern=r"(?i)crack(ed)?(ing)?\s+nipples?",
            on_match=matching_function,
        ),
        TargetRule(
            literal="lanolin",
            category="FEEDING_RELATED",
            # pattern=r"(?i)\blanolin\b",
            on_match=matching_function,
        ),
        TargetRule(
            literal="hand expression",
            category="FEEDING_RELATED",
            # pattern=r"(?i)hand\s+expressions?",
            on_match=matching_function,
        ),
        TargetRule(
            literal="football position",
            category="FEEDING_RELATED",
            # pattern=r"(?i)football\s+positions?",
            on_match=matching_function,
        ),
        TargetRule(
            literal="football hold",
            category="FEEDING_RELATED",
            # pattern=r"(?i)football\s+hold",
            on_match=matching_function,
        ),
        TargetRule(
            literal="feeding difficulties",
            category="FEEDING_RELATED",
            pattern=r"(?i)feeding\s+difficult(y|ies)",
            on_match=matching_function,
        ),
        TargetRule(
            literal="poor weight gain",
            category="FEEDING_RELATED",
            # pattern=r"(?i)poor\s+weight\s+gain",
            on_match=matching_function,
        ),
        # ng tube, nasogastric, gavage, po/gavage
        TargetRule(
            literal="ng tube",
            category="FEEDING_RELATED",
            pattern=r"(?i)\bn?g\s+tube\b", # ensure both are distinct words to avoid capturing "suggesting tuberculosis"
            on_match=matching_function,
        ),
        TargetRule(
            literal="feeding tube",
            category="FEEDING_RELATED",
            on_match=matching_function,
        ),
        TargetRule(
            literal="nasogastric",
            category="FEEDING_RELATED",
            # pattern=r"(?i)nasogastric",
            on_match=matching_function,
        ),
        TargetRule(
            literal="gavage",
            category="FEEDING_RELATED",
            # pattern=r"(?i)gavage",
            on_match=matching_function,
        ),
        # Cached abbreviations expand po, so it doesn't get captured. 
        TargetRule(
            literal="po/gavage",
            category="FEEDING_RELATED",
            pattern=r"(?i)\bp\.?o\.?(/gavage)?\b",
            on_match=matching_function,
        ),
        TargetRule(
            literal="per os",
            category="FEEDING_RELATED",
            on_match=matching_function,
        ),
        TargetRule(
            literal="latch difficulties",
            category="FEEDING_RELATED",
            pattern=r"(?i)latch(ing)?\s+difficult(y|ies)",
            on_match=matching_function,
        ),
        
        # gt
        # gt as a bolus -- a bolus is 1 "meal" of formula/milk. Method of delivery using syringe and gravity, not a continuous pump.
    ]

    # Combine all rules
    all_rules = (
        breast_feeding_rules
        + formula_feeding_rules
        + ambiguous_rules
        + feeding_related_rules
    )

    return all_rules


def add_levenstein_variations(rules: List[TargetRule], max_distance: int = 1) -> List[TargetRule]:
    """
    Add variations of target rules based on Levenstein distance to handle misspellings.
    
    Args:
        rules: List of target rules
        max_distance: Maximum Levenstein distance for variations
        
    Returns:
        Expanded list of target rules with variations
    """
    # This would be implemented to generate spelling variations
    # For now, we'll return the original rules
    return rules 