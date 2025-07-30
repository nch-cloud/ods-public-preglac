"""Target rules for breastfeeding NLP pipeline."""

import re
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
from medspacy.ner import TargetRule
from spacy.tokens import Span



def get_target_rules() -> List[TargetRule]:
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
        ),
        #  - Breastmilk
        #  - Breast Milk
        TargetRule(
            literal="breastmilk",
            category="BREAST_FEEDING",
            pattern=r"(?i)breast\s?milk",
        ),
        #  - EBM
        #  - EBF
        #  - BF
        #  - BM (consider removing)
        TargetRule(
            literal="ebm",
            category="BREAST_FEEDING",
            pattern=r"(?i)\be?b(m|f)\b",
        ),
        #  - Chestfeeding
        #  - Chestfeedings
        TargetRule(
            literal="chest feedings",
            category="BREAST_FEEDING",
            pattern=r"(?i)chest[-\s]?fee?d(ing)?s?",
        ),
        #  - Wet Nursing
        # TODO: nursing assistant, nursing staff, nursing home, trouble with latching/nursing
        TargetRule(
            literal="wet nursing",
            category="BREAST_FEEDING",
            pattern=r"(?i)\b(is\s+)?(wet\s)?nurs(ing)?(ed)?\b",
        ),
        #  - Milk Sharing
        #  - Sharing, Milk
        TargetRule(
            literal="milk sharing",
            category="BREAST_FEEDING",
            pattern=r"(?i)(sharing\s+milk|milk\s+sharing)",
        ),
        # - Expressed Breastmilk
        TargetRule(
            literal="expressed breastmilk",
            category="BREAST_FEEDING",
            pattern=r"(?i)expressed\s?breast\s?milk",
        ),
        # - Breast Milk Expressions
        # - Breastmilk Expression
        # - Breastmilk Expressions
        TargetRule(
            literal="breast milk expression",
            category="BREAST_FEEDING",
            pattern=r"(?i)breast\s?milk\s+expressions?",
        ),
        # - Expression, Breast Milk
        # - Expressions, Breast Milk
        # - Expression, Breastmilk
        # - Expressions, Breastmilk
        TargetRule(
            literal="expression, breast milk",
            category="BREAST_FEEDING",
            pattern=r"(?i)expressions?\s+breast\s?milk",
        ),
        # - Milk Expression, Breast
        # - Milk Expressions, Breast
        TargetRule(
            literal="milk expression, breast",
            category="BREAST_FEEDING",
            pattern=r"(?i)milk\s+expressions?\s+breast",
        ),
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
        ),
        # brand names
        # - Similac
        # - Enfamil
        # - Gerber
        # - Earth's Best
        # - sim advance
        # - NeoSure
        # - EnfaCare
        TargetRule(
            literal="similac",
            category="FORMULA_FEEDING",
            pattern=r"(?i)\bsim(ilac)?\b",
        ),
        TargetRule(
            literal="sim advance",
            category="FORMULA_FEEDING",
            pattern=r"(?i)sim\s+advance",
        ),
        TargetRule(
            literal="enfamil",
            category="FORMULA_FEEDING",
            pattern=r"(?i)enfamil",
        ),
        TargetRule(
            literal="gerber",
            category="FORMULA_FEEDING",
            pattern=r"(?i)gerber",
        ),
        TargetRule(
            literal="earth's best",
            category="FORMULA_FEEDING",
            pattern=r"(?i)earth's\s+best",
        ),
        TargetRule(
            literal="neosure",
            category="FORMULA_FEEDING",
            pattern=r"(?i)neosure",
        ),
        TargetRule(
            literal="enfacare",
            category="FORMULA_FEEDING",
            pattern=r"(?i)enfacare",
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
        ),
        # - Breast Milk Collection
        # - Breast Milk Collections
        # - Breastmilk Collection
        # - Breastmilk Collections
        TargetRule(
            literal="breast milk collection",
            category="AMBIGUOUS",
            pattern=r"(?i)breast\s?milk\s+collections?",
        ),
        # - Milk Collection, Breast
        # - Milk Collections, Breast
        TargetRule(
            literal="milk collection, breast",
            category="AMBIGUOUS",
            pattern=r"(?i)milk\s+collections?\s+breast",
        ),
        # - Milk -- too broad. Removing for now.
        # TargetRule(
        #     literal="milk",
        #     category="AMBIGUOUS",
        #     pattern=r"(?i)milk",
        # ),
        # - Collection, Breast Milk
        # - Collections, Breast Milk
        # - Collection, Breastmilk
        # - Collections, Breastmilk
        TargetRule(
            literal="collection, breast milk",
            category="AMBIGUOUS",
            pattern=r"(?i)collections?\s+breast\s?milk",
        ),
        # - Breast Pumping
        # - Breast Pumpings
        TargetRule(
            literal="breast pumping",
            category="AMBIGUOUS",
            pattern=r"(?i)breast\s?pumpings?",
        ),
        # - Pumping, Breast
        # - Pumpings, Breast
        TargetRule(
            literal="pumping, breast",
            category="AMBIGUOUS",
            pattern=r"(?i)pumpings?\s+breast",
        ),
        # - Pumping -- Just because pumping doesn't mean consumption.
        TargetRule(
            literal="pumping",
            category="AMBIGUOUS",
            pattern=r"(?i)pumping",
        ),
        # human milk fortifier?
        TargetRule(
            literal="human milk fortifier",
            category="AMBIGUOUS",
            # pattern=r"(?i)human\s+milk\s(?:fortifier)",
        ),
        TargetRule(
            literal="human milk",
            category="AMBIGUOUS", # not enough information to be specific
        ),
        TargetRule(
            literal="fortifier",
            category="AMBIGUOUS",
        ),
        TargetRule(
            literal="hmf", # human milk fortifier
            category="AMBIGUOUS",
        ),
        TargetRule(
            literal="elecare",
            category="AMBIGUOUS",
        ),
    ]
    
    # Feeding related but not definitive terms
    feeding_related_rules = [
        TargetRule(
            literal="tender nipples",
            category="FEEDING_RELATED",
            pattern=r"(?i)tender\s+nipples?",
        ),
        TargetRule(
            literal="cracked nipples",
            category="FEEDING_RELATED",
            pattern=r"(?i)crack(ed)?(ing)?\s+nipples?",
        ),
        TargetRule(
            literal="lanolin",
            category="FEEDING_RELATED",
            pattern=r"(?i)\blanolin\b",
        ),
        TargetRule(
            literal="hand expression",
            category="FEEDING_RELATED",
            pattern=r"(?i)hand\s+expressions?",
        ),
        TargetRule(
            literal="football position",
            category="FEEDING_RELATED",
            pattern=r"(?i)football\s+positions?",
        ),
        TargetRule(
            literal="football hold",
            category="FEEDING_RELATED",
            pattern=r"(?i)football\s+hold",
        ),
        TargetRule(
            literal="feeding difficulties",
            category="FEEDING_RELATED",
            pattern=r"(?i)feeding\s+difficult(y|ies)",
        ),
        TargetRule(
            literal="poor weight gain",
            category="FEEDING_RELATED",
            pattern=r"(?i)poor\s+weight\s+gain",
        ),
        # ng tube, nasogastric, gavage, po/gavage
        TargetRule(
            literal="ng tube",
            category="FEEDING_RELATED",
            pattern=r"(?i)ng\s+tube",
        ),
        TargetRule(
            literal="nasogastric",
            category="FEEDING_RELATED",
            pattern=r"(?i)nasogastric",
        ),
        TargetRule(
            literal="gavage",
            category="FEEDING_RELATED",
            pattern=r"(?i)gavage",
        ),
        # Cached abbreviations expand po, so it doesn't get captured. 
        TargetRule(
            literal="po/gavage",
            category="FEEDING_RELATED",
            pattern=r"(?i)\bp\.?o\.?(/gavage)?\b",
        ),
        TargetRule(
            literal="per os",
            category="FEEDING_RELATED",
        ),
        TargetRule(
            literal="latch difficulties",
            category="FEEDING_RELATED",
            pattern=r"(?i)latch(ing)?\s+difficult(y|ies)",
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
