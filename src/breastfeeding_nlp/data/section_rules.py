"""Section rules for breastfeeding NLP pipeline."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from medspacy.section_detection import SectionRule

# class SectionRule(BaseModel):
#     """Rule definition for medspaCy Sectionizer."""

#     literal: str = Field(..., description="The literal string to match")
#     category: str = Field(..., description="The category of the section")
#     pattern: str = Field(None, description="Regular expression pattern for matching")
#     parents: List[str] = Field(default_factory=list, description="Parent sections")
#     on_match: str = Field(None, description="Name of function to call when matched")


def get_section_rules() -> List[SectionRule]:
    """
    Get section rules for the Sectionizer.
    
    Returns:
        List of SectionRule objects for the Sectionizer
    """
    # Common section headers in clinical notes
    section_rules = [
        # Patient information sections
        SectionRule(
            literal="patient information",
            category="PATIENT_INFO",
            pattern=r"(?i)(?:\b|^)(?:patient\s+information|patient\s+info|demographics)(?:\s+|:|$)",
        ),
        
        # History sections
        SectionRule(
            literal="history of present illness",
            category="HISTORY",
            pattern=r"(?i)(?:\b|^)(?:history\s+of\s+present\s+illness|hpi)(?:\s+|:|$)",
        ),
        SectionRule(
            literal="past medical history",
            category="HISTORY",
            pattern=r"(?i)(?:\b|^)(?:past\s+medical\s+history|pmh)(?:\s+|:|$)",
        ),
        SectionRule(
            literal="family history",
            category="HISTORY",
            pattern=r"(?i)(?:\b|^)(?:family\s+history|fh)(?:\s+|:|$)",
        ),
        
        # Feeding related sections
        SectionRule(
            literal="infant feeding",
            category="FEEDING",
            pattern=r"(?i)(?:\b|^)(?:infant\s+feeding|feeding|nutrition)(?:\s+|:|$)",
        ),
        SectionRule(
            literal="breastfeeding",
            category="FEEDING",
            pattern=r"(?i)(?:\b|^)(?:breast\s*feeding|breastfeeding)(?:\s+|:|$)",
        ),
        SectionRule(
            literal="lactation",
            category="FEEDING",
            pattern=r"(?i)(?:\b|^)(?:lactation|lactation\s+consult)(?:\s+|:|$)",
        ),
        
        # Examination sections
        SectionRule(
            literal="physical examination",
            category="EXAMINATION",
            pattern=r"(?i)(?:\b|^)(?:physical\s+examination|physical\s+exam|pe)(?:\s+|:|$)",
        ),
        SectionRule(
            literal="review of systems",
            category="EXAMINATION",
            pattern=r"(?i)(?:\b|^)(?:review\s+of\s+systems|ros)(?:\s+|:|$)",
        ),
        
        # Assessment and plan sections
        SectionRule(
            literal="assessment",
            category="ASSESSMENT",
            pattern=r"(?i)(?:\b|^)(?:assessment|impression)(?:\s+|:|$)",
        ),
        SectionRule(
            literal="plan",
            category="PLAN",
            pattern=r"(?i)(?:\b|^)(?:plan|recommendation)(?:\s+|:|$)",
        ),
        SectionRule(
            literal="assessment and plan",
            category="ASSESSMENT_PLAN",
            pattern=r"(?i)(?:\b|^)(?:assessment\s+and\s+plan|a\/p|impression\s+and\s+plan)(?:\s+|:|$)",
        ),
        
        # Specific newborn sections
        SectionRule(
            literal="newborn assessment",
            category="NEWBORN",
            pattern=r"(?i)(?:\b|^)(?:newborn\s+assessment|newborn\s+exam)(?:\s+|:|$)",
        ),
        SectionRule(
            literal="newborn feeding",
            category="FEEDING",
            pattern=r"(?i)(?:\b|^)(?:newborn\s+feeding|infant\s+feeding)(?:\s+|:|$)",
        ),
        
        # Postpartum sections
        SectionRule(
            literal="postpartum",
            category="POSTPARTUM",
            pattern=r"(?i)(?:\b|^)(?:postpartum|post\s+partum)(?:\s+|:|$)",
        ),
        SectionRule(
            literal="delivery note",
            category="DELIVERY",
            pattern=r"(?i)(?:\b|^)(?:delivery\s+note|delivery\s+summary)(?:\s+|:|$)",
        ),
    ]
    
    return section_rules 