
from collections import Counter
import pandas as pd
from typing import Tuple

def filter_entities(ent_df: pd.DataFrame) -> Tuple[dict[int, str], Counter]:
    """
    Filter entities based on specific criteria and classify sentences.
    
    This function filters out entities that are negated, uncertain, or related to intent,
    as well as those from specific sections like allergies or physical exam. It then
    groups the remaining entities by sentence and returns their classifications.
    
    Args:
        ent_df (pd.DataFrame): DataFrame containing entity information with columns for
                              sentence_classification, is_negated, is_uncertain, intent, 
                              section, and sentence_idx.
    
    Returns:
        tuple: A tuple containing:
            - dict: Mapping of sentence indices to their classifications
            - Counter: Count of each classification type in the filtered results
    """

    filters = [
        "is_negated == False",
        "is_uncertain == False",
        "intent == False",
        "section != 'family_history'",
        "section != 'social_history'",
        "section != 'other'",
        "DISCONTINUED == False",
    ]

    # group by sentence_idx and apply filters
    filtered_df = ent_df.query(' and '.join(filters))
    
    # Get a list of doc_ids where entity == 'nursing' and NURSING != True
    nursing_doc_ids = filtered_df.query("entity == 'nursing' and NURSING != True").doc_id.unique() # get the bad doc_ids
    filtered_df = filtered_df.query("doc_id not in @nursing_doc_ids") # filter out the bad doc_ids

    # Filter out the recommendation/plan and impression mentions within the observation and plan section
    filtered_df['contains_impression'] = filtered_df['sentence'].str.lower().str.contains('impression:')
    filtered_df['contains_recommendation'] = filtered_df['sentence'].str.lower().str.contains('recommendation/plan:')
    planning_ids = filtered_df.query("section == 'observation_and_plan' and (contains_impression == True or contains_recommendation == True)").doc_id.unique()
    filtered_df = filtered_df.query("doc_id not in @planning_ids")

    # recalculate the sentence_classification
    # Group by doc_id and entity_label, then unstack to get entity counts per document
    entity_counts = filtered_df.groupby(['doc_id', 'entity_label']).size().unstack().fillna(0)

    # Create a categorical column based on the specified conditions
    def categorize_feeding(row):
        if row.get('BREAST_FEEDING', 0) > 0:
            return 'BREAST_FEEDING'
        elif row.get('BREAST_FEEDING', 0) == 0 and row.get('FORMULA_FEEDING', 0) > 0:
            return 'FORMULA'
        else:
            return 'NONE'

    entity_counts['sentence_classification'] = entity_counts.apply(categorize_feeding, axis=1)

    # merge the sentence_classification back into the filtered_df
    filtered_df = filtered_df.drop(columns=['sentence_classification']).merge(entity_counts[['sentence_classification']], left_on='doc_id', right_index=True)
    filtered_df = filtered_df.drop(columns=['contains_impression', 'contains_recommendation'])

    result = filtered_df.groupby('sentence_idx')['sentence_classification'].unique().to_dict()
    result = {key: value[0] for key, value in result.items()}
    return result, Counter(list(result.values()))

def simple_note_level_classifier(df: pd.DataFrame, apply_filters: bool = True) -> str:
    """
    Classify a clinical note at the document level based on filtered entity classifications.
    
    This function determines the overall feeding classification of a clinical note by analyzing
    the classifications of individual sentences. It handles various scenarios including exclusive
    breastfeeding, exclusive formula feeding, mixed feeding, and ambiguous cases.
    
    Args:
        df (pd.DataFrame): DataFrame containing entity information with necessary columns for
                          filtering through the filter_entities function.
    
    Returns:
        str: The document-level classification, one of:
             - 'BREAST_FEEDING': Exclusively breastfeeding
             - 'FORMULA_FEEDING': Exclusively formula feeding
             - 'MIXED': Both breast and formula feeding
             - 'UNDETERMINED': Contains ambiguous classifications
             - 'NONE-NOT_FEEDING_RELATED': No relevant feeding information
             - 'ERROR PLEASE CHECK': Unexpected classification pattern
    """
    if apply_filters:
        sentence_classifications, _ = filter_entities(df)
    else:
        # sentence_classifications = df['sentence_classification'].to_dict()
        result = df.groupby('sentence_idx')['sentence_classification'].unique().to_dict()
        result = {key: value[0] for key, value in result.items()}
        sentence_classifications = result
    if not sentence_classifications: # no entities found?
        return 'NONE-NOT_FEEDING_RELATED'
    
    values = list(sentence_classifications.values())
    
    # Check if all values are the same
    if all(val == 'BREAST_FEEDING' for val in values):
        return 'BREAST_FEEDING'
    elif all(val == 'FORMULA_FEEDING' for val in values):
        return 'FORMULA_FEEDING'
    
    # Check for ambiguous classifications
    ## only flag as ambiguous if all entities are ambiguous. Ignore other mentions.
    # if any(val == 'AMBIGUOUS' or val == 'NONE-FEEDING_RELATED' for val in values):
    if all(val == 'AMBIGUOUS' for val in values):
        return 'UNDETERMINED'
    
    ## only flag as none-feeding related if all entities are none-feeding related. Ignore other mentions.
    if all(val == 'NONE-FEEDING_RELATED' for val in values):
        return 'NONE-NOT_FEEDING_RELATED'
    
    # Check for mixed feeding (both breast and formula)
    has_breast = any(val == 'BREAST_FEEDING' for val in values)
    has_formula = any(val == 'FORMULA_FEEDING' for val in values)
    is_mixed = any(val == 'MIXED' for val in values)
    if has_breast and has_formula or is_mixed:
        return 'MIXED'
    
    # Flag all documents with any breast feeding mentions as 'BREAST_FEEDING'
    ## This overwrites some of the above checks but oh well.
    if has_breast:
        return 'BREAST_FEEDING'
    if not has_breast and has_formula:
        return 'FORMULA_FEEDING'
    if not has_breast and not has_formula:
        return 'NONE-NOT_FEEDING_RELATED'
    return 'ERROR PLEASE CHECK'