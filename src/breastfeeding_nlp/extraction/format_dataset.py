from datasets import Dataset
import pandas as pd

def format_entity_ds(ds: Dataset) -> pd.DataFrame:
    """
    Format entity dataset into a pandas DataFrame with merged sentence information.
    
    This function takes a Dataset containing entities and sentences, merges them based on
    sentence_idx, and renames columns for clarity. It also attempts to clean document IDs
    by removing file extensions.
    
    Args:
        ds: A Dataset object containing 'entities' and 'sentences' tables
        
    Returns:
        A pandas DataFrame with entity information merged with sentence context
    """
    ent_df = ds['entities'].to_pandas()
    sent_df = ds['sentences'].to_pandas()

    ent_df = ent_df.merge(sent_df[['sentence_idx', 'text', 'classification']], on='sentence_idx', how='left')
    ent_df.rename(columns={'text_x': 'entity', 
                        'text_y': 'sentence',
                        'label_': 'entity_label',
                        'classification': 'sentence_classification',
                        'EXCLUSIVE': 'is_exclusive',
                        'AMOUNT': 'amount',
                        }, 
                        inplace=True)

    try:
        ent_df.doc_id = ent_df.doc_id.apply(lambda x: x.split('.')[0])
    except:
        pass
    return ent_df