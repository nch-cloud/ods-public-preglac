# You'll need to log into AWS first and have access to the model and buckets.

import os
from typing import Dict, List, Tuple, Optional, Any
# import numpy as np
import pandas as pd
# from collections import Counter
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path

from breastfeeding_nlp.pipeline import BreastfeedingNLPPipeline, BreastfeedingNLPConfig
from breastfeeding_nlp.utils.preprocessing import PreprocessingConfig
# from breastfeeding_nlp.utils.abbreviation_expansion import TextReplacer
# from breastfeeding_nlp.utils.deid_notes import load_notes_to_dataframe
from breastfeeding_nlp.extraction.format_dataset import format_entity_ds
from breastfeeding_nlp.classification.note_level import simple_note_level_classifier # filter_entities
# from breastfeeding_nlp.utils.agents import CorrectionAgent
# from breastfeeding_nlp.prompts import typo_resolution_prompt, nursing_disambiguation_prompt
from breastfeeding_nlp.utils.utils import OptimizedDataFrame


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load data from Excel file and perform initial preprocessing.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Optimized DataFrame with preprocessed data
    """
    df = pd.read_excel(file_path)
    # Replace NaN values in NOTE_TYPE column with 'misc'
    df['NOTE_TYPE'] = df['NOTE_TYPE'].fillna('misc')
    return OptimizedDataFrame(df).optimize(show_mem_reduction=True)


def create_nlp_pipeline(size: str) -> BreastfeedingNLPPipeline:
    """
    Create and configure the NLP pipeline.
    
    Returns:
        Configured NLP pipeline
    """
    preprocessing_config = PreprocessingConfig(use_preexpanded_abbreviations=False,
                                               typo_resolution=False)
    main_config = BreastfeedingNLPConfig(preprocessing_config=preprocessing_config)
    
    return BreastfeedingNLPPipeline(
        nlp_method=size,
        config=main_config,
        target_rules_on_match={
            "EXCLUSIVE": ["exclusive", "exclusively"],
            "BOTTLE": ['bottle'],
            "DISCONTINUED": ['d/c', 'd/c\'d', 'discontinued', "discontinued'd", "DISCONTD", "DISCONT'D", "stopped"]
        }
    )


def process_note(pipeline: BreastfeedingNLPPipeline, note_id: str, notes: Dict[str, str]) -> Tuple[Any, Any]:
    """
    Process a single note through the NLP pipeline.
    
    Args:
        pipeline: The NLP pipeline
        note_id: ID of the note to process
        notes: Dictionary mapping note IDs to note text
        
    Returns:
        Tuple of (dataset, result) from pipeline processing
    """
    try:
        result = pipeline.process(
            text=notes[note_id],
            doc_id=note_id)
    except KeyError:  # For testing - the note_id is actually the text
        # TODO: probably bad to keep this here. Once testing is done, need to remove this.
        result = pipeline.process(text=note_id)
    
    return pipeline.to_hf_ds(result), result


def create_empty_entity_dataframe(note_id: str) -> pd.DataFrame:
    """
    Create an empty entity dataframe with the correct structure.
    
    Args:
        note_id: ID of the note
        
    Returns:
        Empty DataFrame with entity structure
    """
    empty_df = pd.DataFrame(columns=[
        'doc_id', 'entity', 'entity_label', 'start', 'end', 
        'is_negated', 'is_uncertain', 'intent', 'section', 
        'sentence_idx', 'sentence', 'sentence_classification'
    ])
    empty_df['doc_id'] = [note_id]
    return empty_df


def process_patient_notes(patient_df: pd.DataFrame, pipeline: BreastfeedingNLPPipeline) -> pd.DataFrame:
    """
    Process all notes for a single patient.
    
    Args:
        patient_df: DataFrame containing patient's notes
        pipeline: The NLP pipeline
        
    Returns:
        DataFrame with processed results
    """
    # Create a dictionary of notes
    notes = {f"{row['VISIT_ID']}_{row['NOTE_TYPE']}_{idx}": row['NOTE_TEXT'] 
             for idx, row in patient_df.iterrows()}
    
    # Process each note
    ent_dfs = []
    for note_id in tqdm(notes):
        res_ds, _ = process_note(pipeline, note_id, notes)
        
        # This is the sentence level results. Modify things here.
        if res_ds['entities'].to_pandas().shape[0] > 0:
            ent_df = format_entity_ds(res_ds)  # Format entities and add sentence data
            # Modify rows with nursing entities. If NURSING == False, set entity_label to NONE-NOT_FEEDING_RELATED
            ent_df.loc[ent_df['NURSING'] == False, 'entity_label'] = 'NONE-NOT_FEEDING_RELATED'
            ent_dfs.append(ent_df)
        else:  # No entities found
            empty_df = create_empty_entity_dataframe(note_id)
            ent_dfs.append(empty_df)
    
    # Combine and process results
    if not ent_dfs:
        return patient_df  # Return original if no entities found
        
    res = pd.concat(ent_dfs).sort_values(by=['doc_id', 'sentence_idx']).reset_index(drop=True)
    res[['doc_id', 'NOTE_TYPE', 'idx']] = res.doc_id.str.split('_', expand=True)
    res['doc_id'] = res['doc_id'].astype(int)
    res['idx'] = res['idx'].astype(int)
    
    # Merge with patient data
    merged_res = patient_df.reset_index().merge(
        res, 
        left_on=['VISIT_ID', 'NOTE_TYPE', 'index'], 
        right_on=['doc_id', 'NOTE_TYPE', 'idx'], 
        how='outer'
    )
    merged_res = merged_res.fillna({"idx": merged_res['index']})
    merged_res.sentence_classification.fillna("NONE-NOT_FEEDING_RELATED", inplace=True)
    merged_res = merged_res.sort_values('index')
    
    return merged_res


def classify_notes(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    Classify notes based on their content.
    
    Args:
        merged_data: DataFrame with merged patient and entity data
        
    Returns:
        DataFrame with classification results
    """
    # Group by visit_id, note_type, and idx
    visit_groups = merged_data.groupby(["VISIT_ID", "NOTE_TYPE", 'idx'])
    
    # Classify each note
    # TODO: once with filtering, once without filtering
    results_1 = {}
    results_2 = {}
    for visit_id, visit_data in visit_groups:
        result_1 = simple_note_level_classifier(visit_data, apply_filters=True)
        result_2 = simple_note_level_classifier(visit_data, apply_filters=False)
        results_1[visit_id] = result_1
        results_2[visit_id] = result_2
    
    # Convert results to DataFrame
    filtered_results_df = pd.DataFrame([
        {
            'VISIT_ID': key[0],
            'NOTE_TYPE': key[1],
            'idx': key[2],
            'classification': value
        }
        for key, value in results_1.items()
    ])
    unfiltered_results_df = pd.DataFrame([
        {
            'VISIT_ID': key[0],
            'NOTE_TYPE': key[1],
            'idx': key[2],
            'classification': value
        }
        for key, value in results_2.items()
    ])
    
    return filtered_results_df.sort_values("idx").set_index("idx"), unfiltered_results_df.sort_values("idx").set_index("idx")


def merge_classifications(patient_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge classification results back into patient data.
    
    Args:
        patient_df: Original patient DataFrame
        results_df: DataFrame with classification results
        
    Returns:
        Merged DataFrame
    """
    return patient_df.reset_index().merge(
        results_df.reset_index(),
        left_on=['VISIT_ID', 'NOTE_TYPE', 'index'],
        right_on=['VISIT_ID', 'NOTE_TYPE', 'idx'],
        how='outer'
    ).drop(columns=['idx'])


def process_all_patients(df: pd.DataFrame, 
                         output_dir: str,
                         size: str
                         ) -> pd.DataFrame:
    """
    Process all patients in the dataset.
    
    Args:
        df: DataFrame with all patient data
        output_dir: Directory to save results
        
    Returns:
        DataFrame with results for all patients
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create pipeline once to reuse
    pipeline = create_nlp_pipeline(size=size)

    # Group by patient ID
    patient_groups = df.groupby('PAT_ID')
    num_patient_groups = len(patient_groups)
    _count = 0

    filtered_patient_group_results = []
    unfiltered_patient_group_results = []
    for patient_id, patient_df in patient_groups:
        _count += 1
        print(f"Processing patient {_count} of {num_patient_groups}")
        try:
            # drop notes with no text
            patient_df = patient_df[patient_df['NOTE_TEXT'].notna()]

            # Process patient notes
            merged_data = process_patient_notes(patient_df, pipeline)
            
            # Classify notes
            filtered_results_df, unfiltered_results_df = classify_notes(merged_data)
            
            # Merge classifications back into patient data
            filtered_final_patient_df = merge_classifications(patient_df, filtered_results_df)
            unfiltered_final_patient_df = merge_classifications(patient_df, unfiltered_results_df)
            
            # Save patient results
            filtered_final_patient_df.to_csv(f"{output_dir}/{patient_id}_filtered.csv", index=False)
            unfiltered_final_patient_df.to_csv(f"{output_dir}/{patient_id}_unfiltered.csv", index=False)
            
            filtered_patient_group_results.append(filtered_final_patient_df)
            unfiltered_patient_group_results.append(unfiltered_final_patient_df)
        except TypeError: # two notes are missing -- assuming that's what was causing the error
            # TODO: append a blank row to the results
            print(f"Error with patient {patient_id}")
            pass
    
    # Combine all results
    filtered_all_df = pd.concat(filtered_patient_group_results)
    unfiltered_all_df = pd.concat(unfiltered_patient_group_results)
    # Add binary column for feeding related
    filtered_all_df = filtered_all_df.assign(feeding_related=filtered_all_df['classification'].notna() & 
              ~filtered_all_df['classification'].str.contains('NONE-NOT_FEEDING_RELATED', na=False))
    unfiltered_all_df = unfiltered_all_df.assign(feeding_related=unfiltered_all_df['classification'].notna() & 
              ~unfiltered_all_df['classification'].str.contains('NONE-NOT_FEEDING_RELATED', na=False))
    return filtered_all_df, unfiltered_all_df


# def main():
#     """Main function to run the entire pipeline."""
#     # Load and preprocess data
#     input_file = "/Volumes/specialpermissions/RISIDataServices_MPrint_NCH/MPRINT_LACTATE_1_20250327.xlsx"
#     output_dir = '../../../dev_cg/data/patient_group_results'
    
#     df = load_and_preprocess_data(input_file)
    
#     # Process all patients
#     all_results = process_all_patients(df, output_dir)
    
#     # Save combined results
#     all_results.to_csv(f"{output_dir}/patient_group_results.csv", index=False)


# if __name__ == "__main__":
#     main()
