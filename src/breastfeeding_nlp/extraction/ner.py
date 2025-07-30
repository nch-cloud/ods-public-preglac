import pandas as pd
from typing import Union
from breastfeeding_nlp.utils.text_clean_up import clean_up_text
from breastfeeding_nlp.data.ontology_prep.get_phenotypes import extract_phenotypes_df
from breastfeeding_nlp.data.ontology_prep.ancestry import load_ontology, rollup


from breastfeeding_nlp.classification.temp import (
    create_nlp_pipeline,
    process_note,
    format_entity_ds,
    create_empty_entity_dataframe
)

class ClinPhenLabeler:
    def __init__(self, ontology_path: str, root_term: str = "FPO:0000003"):
        """
        Initialize the labeler with the ontology file path and root term for rollup.
        """
        self.ontology = load_ontology(ontology_path)
        self.root_term = root_term

    def tag_text(self, text: str, row_ix: Union[int, str] = 0) -> pd.DataFrame:
        """
        Tag a single note text with ClinPhen, returning a DataFrame of extracted entities
        with the row index set to row_ix.
        """
        note_text = clean_up_text(text)
        entities = extract_phenotypes_df(f":{note_text}")
        if entities.empty:
            cols = ['term_id', 'term_name', 'synonym_match', 'context', 'earliness', 'positive']
            return pd.DataFrame(columns=cols, index=[row_ix])
        entities.index = [row_ix] * len(entities)
        return entities

    def tag_dataframe(self, df: pd.DataFrame, text_col: str = "NOTE_TEXT", index_col: str = None) -> pd.DataFrame:
        """
        Tag an entire DataFrame of notes. Returns a DataFrame of all extracted entities
        with a 'row_ix' column linking back to the original rows.
        """
        records = []
        if index_col:
            df = df.copy()
            df["_row_ix_"] = df[index_col]
            idx = "_row_ix_"
        else:
            df = df.copy().reset_index().rename(columns={"index": "_row_ix_"})
            idx = "_row_ix_"

        for row_ix, text in zip(df[idx], df[text_col]):
            tagged = self.tag_text(text, row_ix)
            records.append(tagged)

        result = pd.concat(records)
        return result

    def label_documents(self, entities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Roll up entities to ancestors, pivot counts, and assign document-level labels.
        Returns a DataFrame with columns ['row_ix', 'document_label'].
        """
        df = entities_df.reset_index(names=['row_ix']).copy()
        df["top_ancestor"] = df["term_name"].apply(
            lambda name: rollup(name, self.ontology, self.root_term)
        )

        pivot = (
            df.groupby(["row_ix", "top_ancestor"], dropna=False)
            .size()
            .unstack(fill_value=0)
        )

        # Rename any None/NaN column to "MISSING"
        missing_cols = [col for col in pivot.columns if pd.isna(col) or col is None]
        for col in missing_cols:
            pivot.rename(columns={col: "MISSING"}, inplace=True)

        labels = pivot.apply(self._assign_label, axis=1)
        return labels.reset_index().rename(columns={0: "document_label"})

    def get_document_label(self, text: str, row_ix: Union[int, str] = 0) -> str:
        """
        Convenience method: tag a single text and return its document label.
        """
        entities = self.tag_text(text, row_ix)
        labels_df = self.label_documents(entities)
        return labels_df.loc[labels_df["row_ix"] == row_ix, "document_label"].iloc[0]

    @staticmethod
    def _assign_label(row: pd.Series, breastfeeding_label: str = "human breast milk", formula_label: str = "formula") -> str:
        if row.get(breastfeeding_label, 0) > 0:
            return "positive"
        if row.get(formula_label, 0) > 0 or row.get("FEEDING_RELATED", 0) > 0:
            return "negative"
        if row.get("AMBIGUOUS", 0) > 0:
            return "absent"
        return "absent"

# ------------------------------------------------------------------------
# MedSpaCyLabeler: Wrapper for running medSpaCy pipeline on single texts or DataFrames
class MedSpaCyLabeler:
    """
    Wrapper for running the medSpaCy pipeline on single texts or entire DataFrames.
    """
    def __init__(self, size: str = "custom"):
        """
        Initialize and load the medSpaCy pipeline once.
        """
        self.pipeline = create_nlp_pipeline(size=size)

    def process_text(self, text: str, doc_id: Union[int, str] = 0) -> pd.DataFrame:
        """
        Process a single note string through the pipeline and return entities DataFrame.
        """
        cleaned = clean_up_text(text)
        notes = {doc_id: cleaned}
        hf_ds, _ = process_note(self.pipeline, doc_id, notes)
        try:
            df = format_entity_ds(hf_ds)
        except KeyError:
            df = create_empty_entity_dataframe(doc_id)
        df["NOTE_TEXT"] = cleaned
        df["Document_ID"] = doc_id
        return df

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "NOTE_TEXT",
        index_col: str = None
    ) -> pd.DataFrame:
        """
        Process an entire DataFrame of notes. Returns a concatenated DataFrame of all entities.
        """
        records = []
        if index_col:
            df_copy = df.copy()
            df_copy["_doc_id_"] = df_copy[index_col]
            idx = "_doc_id_"
        else:
            df_copy = df.copy().reset_index().rename(columns={"index": "_doc_id_"})
            idx = "_doc_id_"

        for doc_id, text in zip(df_copy[idx], df_copy[text_col]):
            rec = self.process_text(text, doc_id)
            records.append(rec)

        return pd.concat(records).reset_index(drop=True)

    # Standard mapping for medSpaCy entity labels to document-level labels
    STANDARD_ENTITY_LABELS = {
        "FORMULA_FEEDING": "negative",
        "FEEDING_RELATED": "absent",
        "BREAST_FEEDING": "positive",
        "AMBIGUOUS": "absent"
    }

    def label_documents(self, entities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map medSpaCy entity labels, roll up to document-level counts, and assign a document label.
        Returns a DataFrame with columns ['row_ix', 'medspacy_document_label'].
        """
        df = entities_df.copy()
        # Map and fill missing labels
        df["entity_label"] = df["entity_label"].map(self.STANDARD_ENTITY_LABELS).fillna("absent")

        # Drop unneeded columns
        drop_cols = [
            "start", "end", "section", "is_exclusive", "amount",
            "FREQUENCY", "BOTTLE", "BREAST", "sentence_idx",
            "NOTE_TEXT", "Document_ID"
        ]
        red = df.drop(drop_cols, axis=1)

        # Override nursing mentions to positive
        # NOTE: Leave the following line commented out! "Nursing" is not handled well and removing it from 
            # consideration increases the positive precision by ~40 points.
        # red.loc[red.get("NURSING") == True, "entity_label"] = "positive"
        # Force all nursing mentions to absent
        mask = red['entity'].notna() & red['entity'].str.contains('nursing', na=False)
        red.loc[mask, 'entity_label'] = 'absent'

        # Pivot to get counts per document and label
        counts = red.groupby(["doc_id", "entity_label"]).size().unstack(fill_value=0)

        # Categorize document-level label
        def categorize_feeding(row: pd.Series) -> str:
            if row.get("positive", 0) > 0:
                return "positive"
            elif row.get("positive", 0) == 0 and row.get("negative", 0) > 0:
                return "negative"
            else:
                return "absent"

        counts["medspacy_document_label"] = counts.apply(categorize_feeding, axis=1)

        # Prepare final DataFrame
        result = (
            counts
            .reset_index()
            .rename(columns={"doc_id": "row_ix"})
            [["row_ix", "medspacy_document_label"]]
        )
        return result

