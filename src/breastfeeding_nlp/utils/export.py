"""Utilities for exporting NLP pipeline results to HuggingFace datasets."""

from typing import Dict, Any, List, Optional
import datasets
from pydantic import BaseModel, Field
import spacy


class ExportConfig(BaseModel):
    """Configuration for exporting to HuggingFace datasets."""

    include_token_attributes: List[str] = Field(
        # ["doc_id", "text", "is_alpha", "is_punct", "is_like_num", "pos", "tag", "ent_type_", "lemma"],
        ["doc_id", "text", "is_alpha", "is_punct", "ent_type_"],
        description="Token attributes to include in the export",
    )
    include_entity_attributes: List[str] = Field(
        ["doc_id", "text", "label_", "start", "end", "is_negated", "is_historical", "is_uncertain", "intent", 
         "section", "EXCLUSIVE", "AMOUNT", "FREQUENCY", "BOTTLE", "DISCONTINUED", "NURSING", "BREAST"],
        description="Entity attributes to include in the export",
    )
    include_sentence_attributes: List[str] = Field(
        ["doc_id", "text", "start", "end"],
        description="Sentence attributes to include in the export",
    )


def token_to_dict(token: spacy.tokens.Token, config: ExportConfig) -> Dict[str, Any]:
    """
    Convert a spaCy token to a dictionary with selected attributes.
    
    Args:
        token: The spaCy token to convert
        config: Export configuration specifying which attributes to include
        
    Returns:
        Dictionary with token attributes
    """
    result = {}
    for attr in config.include_token_attributes:
        # Handle head.text specially
        if attr == "head.text":
            result["head_text"] = token.head.text
        else:
            # Try to get the attribute, default to None if not found
            try:
                result[attr] = getattr(token, attr)
            except AttributeError:
                result[attr] = None
    
    return result


def entity_to_dict(ent: spacy.tokens.Span, config: ExportConfig) -> Dict[str, Any]:
    """
    Convert a spaCy entity span to a dictionary with selected attributes.
    
    Args:
        ent: The spaCy entity span to convert
        config: Export configuration specifying which attributes to include
        
    Returns:
        Dictionary with entity attributes
    """
    result = {}
    for attr in config.include_entity_attributes:
        # Try to get the attribute directly from the entity
        try:
            result[attr] = getattr(ent, attr)
        except AttributeError:
            # Try to get from the underscore extensions
            try:
                result[attr] = ent._.get(attr)
            except (AttributeError, KeyError):
                result[attr] = None
    
    return result


def sentence_to_dict(sent: spacy.tokens.Span, config: ExportConfig) -> Dict[str, Any]:
    """
    Convert a spaCy sentence span to a dictionary with selected attributes.
    
    Args:
        sent: The spaCy sentence span to convert
        config: Export configuration specifying which attributes to include
        
    Returns:
        Dictionary with sentence attributes
    """
    result = {}
    for attr in config.include_sentence_attributes:
        # Try to get the attribute directly from the sentence
        try:
            result[attr] = getattr(sent, attr)
        except AttributeError:
            # Try to get from the underscore extensions
            try:
                result[attr] = sent._.get(attr)
            except (AttributeError, KeyError):
                result[attr] = None
    
    return result


def to_huggingface_dataset(doc, config: Optional[ExportConfig] = None) -> datasets.Dataset:
    """
    Convert a processed spaCy Doc to a HuggingFace dataset.
    
    Args:
        doc: The processed spaCy Doc
        config: Export configuration
        
    Returns:
        HuggingFace Dataset with token, sentence, and entity level data
    """
    if config is None:
        config = ExportConfig()
    
    # Extract token-level data
    tokens_data = []
    for token in doc:
        token_dict = token_to_dict(token, config)
        token_dict["doc_id"] = doc._.doc_id
        tokens_data.append(token_dict)
    
    # Create a mapping from sentence start index to enumeration index
    sent_start_to_idx = {}
    
    # Extract sentence-level data
    sentences_data = []
    for i, sent in enumerate(doc.sents):
        sent_dict = sentence_to_dict(sent, config)
        sent_dict["doc_id"] = doc._.doc_id
        sent_dict["sentence_idx"] = i
        
        # Store the mapping from sentence start token index to enumeration index
        sent_start_to_idx[sent.start] = i
        
        # Get the classification for this sentence if available
        sent_dict["classification"] = getattr(sent._, "classification", None)
        
        sentences_data.append(sent_dict)
    
    # Extract entity-level data
    entities_data = []
    for ent in doc.ents:
        ent_dict = entity_to_dict(ent, config)
        ent_dict["doc_id"] = doc._.doc_id
        
        # Use the sentence's enumeration index instead of its start token index
        sent_start = doc[ent.start].sent.start
        ent_dict["sentence_idx"] = sent_start_to_idx.get(sent_start, None)
        
        try:
            ent_dict["section"] = doc[ent.start]._.section.category
        except AttributeError:
            ent_dict["section"] = None
        entities_data.append(ent_dict)
    
    # Create separate datasets for each level
    tokens_dataset = datasets.Dataset.from_list(tokens_data)
    sentences_dataset = datasets.Dataset.from_list(sentences_data)
    entities_dataset = datasets.Dataset.from_list(entities_data)
    
    # Create a DatasetDict to hold all levels
    dataset_dict = datasets.DatasetDict({
        "tokens": tokens_dataset,
        "sentences": sentences_dataset,
        "entities": entities_dataset,
    })
    
    return dataset_dict