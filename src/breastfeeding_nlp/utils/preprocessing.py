"""Text preprocessing utilities for the breastfeeding NLP pipeline."""

import re
from spellchecker import SpellChecker
from typing import Dict, Any, Optional
import spacy
from pydantic import BaseModel, Field
import medspacy
from breastfeeding_nlp.utils.abbreviation_expansion import AbbreviationExpander

class PreprocessingConfig(BaseModel):
    """Configuration for text preprocessing."""

    lowercase: bool = Field(True, description="Convert text to lowercase")
    remove_non_alphanumeric: bool = Field(False, description="Remove non-alphanumeric characters")
    lemmatize: bool = Field(False, description="Apply lemmatization")
    strip_whitespace: bool = Field(True, description="Strip unnecessary whitespace")
    spacy_model: str = Field("en_core_web_sm", description="SpaCy model to use for preprocessing")
    llm_abbreviation_expansion: bool = Field(False, description="Expand abbreviations using LLM")
    use_preexpanded_abbreviations: bool = Field(False, description="Use pre-expanded abbreviations")
    typo_resolution: bool = Field(True, description="Resolve typos in text")


def preprocess_text(
    text: str, config: Optional[PreprocessingConfig] = None
) -> Dict[str, Any]:
    """
    Preprocess text for NLP analysis.

    This function applies several preprocessing steps to normalize text:
    1. Convert to lowercase (optional)
    2. Remove non-alphanumeric characters (optional)
    3. Apply part-of-speech-informed lemmatization (optional)
    4. Strip unnecessary whitespace (optional)

    Args:
        text: The input text to preprocess
        config: Configuration object for preprocessing steps

    Returns:
        A dictionary containing the preprocessed text and metadata
    """
    if config is None:
        config = PreprocessingConfig()

    # Store original text
    result = {"original_text": text}

    # Apply preprocessing steps
    processed_text = text

    # putting this before the pre_expand to reduce redundancy
    if config.llm_abbreviation_expansion: # implicitly also pre_expands, FYI
        # Initialize the abbreviation expander
        aae = AbbreviationExpander()
        processed_text = aae.expand_abbreviations(processed_text)
        result["abbreviation_expanded"] = True

    if config.use_preexpanded_abbreviations and not config.llm_abbreviation_expansion: # only calling if not already expanded
        # Initialize the abbreviation expander
        aae = AbbreviationExpander()
        processed_text = aae.pre_expand_abbreviations(processed_text)
        result["pre_expanded_abbreviations_applied"] = True

    # typos
    if config.typo_resolution:
        processed_text = correct_text(processed_text)
        result["typo_resolved"] = True
    
    # Convert to lowercase
    if config.lowercase:
        processed_text = processed_text.lower()
        result["lowercase_applied"] = True

    # Strip unnecessary whitespace
    if config.strip_whitespace:
        processed_text = re.sub(r"\s+", " ", processed_text).strip()
        result["whitespace_stripped"] = True

    # Apply lemmatization and remove non-alphanumeric characters
    if config.lemmatize or config.remove_non_alphanumeric:
        # Load spaCy model
        # TODO: This is why the sentence segmentation is not working
        nlp = spacy.load(config.spacy_model, disable=["ner", "parser"])
        
        # Process text with spaCy
        doc = nlp(processed_text)
        
        # Apply lemmatization and/or remove non-alphanumeric characters
        tokens = []
        for token in doc:
            if config.remove_non_alphanumeric and not token.is_alpha and not token.is_digit and not token.is_punct:
                continue
            
            if config.lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)
        
        processed_text = " ".join(tokens)
        
        result["lemmatization_applied"] = config.lemmatize
        result["non_alphanumeric_removed"] = config.remove_non_alphanumeric

    # Store the final preprocessed text
    result["preprocessed_text"] = processed_text

    return result 

def correct_text(text):
    """
    Corrects typos in the provided text using PySpellChecker with a Levenshtein distance of 1.
    
    Parameters:
        text (str): The text that may contain typos.
        
    Returns:
        str: The corrected text or original text if an error occurs.
        
    Raises:
        ValueError: If input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    if not text:
        return ""
    
    try:
        # Initialize the spell checker with a distance of 1
        spell = SpellChecker(distance=1)
        
        # Tokenize the text. This regex splits text into words (including words with apostrophes) and punctuation.
        tokens = re.findall(r'\w+\'?\w*|[^\w\s]', text, re.UNICODE)
        
        corrected_tokens = []
        for token in tokens:
            try:
                # Check if the token is alphabetic (i.e., a potential word)
                if token.isalpha() or (token.replace("'", "").isalpha() and "'" in token):
                    # Check if the token is known (using lowercasing for matching)
                    if token.lower() in spell:
                        corrected_tokens.append(token)
                    else:
                        # Get the most likely correction
                        corrected = spell.correction(token.lower())
                        if corrected is None:
                            corrected_tokens.append(token)
                            continue
                            
                        # Preserve original capitalization if necessary
                        if token[0].isupper() and corrected:
                            corrected = corrected.capitalize()
                        corrected_tokens.append(corrected)
                else:
                    # If not a word (e.g. punctuation), keep the token as is
                    corrected_tokens.append(token)
            except (AttributeError, IndexError) as e:
                # If any token processing fails, keep the original token
                corrected_tokens.append(token)
        
        # Find the original positions of hyphens in the text to preserve spacing
        hyphen_positions = [m.start() for m in re.finditer('-', text)]
        
        # Rebuild the text with proper spacing
        corrected_text = ""
        last_end = 0
        
        for i, token in enumerate(corrected_tokens):
            if token is None:
                continue
                
            try:
                # Handle hyphen specially to preserve original spacing
                if token == '-':
                    corrected_text += token
                # Add punctuation without spaces
                elif re.match(r'[^\w\s]', token):
                    corrected_text += token
                # Add words with appropriate spacing
                elif i > 0 and corrected_text:
                    # Check if we're adding a word after a hyphen
                    if corrected_text.endswith('-'):
                        corrected_text += token  # No space after hyphen
                    else:
                        corrected_text += " " + token
                else:
                    corrected_text += token
            except (TypeError, AttributeError):
                # Fallback for any token that can't be processed normally
                if corrected_text and i > 0:
                    if corrected_text.endswith('-'):
                        corrected_text += str(token)  # No space after hyphen
                    else:
                        corrected_text += " " + str(token)
                else:
                    corrected_text += str(token)
        
        # Add a space between "." and "-" if they appear together
        corrected_text = re.sub(r'\.-', '. -', corrected_text)
        corrected_text = re.sub(r'\:-', ': -', corrected_text)
                    
        return corrected_text
    
    except Exception as e:
        # Log the error (in a production environment, use proper logging)
        print(f"Error in spell correction: {str(e)}")
        # Return original text if spell correction fails
        return text
