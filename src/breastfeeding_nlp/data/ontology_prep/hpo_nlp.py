#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified HPO Natural Language Processing module

This module unifies the functionality of get_phenotypes.py and ner.py,
providing options to extract HPO terms with or without context flags.
"""

import os
import re
import sys
from collections import defaultdict
import pandas as pd
from nltk.stem import WordNetLemmatizer
import itertools

# --- Constants & Global Initialization ---

# Paths to data files
# BASE_DIR = os.path.dirname(__file__)
BASE_DIR = "/Users/cxg042/Documents/git/ods-preglac-refactor/src/breastfeeding_nlp/data/ontology_prep"
ID_TO_SYN_MAP_FILE = f"{BASE_DIR}/hpo_synonyms.txt"
ID_TO_NAME_MAP_FILE = f"{BASE_DIR}/hpo_term_names.txt"
PHENOTYPIC_ABNORMALITY_ID = "FPO:0000001"

# Initialize lemmatizer once
LEMMATIZER = WordNetLemmatizer()

# Flag words (for negation, family mentions, etc.)
FLAGS = {
    'negative': [],
    'family': [],
    'healthy': [],
    'disease': [],
}
for flag in FLAGS:
    flag_file = os.path.join(f"/Users/cxg042/Documents/git/ods-preglac-refactor/src/breastfeeding_nlp/data/ontology_prep/flags/{flag}.txt")
    with open(flag_file, 'r') as fil:
        # Remove comments and lowercase all lines
        lines = fil.read().lower()
        lines = re.sub(r"#[^\n]*\n", "\n", lines)
        FLAGS[flag] = lines.split('\n')

# For mapping a flag word to its flag type
FLAG_TYPE = dict()

# --- Text Preprocessing Helpers ---
def lemmatize(word):
    """Lemmatize a word after removing non-alphanumeric characters."""
    word = re.sub('[^0-9a-zA-Z]+', '', word).lower()
    return LEMMATIZER.lemmatize(word)

def add_lemmas(word_set):
    """Return a new set containing the original words and their lemma variants."""
    lemmas = set()
    for word in word_set:
        lemma = lemmatize(word)
        if lemma:
            lemmas.add(lemma)
        lemmas |= synonym_lemmas(word)
        lemmas |= custom_lemmas(word)
    return word_set | lemmas


def synonym_lemmas(word):
    """Return the union of any common synonym sets that include the word."""
    result = set()
    for syn_set in common_synonyms:
        if word in syn_set:
            result |= syn_set
    return result

def custom_lemmas(word):
    """Generate custom lemma variants for a word based on common suffix rules."""
    result = set()
    if len(word) < 2:
        return result
    if word[-1] == "s":
        result.add(word[:-1])
    if word[-1] == "i":
        result.add(word[:-1] + "us")
    if word[-1] == "a":
        result.add(word[:-1] + "um")
        result.add(word[:-1] + "on")
    if len(word) < 3:
        return result
    if word[-2:] == "es" and word[:-2].lower() != "not":
        result.add(word[:-2])
        result.add(word[:-2] + "is")
    if word[-2:] == "ic":
        result.add(word[:-2] + "ia")
        result.add(word[:-2] + "y")
    if word[-2:] == "ly":
        result.add(word[:-2])
    if word[-2:] == "ed" and word[:-2].lower() != "not":
        result.add(word[:-2])
    if len(word) < 4:
        return result
    if word[-3:] == "ata":
        result.add(word[:-2])
    if word[-3:] == "ies":
        result.add(word[:-3] + "y")
    if word[-3:] == "ble":
        result.add(word[:-2] + "ility")
    if len(word) < 7:
        return result
    if word[-6:] == "bility":
        result.add(word[:-5] + "le")
    if len(word) < 8:
        return result
    if word[-7:] == "ication":
        result.add(word[:-7] + "y")
        result.add(word[:-7] + "ied")
    return result

def alphanum_only(word_set):
    """Convert each word in the set to its alphanumeric tokens (splitting on non-alphanumerics)."""
    result = set()
    for word in word_set:
        tokens = re.sub('[^0-9a-zA-Z]+', ' ', word).split()
        result.update(token for token in tokens if token)
    return result

# Common synonym sets for common medical modifiers
low_synonyms = {"low", "decreased", "decrease", "deficient", "deficiency", "deficit",
                "deficits", "reduce", "reduced", "lack", "lacking", "insufficient",
                "impairment", "impaired", "impair", "difficulty", "difficulties", "trouble"}
high_synonyms = {"high", "increased", "increase", "elevated", "elevate", "elevation"}
abnormal_synonyms = {"abnormal", "unusual", "atypical", "abnormality", "anomaly", "anomalies", "problem"}
common_synonyms = [low_synonyms, high_synonyms, abnormal_synonyms]

# When adding lemmas below, both original and variants are mapped
def add_flag_lemmas(flag_list):
    return add_lemmas(set(flag_list))

for f in FLAGS:
    for w in add_flag_lemmas(FLAGS[f]):
        FLAG_TYPE[w] = f

# --- Data Loading Functions ---

def load_hpo_names(filename=ID_TO_NAME_MAP_FILE):
    """Load HPO term names from file into a dictionary (id -> name)."""
    with open(filename, 'r') as f:
        return {x: name for x, name in (line.strip().split('\t') for line in f if line.strip())}

def load_all_hpo_synonyms(filename=ID_TO_SYN_MAP_FILE):
    """Return a map from an HPO ID to the set of its synonymous names."""
    syn_map = defaultdict(set)
    with open(filename) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                hpo, syn = parts[0], parts[1]
                syn_map[hpo].add(syn)
    return syn_map

# Load HPO name and synonym data once
ID_TO_NAME_MAP = load_hpo_names()
ID_TO_SYN_MAP = load_all_hpo_synonyms()

# --- Sentence & Record Parsing Helpers ---

point_enders = [".", "•", ";", "\t"]
def end_of_point(word):
    """Return True if the word implies the end of a sentence."""
    return word[-1] in point_enders or word in {"but", "except", "however", "though"}

subpoint_enders = [",", ":"]
def end_of_subpoint(word):
    """Return True if the word implies the end of a sub-sentence."""
    return word[-1] in subpoint_enders or word == "and"

def load_medical_record_linewise(medical_record):
    """Split a medical record (with colons) into sentences line by line."""
    lines = medical_record.split("\n")
    sentences = []
    for line in lines:
        if ":" not in line:
            continue
        cur_sentence = []
        for word in line.strip().split():
            if not word:
                continue
            cur_sentence.append(word.lower())
            if end_of_point(word):
                sentences.append(" ".join(cur_sentence))
                cur_sentence = []
        if cur_sentence:
            sentences.append(" ".join(cur_sentence))
    subsentence_sets = []
    for sent in sentences:
        subsents = []
        cur_subsent = []
        for word in sent.split():
            cur_subsent.append(word.lower())
            if end_of_subpoint(word):
                subsents.append(" ".join(cur_subsent))
                cur_subsent = []
        if cur_subsent:
            subsents.append(" ".join(cur_subsent))
        subsentence_sets.append(subsents)
    return subsentence_sets

def load_medical_record_subsentences(medical_record):
    """
    Parse a medical record into subsentences.
    This function first removes extraneous whitespace and then splits the record.
    """
    # First, join non-empty lines into one string and split by spaces.
    lines = [line.strip('\t') for line in medical_record.split("\n") if line]
    text = " ".join(lines)
    # Then use the linewise method for sentence/subsentence splitting.
    return load_medical_record_linewise(medical_record)

def load_mr_map(parsed_words):
    """
    Create a map from each token (from medical record words) to the set of subsentence indices in which it appears.
    """
    mr_map = defaultdict(set)
    for i, word_set in enumerate(parsed_words):
        for word in word_set:
            mr_map[word].add(i)
    return mr_map

def get_flags(line_tokens, flag_dict=FLAG_TYPE):
    """Return a list of flag types found in the list of tokens."""
    line_set = add_lemmas(set(line_tokens))
    found = []
    for word in line_set:
        if word in flag_dict:
            flag_type = flag_dict[word]
            if flag_type not in found:
                found.append(flag_type)
    return found

# --- Span Finding Helper ---

# def find_spans(text, terms):
#     """
#     Find character spans of given terms in text.
    
#     Args:
#         text (str): Original text (assumed lowercased).
#         terms (list): List of terms to search for.
        
#     Returns:
#         list: List of tuples (matched_term, start_idx, end_idx)
#     """
#     spans = []
#     for term in terms:
#         pattern = r'\b' + re.escape(term) + r'\b'
#         for match in re.finditer(pattern, text, re.IGNORECASE):
#             spans.append((term, match.start(), match.end()))
#     return spans

def find_spans(text, terms):
    """
    Find character spans of given terms in text.
    
    Args:
        text (str): Original text (assumed lowercased).
        terms (list): List of terms to search for.
        
    Returns:
        list: List of tuples (matched_term, start_idx, end_idx)
    """
    spans = []
    for term in terms:
        # Handle exact matches
        pattern = r'\b' + re.escape(term) + r'\b'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            spans.append((term, match.start(), match.end()))
        
        # Handle disjointed matches
        term_words = term.split()
        if len(term_words) > 1:
            # Create a pattern that allows for words in between the term words
            # but requires the words to appear in the correct order
            disjointed_pattern = r'\b' + r'\b.{1,30}?\b'.join(re.escape(word) for word in term_words) + r'\b'
            for match in re.finditer(disjointed_pattern, text, re.IGNORECASE):
                matched_text = match.group(0)
                # Only include if all term words are in the match
                if all(re.search(r'\b' + re.escape(word) + r'\b', matched_text, re.IGNORECASE) for word in term_words):
                    spans.append((term, match.start(), match.end()))
    
    return spans

# --- Medical Record Parsing (Unified for Both Modes) ---

def parse_medical_record(original_text, with_context):
    """
    Parse the medical record into subsentences and build supporting data.
    
    Returns:
        subsentences: list of subsentence strings.
        subsent_to_sentence: list mapping each subsentence to its full sentence context.
        mr_words: list of token sets (after lemmatization) per subsentence.
        mr_flags: list of flag lists per subsentence (empty list if not with_context).
        position_map: dict mapping subsentence index to its start position in original_text.
    """
    subsentence_sets = load_medical_record_subsentences(original_text)
    subsentences = []
    subsent_to_sentence = []
    mr_words = []
    mr_flags = [] if with_context else None
    position_map = {}
    current_pos = 0
    for subsents in subsentence_sets:
        # Combine subsentences to form the full sentence context.
        full_sentence = " ".join(subsents).strip()
        if original_text.startswith(":") and full_sentence.startswith(":"):
            full_sentence = full_sentence[1:]
        flags = []
        if with_context:
            clean_sentence = re.sub('[^0-9a-zA-Z]+', ' ', full_sentence)
            flags = get_flags(clean_sentence.split())
        for subsent in subsents:
            subsentences.append(subsent)
            subsent_to_sentence.append(full_sentence)
            # Tokenize and add lemmas from the subsentence.
            mr_words.append(add_lemmas(alphanum_only({subsent})))
            if with_context:
                mr_flags.append(flags)
            # Find the subsentence's position in the original text.
            search_start = current_pos
            pos = original_text.lower().find(subsent.lower(), search_start)
            if pos >= 0:
                position_map[len(subsentences) - 1] = pos
                current_pos = pos + len(subsent)
    return subsentences, subsent_to_sentence, mr_words, mr_flags, position_map

# --- Main Extraction Function ---

def extract_hpo_terms(text, with_context=True, id_to_name_map=ID_TO_NAME_MAP, id_to_syn_map=ID_TO_SYN_MAP):
    """
    Extract HPO terms from text with optional context flag detection.
    
    Args:
        text (str): The input medical text.
        with_context (bool): If True, add context flags (e.g. negation, family).
        id_to_name_map (dict): Mapping from HPO ID to term name.
        id_to_syn_map (dict): Mapping from HPO ID to synonyms.
    
    Returns:
        str: Tab-delimited string of entity data.
    """
    # Remove leading colon if present
    # starts_with_colon = text.startswith(":")
    starts_with_colon = False
    original_text = text[1:] if starts_with_colon else text

    # Parse the medical record once
    subsentences, subsent_to_sentence, mr_words, mr_flags, pos_map = parse_medical_record(original_text, with_context)
    mr_map = load_mr_map(mr_words)

    used_spans = set()
    entity_data = []

    # Iterate over all HPO terms (by id) and their synonyms.
    for hpo_id, syns in id_to_syn_map.items():
        for syn in syns:
            syn_lower = syn.lower()
            syn_tokens = alphanum_only({syn_lower})
            if not syn_tokens:
                continue

            # Identify candidate subsentence indices by intersecting tokens.
            candidate_idxs = set(mr_map.get(next(iter(syn_tokens)), set()))
            for token in syn_tokens:
                candidate_idxs &= mr_map.get(token, set())
            if not candidate_idxs:
                continue

            # Build a set of possible forms (e.g. singular/plural)
            forms = {syn_lower}
            if syn_lower.endswith('s'):
                forms.add(syn_lower[:-1])
            else:
                forms.add(syn_lower + 's')

            # Find all matching spans in the original text.
            spans = []
            for form in forms:
                if " " in form:
                    words = form.split()
                    # Create all possible permutations of the words
                    permutations = list(itertools.permutations(words))
                    # Convert permutations back to strings
                    permuted_terms = [' '.join(perm) for perm in permutations]
                    spans.extend(find_spans(original_text.lower(), permuted_terms))
                else:
                    spans.extend(find_spans(original_text.lower(), [form]))
            if not spans:
                # continue
                spans.extend([(original_text.lower(), 999, 999)])
            spans.sort(key=lambda x: x[1])  # sort by start index

            # For each candidate subsentence, select the best (unused) span.
            for idx in candidate_idxs:
                if idx not in pos_map:
                    continue
                line_pos = pos_map[idx]
                best_span = None
                min_distance = float('inf')
                # Check each span not yet used.
                for term_found, start, end in spans:
                    if (start, end) in used_spans:
                        continue
                    # Verify that the found term exists in the subsentence tokens.
                    subsent_text = subsentences[idx].lower()
                    subsent_tokens = add_lemmas(alphanum_only({subsent_text}))
                    term_tokens = alphanum_only({term_found})
                    if not all(token in subsent_tokens for token in term_tokens):
                        continue
                    distance = abs(start - line_pos)
                    if distance < min_distance:
                        min_distance = distance
                        best_span = (start, end, term_found)
                if not best_span:
                    # continue
                    best_span = (999, 999, term_found)
                used_spans.add((best_span[0], best_span[1]))

                # Build the entity entry.
                entry = {
                    'term_id': hpo_id,
                    'term_name': id_to_name_map.get(hpo_id, ""),
                    'synonym_match': syn,
                    'context': subsent_to_sentence[idx],
                    'start_idx': best_span[0] - (1 if starts_with_colon else 0),
                    'end_idx': best_span[1] - (1 if starts_with_colon else 0)
                }
                if with_context:
                    entry['earliness'] = idx
                    flags = mr_flags[idx] if mr_flags and idx < len(mr_flags) else []
                    if not flags:
                        flags = ['positive']
                    entry['positive'] = 1 if 'positive' in flags else 0
                    for flag in FLAGS:
                        entry[flag] = 1 if flag in flags else 0
                else:
                    entry['matched_text'] = best_span[2]
                entity_data.append(entry)

    # Prepare tab-delimited output.
    if not entity_data:
        if with_context:
            # header = "term_id\tterm_name\tsynonym_match\tcontext\tearliness\tpositive\t" + "\t".join(FLAGS) + "\tstart_idx\tend_idx"
            header = "term_id\tterm_name\tsynonym_match\tcontext\tmatched_text\tstart_idx\tend_idx" + "\t".join(FLAGS)
        else:
            header = "term_id\tterm_name\tsynonym_match\tcontext\tmatched_text\tstart_idx\tend_idx"
        return header

    if with_context:
        # headers = ["term_id", "term_name", "synonym_match", "context", "earliness", "positive"] + list(FLAGS) + ["start_idx", "end_idx"]
        headers = ["term_id", "term_name", "synonym_match", "context", "matched_text", "start_idx", "end_idx"] + list(FLAGS)
    else:
        headers = ["term_id", "term_name", "synonym_match", "context", "matched_text", "start_idx", "end_idx"]

    lines = ["\t".join(headers)]
    for ent in entity_data:
        row = [str(ent.get(col, "")) for col in headers]
        lines.append("\t".join(row))
    return "\n".join(lines)

# --- DataFrame Extraction Wrapper ---

def extract_hpo_terms_df(text, with_context=True, id_to_name_map=ID_TO_NAME_MAP, id_to_syn_map=ID_TO_SYN_MAP):
    """
    Extract HPO terms and return the results as a pandas DataFrame.
    """
    data_str = extract_hpo_terms(text, with_context, id_to_name_map, id_to_syn_map)
    lines = data_str.split('\n')
    if len(lines) <= 1:
        return pd.DataFrame(columns=lines[0].split('\t'))
    headers = lines[0].split('\t')
    rows = [line.split('\t') for line in lines[1:]]
    df = pd.DataFrame(rows, columns=headers)
    # Convert numeric columns if present.
    # numeric_cols = ['earliness', 'positive', 'negative', 'family', 'healthy', 'disease', 'start_idx', 'end_idx']
    numeric_cols = ['earliness', 'negative', 'family', 'healthy', 'disease', 'start_idx', 'end_idx']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df

# --- Backwards Compatibility Aliases ---

extract_phenotypes = lambda text, **kwargs: extract_hpo_terms(text, with_context=True, **kwargs)
extract_phenotypes_df = lambda text, **kwargs: extract_hpo_terms_df(text, with_context=True, **kwargs)
extract_entities = lambda text, **kwargs: extract_hpo_terms(text, with_context=False, **kwargs)
extract_entities_df = lambda text, **kwargs: extract_hpo_terms_df(text, with_context=False, **kwargs)