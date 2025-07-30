import os
import sys
import re
import json
from typing import Callable, Dict, Any
import pandas as pd
import tiktoken

from collections import Counter
from nltk.stem import WordNetLemmatizer

from breastfeeding_nlp.llm.agents import BedrockClient

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from breastfeeding_nlp.utils.utils import StopWatch, load_dataframe
from breastfeeding_nlp.utils.similarity import deduplicate_similar_terms

import nltk
from nltk.corpus import words

# Download the words corpus if not already downloaded
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

class AbbreviationCache(dict):
    """
    Implements a persistent cache for abbreviation expansions.
    This class combines an in-memory dictionary with a JSON file for persistent storage.
    """
    
    def __init__(self, cache_file: str = "abbreviation_cache.json") -> None:
        """
        Initialize the cache by attempting to load from a file, and then add default entries.
        
        Args:
            cache_file (str): Path to the JSON file for persistent caching.
        """
        super().__init__()
        self.cache_file = cache_file
        self.load_cache()
        # filter out non-string keys and any float keys that might be causing problems
        self.cache = {key: value for key, value in self.items() if isinstance(key, str)}
    
    def load_cache(self) -> None:
        """
        Load the cache from a JSON file. If the file doesn't exist, the cache remains empty.
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.update(data)
            except Exception as e:
                print(f"Error loading cache: {e}")
        else:
            self.initialize_cache()
    
    def save_cache(self) -> None:
        """
        Save the in-memory cache to the JSON file.
        """
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error saving cache: {e}")
    

    def initialize_cache(self) -> None:
        """
        Initialize the cache with default abbreviation expansions if not already present.
        """
        # load unambiguous abbreviations from s3
        unambiguous_abbreviations = load_dataframe("s3://nch-ods-data-science/projects/deid-test-data/unambiguous_abbreviations_03122025.csv")
        
        # convert from pd.DataFrame to JSON
        unambiguous_abbreviations = unambiguous_abbreviations.to_dict(orient="records")
        
        # add to cache
        for abbr in unambiguous_abbreviations:
            self[abbr["Abbreviation"]] = abbr["preproc_exp"]
        
        # save the cache locally
        # self.save_cache()

class TextReplacer(dict):
    """A simple class for replacing multiple strings in text at once."""
    
    def replace_all(self, text: str) -> str:
        """
        Replace all occurrences of dictionary keys with their values in the text.
        
        Args:
            text (str): The text to perform replacements on.
            
        Returns:
            str: Text with all replacements applied.
        """
        # Create a regex pattern matching keys as standalone words.
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, self.keys())) + r')\b')
        return pattern.sub(lambda match: self[match.group(0)], text)
    
    # For backward compatibility with the original Xlator
    def xlat(self, text: str) -> str:
        """Legacy method that calls replace_all."""
        return self.replace_all(text)


class AbbreviationExpander:
    """
    Expands medical abbreviations in text using an LLM while employing a hybrid caching system.
    
    The process checks the persistent in-memory cache for known, unambiguous abbreviations.
    Abbreviations that are not in the cache—or those flagged as ambiguous (e.g., "ASD", "MR")—
    trigger a call to the LLM. The results from the LLM are then merged with cached values, and the
    persistent cache is updated for future reference.
    """
    
    def __init__(self, model_id: str = 'haiku') -> None:
        """
        Initialize the expander with a Bedrock client, load the persistent cache, and define
        any ambiguous abbreviations that should always be handled by the LLM.
        """
        if model_id == "haiku":
            model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        elif model_id == "sonnet":
            model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        self.expander = BedrockClient(model_id=model_id)  # default LLM client settings
        self.system_prompt = (
            "You are a medical expert specializing in clinical terminology. Your task is to identify and expand all medical abbreviations and acronyms in the provided text. "
            "Always reference the UMLS (Unified Medical Language System) terminology. Ignore common words that are in all caps in the text as well as "
            "short words that are not UMLS abbreviations. NEVER expand 'in', 'hx', 'cm', 'ile', 'ft', 'lb', 'oz', 'kg'.\n\n"
            "Do not include abbreviations for terms that have already been expanded. For example, if the text contains 'attention deficit hyperactivity disorder', "
            "do not include 'ADHD' in your response.\n\n"
            "Pay particular attention to the following abbreviation candidates: {}\n\n"
            "Return your response as a Python dictionary with abbreviations as keys and their expanded forms as values.\n\n"
            "Do not include any introductory or concluding sentences in your response."
        )
        # Initialize the persistent cache (both in-memory and on disk)
        self.cache = AbbreviationCache() # if no local cache provided, load from s3

        # Clean the cache values
        for abbr, expansion in self.cache.items():
            self.cache[abbr] = expansion[2:-2] # remove leading and trailing quotes and brackets

        # Define ambiguous abbreviations that should always trigger LLM processing.
        self.ambiguous_set = {"ASD", "MR", "BM", 'bm'}  # Extend this set as needed.
        
        # Placeholders to store text and results for the current document.
        self.original_text: str = ""
        self.abbreviation_dict: Dict[str, str] = {}
        self.new_text: str = ""
    
    @staticmethod
    def find_candidate_abbreviations(text: str) -> set:
        """
        Identify candidate abbreviations in the text using a heuristic regex.
        
        The regex looks for tokens 1 to 5 characters long that start with an uppercase letter,
        followed by uppercase letters, numbers, or hyphens/slashes (if followed by uppercase/number).
        The results are filtered to exclude common English words and specified non-medical terms.
        
        Args:
            text (str): The text to search for abbreviations.
            
        Returns:
            set: A set of candidate abbreviation strings that are likely medical abbreviations.
        """
        pattern = re.compile(r'\b(?=.{1,5}\b)[A-Z](?:[A-Z0-9]|[-/](?=[A-Z0-9])){0,4}(?<![-./:;,!?])\b')

        # remove nltk stopwords
        from nltk.corpus import stopwords
        stopwords = set(stopwords.words('english'))

        # Get the English dictionary
        english_words = set(words.words())
        
        # Words to explicitly ignore
        ignore_nonwords = ["DOB", "ICTAL", "EARS", "EYES", "NEUTR", "LUNGS", "SIBS", "NEURO", "CELLS", "MISC", "XRAY", "MED", "STATS"]
        
        # Get all matches from the pattern
        candidates = pattern.findall(text)
        
        # Filter to only include results that are:
        # 1. Not in stopwords
        # 2. Not in the English dictionary
        # 3. Not in the ignore_nonwords list
        return set(word for word in candidates 
                  if word.lower() not in stopwords 
                  and word.lower() not in english_words
                  and word not in ignore_nonwords)
    
    def extract_abbreviation_dict(self, text: str, candidates: set) -> Dict[str, str]:
        """
        Extract a dictionary of abbreviations and their expanded forms using the LLM.
        
        Args:
            text (str): The text to extract abbreviations from.
        
        Returns:
            Dict[str, str]: Abbreviations as keys and their expanded forms as values.
        """
        res = self.expander.invoke_model(
            system_message=self.system_prompt.format(candidates), # tell the LLM which abbreviations to pay particular attention to
            messages=[{"role": "user", "content": text}],
            temperature=0.1,  # Controls randomness.
            top_p=0.4,        # Controls diversity.
            max_tokens=2000
        )
        # Extract the dictionary portion from the response.
        dict_text = (
            res['text']
                [res['text'].find('{\n')+2: res['text'].find('\n}')]
            .strip()
        )
        abbreviation_dict: Dict[str, str] = {}
        for line in dict_text.split('\n'):
            line = line.strip()
            if line.endswith(','):
                line = line[:-1]  # Remove trailing comma.
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip().strip('"\'')
                value = parts[1].strip().strip('"\'')
                abbreviation_dict[key] = value
        
        return abbreviation_dict
    
    def get_cached_abbreviations(self, text: str) -> Dict[str, str]:
        cached_abbrs = {}
        for abbr, expansion in self.cache.items():
            if isinstance(abbr, str) and isinstance(expansion, str):
                if (abbr not in self.ambiguous_set) and (re.search(r'\b' + re.escape(abbr) + r'\b', text)):
                    cached_abbrs[abbr] = expansion
        
        # if there are no cached abbreviations, return a placeholder dictionary
        if not cached_abbrs:
            cached_abbrs = {"":""}
        
        return cached_abbrs
    
    def pre_expand_abbreviations(self, text: str) -> str:
        """
        Pre-expand abbreviations in the given text using the persistent cache.
        """
        self.cached_abbrs = self.get_cached_abbreviations(text)
        return TextReplacer(self.cached_abbrs).replace_all(text)

    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand abbreviations in the given text using a two-step process:
        
        1. Pre-replace unambiguous abbreviations that exist in the persistent cache.
        This step quickly handles known cases and reduces the workload for the LLM.
        
        2. Invoke the LLM only on the resulting text to expand remaining out-of-cache
        abbreviations (including ambiguous ones). The LLM results are then merged with
        the cached ones.
        
        The final replacement is applied to the original text using the merged mapping.
        
        Args:
            text (str): The original text containing abbreviations to expand.
        
        Returns:
            str: The text with all abbreviations expanded.
        """
        self.original_text = text

        # Step 1: Retrieve and apply cached expansions for unambiguous abbreviations.
        # Pre-replace known unambiguous abbreviations in the text.
        pre_replaced_text = self.pre_expand_abbreviations(text)
        self.pre_replaced_text = pre_replaced_text

        # Step 2: Identify remaining candidate abbreviations in the pre-replaced text.
        remaining_candidates = self.find_candidate_abbreviations(pre_replaced_text)

        # If there are any remaining abbreviations (ambiguous or unknown),
        # call the LLM on the pre-replaced text.
        if remaining_candidates:
            llm_abbr_dict = self.extract_abbreviation_dict(pre_replaced_text, remaining_candidates)
            # Update the persistent cache with new, non-ambiguous abbreviations from the LLM results.
            # vv commenting out for now, pending manual review of the new abbreviations.
            # for abbr, expansion in llm_abbr_dict.items():
            #     if abbr not in self.ambiguous_set and abbr not in self.cache:
            #         self.cache[abbr] = expansion
            # Merge cached abbreviations with LLM results.
            final_abbr_dict = {**self.cached_abbrs, **llm_abbr_dict}
        else:
            final_abbr_dict = self.cached_abbrs

        # Save the merged abbreviation dictionary for later analysis.
        self.abbreviation_dict = final_abbr_dict

        # Apply the final replacement to the original text.
        self.new_text = TextReplacer(final_abbr_dict).replace_all(text)

        # Persist the updated cache to disk.
        # self.cache.save_cache()
        # Don't persist the cache to disk until it's been manually reviewed and verified.

        return self.new_text
    
    def analyze_abbreviations(self) -> pd.DataFrame:
        """
        Analyze abbreviations in the original text and return span information in a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing abbreviations, their expansions, and their positions.
        """
        abbreviation_dict = self.abbreviation_dict
        original_text = self.original_text
        
        rows = []
        # Locate every occurrence of each abbreviation.
        for abbrev, expansion in abbreviation_dict.items():
            start_pos = 0
            while True:
                pos = original_text.find(abbrev, start_pos)
                if pos == -1:
                    break
                rows.append({
                    'Abbreviation': abbrev,
                    'Expansion': expansion,
                    'start_idx': pos,
                    'stop_idx': pos + len(abbrev)
                })
                start_pos = pos + 1
        
        abbrev_df = pd.DataFrame(rows)
        if not abbrev_df.empty:
            abbrev_df = abbrev_df.sort_values(['start_idx', 'stop_idx'])
        
        return abbrev_df.reset_index(drop=True)


def expand_abbreviations_in_text(row: pd.Series) -> tuple[float, Dict[str, str], pd.DataFrame, str, int, int]:
    """
    Expand abbreviations in a text note and return details including processing time, token counts,
    and analysis of abbreviation spans.
    
    Args:
        row (pd.Series): A DataFrame row containing a 'note_text' field.
        
    Returns:
        tuple: Contains processing time, abbreviation dictionary, abbreviation span DataFrame,
               expanded text, input token count, and output token count.
    """
    AAE = AbbreviationExpander()
    stopwatch = StopWatch()
    encoding = tiktoken.get_encoding("cl100k_base")

    stopwatch.start()
    AAE.expand_abbreviations(row['note_text'])
    stopwatch.stop()

    input_tokens = len(encoding.encode(AAE.original_text)) + len(encoding.encode(AAE.system_prompt))
    output_tokens = len(encoding.encode(str(AAE.abbreviation_dict)))

    return (
        stopwatch.elapsed_time(),
        AAE.abbreviation_dict,
        AAE.analyze_abbreviations(),
        AAE.new_text,
        input_tokens,
        output_tokens
    )


def process_dataframe(
    df: pd.DataFrame, 
    expand_function: Callable, 
    input_cost_per_million: float = 3, 
    output_cost_per_million: float = 15
) -> pd.DataFrame:
    """
    Process a DataFrame by applying the abbreviation expansion function to each row and
    calculating token costs.
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
        expand_function (Callable): Function to apply to each row.
        input_cost_per_million (float, optional): Cost per million input tokens. Defaults to 3.
        output_cost_per_million (float, optional): Cost per million output tokens. Defaults to 15.
        
    Returns:
        pd.DataFrame: Updated DataFrame with additional columns for processing metrics and costs.
    """
    expansion_results = df.apply(expand_function, axis=1)
    
    elapsed_times = [result[0] for result in expansion_results]
    abbreviation_dicts = [result[1] for result in expansion_results]
    abbreviation_df_list = [result[2] for result in expansion_results]
    expanded_texts = [result[3] for result in expansion_results]
    input_tokens = [result[4] for result in expansion_results]
    output_tokens = [result[5] for result in expansion_results]
    
    df['processing_time'] = elapsed_times
    df['abbreviation_dict'] = abbreviation_dicts
    df['expanded_text'] = expanded_texts
    df['abbreviation_df'] = abbreviation_df_list
    df['input_tokens'] = input_tokens
    df['output_tokens'] = output_tokens
    
    df['input_cost'] = df['input_tokens'] * (input_cost_per_million / 1000000)
    df['output_cost'] = df['output_tokens'] * (output_cost_per_million / 1000000)
    df['total_cost'] = df['input_cost'] + df['output_cost']
    
    return df


def preproc(expansions: list[str], similarity_threshold: float = 0.7) -> list[str]:
    """
    Preprocess a list of expansion terms by cleaning, lemmatizing, and deduplicating them.
    
    Args:
        expansions (list[str]): List of expansion terms to preprocess.
        similarity_threshold (float, optional): Threshold for considering terms as duplicates.
                                               Defaults to 0.7.
    
    Returns:
        list[str]: Deduplicated list of preprocessed expansion terms.
    """
    LEMMATIZER = WordNetLemmatizer()
    # Remove non-alphanumeric characters (except spaces).
    expansions = [re.sub(r'[^a-zA-Z0-9\s]', '', exp) for exp in expansions]
    expansions = [exp.lower() for exp in expansions]
    expansions = [LEMMATIZER.lemmatize(exp) for exp in expansions]
    expansions = deduplicate_similar_terms(expansions, threshold=similarity_threshold)
    
    return list(set(expansions))