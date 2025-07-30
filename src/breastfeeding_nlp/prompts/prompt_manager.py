from typing import Dict
from .document_classification import document_classification_prompt

class PromptManager:
    """
    A class to manage and provide access to all prompts used in the breastfeeding NLP system.
    
    This class centralizes prompt storage and access, making it easier to maintain and update
    prompts as needed.
    """
    
    def __init__(self):
        """Initialize the PromptManager with all available prompts."""
#         # Typo resolution prompt
#         self.typo_resolution_prompt = """
# Correct ONLY spelling errors in the provided text. DO NOT modify or expand any abbreviations or acronyms.

# - Identify and correct misspelled words only.
# - ALL abbreviations and acronyms MUST remain unchanged, regardless of their form.
# - Medical terms like "ebm" (expressed breast milk) are abbreviations and should NOT be modified.

# # Critical Instructions

# 1. NEVER expand abbreviations or acronyms, even if you recognize them.
# 2. Only correct obvious spelling errors of regular words.
# 3. When in doubt, leave the word as is.
# 4. Treat any capitalized words, all-caps words, or short terms as potential abbreviations and DO NOT modify them.

# # Output Format

# Return your response as a Python dictionary with misspelled words as keys and their corrected forms as values. Do not include any introductory or concluding sentences in your response.

# # Examples

# **Input**:
# "The patient presented with an abnormalety in the braest area and reports feeding her child ebm via bottle."

# **Output**:
# {
#     "abnormalety": "abnormality",
#     "braest": "breast"
# }

# Note that "ebm" was preserved as an abbreviation and not expanded or modified.

# # Additional Example

# **Input**:
# "Pt with hx of HTN complaned of hedache and dizzness."

# **Output**:
# {
#     "complaned": "complained",
#     "hedache": "headache",
#     "dizzness": "dizziness"
# }

# Note that "Pt", "hx", and "HTN" were all preserved as abbreviations and not expanded.
# """.strip()

#         # Nursing disambiguation prompt
#         self.nursing_disambiguation_prompt = """
# Identify if the usage of the word "nursing" in a given sentence is related to breastfeeding. If it is, append "_bm" to "nursing" and return the result as a Python dictionary.

# # Steps

# 1. Analyze the context in which "nursing" appears to determine if it relates to breastfeeding.
# 2. If it is related to breastfeeding, append "_bm" to the word "nursing".
# 3. Construct a Python dictionary where the key is "nursing" and the value is the modified word if applicable, or an empty dictionary if not related.

# # Output Format

# - The response should be a Python dictionary.
# - If "nursing" is related to breastfeeding, format: `{"nursing": "nursing_bm"}`
# - If it is not related, output: `{}`
# - Do not include any introductory or concluding sentences in your response. Return ONLY the dictionary and nothing else.

# # Examples

# **Example 1:**
# - Input: "He is a nursing student."
# - Output: `{}`

# **Example 2:**
# - Input: "She lives in a nursing home."
# - Output: `{}`

# **Example 3:**
# - Input: "The patient has trouble latching/nursing."
# - Output: `{"nursing": "nursing_bm"}` 

# **Example 4:**
# - Input: "Mother reports nursing every 2-3 hours."
# - Output: `{"nursing": "nursing_bm"}`

# **Example 5:**
# - Input: "The nursing staff administered medication."
# - Output: `{}`

# # Notes

# - Consider the context in which "nursing" is used to make the determination.
# - Medical contexts may use "nursing" for both healthcare profession and breastfeeding - carefully analyze surrounding words.
# - Words like "latch," "feed," "breast," "milk," or references to infants often indicate breastfeeding context.
# - Ensure to output an empty dictionary if "nursing" does not relate to breastfeeding.
# - The output must be a valid Python dictionary with no additional text.
# - NEVER output the reasoning, ONLY return the output dictionary.
# """.strip()

        # Disambiguate prompt
        self.disambiguate_prompt = """
Identify and extract information regarding infant feeding from a clinical note, specifically focusing on instances of breast feeding and formula feeding. For ambiguous mentions, use context to determine the type of feeding. Extract details about the amount consumed and feeding frequency.

# Steps

1. **Identify Feeding Mention**: Scan the text for any mention of feeding methods, specifically "breast feeding" and "formula feeding".
   
2. **Resolve Ambiguities**: For ambiguous terms like "bottle feeding", analyze surrounding context to classify it as breast milk, formula, or other. 
   
3. **Extract Details**:
   - **Amount Consumed**: Identify and extract any numerical data or measurement units describing the volume of milk consumed.
   - **Feeding Frequency**: Determine the frequency at which the feedings occur.

# Output Format

The extracted information should be provided in JSON format with the following keys:
- `feeding_type`: "breast", "formula", or determined type for ambiguous cases.
- `amount`: extracted amount of milk consumed.
- `frequency`: extracted feeding frequency.

# Examples

**Example 1:**

*Input*: 
"The infant was mostly bottle feeding with breast milk. Intake was about 90 ml every 3 hours."

*Output*:
```json
{
  "feeding_type": "breast",
  "amount": "90 ml",
  "frequency": "every 3 hours"
}
```

**Example 2:**

*Input*: 
"The baby is given formula twice a day."

*Output*:
```json
{
  "feeding_type": "formula",
  "amount": null,
  "frequency": "twice a day"
}
```

**Example 3:**

*Input*: 
"The infant receives bottle feeding extensively."

*Reasoning*: Without further context indicating the type of milk, the feeding type remains unresolved. However, if context suggests the content of the bottle, adjust accordingly.

*Output*:
```json
{
  "feeding_type": "unknown",
  "amount": null,
  "frequency": null
}
```

# Notes

- Ensure to interpret the context surrounding ambiguous terms correctly to deduce the appropriate feeding type.
- If context is insufficient to clarify an ambiguous term, denote feeding type as "unknown".
- Return ONLY the JSON output, nothing else.
""".strip()

    def get_prompt(self, prompt_name: str) -> str:
        """
        Get a prompt by name.
        
        Args:
            prompt_name: The name of the prompt to retrieve
            
        Returns:
            The requested prompt text
            
        Raises:
            AttributeError: If the prompt does not exist
        """
        if hasattr(self, prompt_name):
            return getattr(self, prompt_name)
        raise AttributeError(f"Prompt '{prompt_name}' does not exist in PromptManager")
    
    def get_all_prompts(self) -> Dict[str, str]:
        """
        Get a dictionary of all available prompts.
        
        Returns:
            Dictionary mapping prompt names to their text content
        """
        prompts = {}
        for attr_name in dir(self):
            # Skip private/special attributes and methods
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                prompts[attr_name] = getattr(self, attr_name)
        return prompts 