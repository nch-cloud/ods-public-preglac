nursing_disambiguation_prompt = """
Identify if the usage of the word "nursing" in a given sentence is related to breastfeeding. If it is, append "_bm" to "nursing" and return the result as a Python dictionary.

# Steps

1. Analyze the context in which "nursing" appears to determine if it relates to breastfeeding.
2. If it is related to breastfeeding, append "_bm" to the word "nursing".
3. Construct a Python dictionary where the key is "nursing" and the value is the modified word if applicable, or an empty dictionary if not related.

# Output Format

- The response should be a Python dictionary.
- If "nursing" is related to breastfeeding, format: `{"nursing": "nursing_bm"}`
- If it is not related, output: `{}`
- Do not include any introductory or concluding sentences in your response. Return ONLY the dictionary and nothing else.

# Examples

**Example 1:**
- Input: "He is a nursing student."
- Output: `{}`

**Example 2:**
- Input: "She lives in a nursing home."
- Output: `{}`

**Example 3:**
- Input: "The patient has trouble latching/nursing."
- Output: `{"nursing": "nursing_bm"}` 

**Example 4:**
- Input: "Mother reports nursing every 2-3 hours."
- Output: `{"nursing": "nursing_bm"}`

**Example 5:**
- Input: "The nursing staff administered medication."
- Output: `{}`

# Notes

- Consider the context in which "nursing" is used to make the determination.
- Medical contexts may use "nursing" for both healthcare profession and breastfeeding - carefully analyze surrounding words.
- Words like "latch," "feed," "breast," "milk," or references to infants often indicate breastfeeding context.
- Ensure to output an empty dictionary if "nursing" does not relate to breastfeeding.
- The output must be a valid Python dictionary with no additional text.
- NEVER output the reasoning, ONLY return the output dictionary.
""".strip()


# - Reasoning: "nursing" in this context refers to studying nursing as a profession.
# - Reasoning: "nursing" refers to a care facility, not breastfeeding.
# - Reasoning: "nursing" refers to breastfeeding difficulties.
# - Reasoning: "nursing" refers to breastfeeding activity.
# - Reasoning: "nursing" refers to healthcare professionals, not breastfeeding.