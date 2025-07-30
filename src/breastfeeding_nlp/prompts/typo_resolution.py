typo_resolution_prompt = """
Correct ONLY spelling errors in the provided text. DO NOT modify or expand any abbreviations or acronyms.

- Identify and correct misspelled words only.
- ALL abbreviations and acronyms MUST remain unchanged, regardless of their form.
- Medical terms like "ebm" (expressed breast milk) are abbreviations and should NOT be modified.

# Critical Instructions

1. NEVER expand abbreviations or acronyms, even if you recognize them.
2. Only correct obvious spelling errors of regular words.
3. When in doubt, leave the word as is.
4. Treat any capitalized words, all-caps words, or short terms as potential abbreviations and DO NOT modify them.

# Output Format

Return your response as a Python dictionary with misspelled words as keys and their corrected forms as values. Do not include any introductory or concluding sentences in your response.

# Examples

**Input**:
"The patient presented with an abnormalety in the braest area and reports feeding her child ebm via bottle."

**Output**:
{
    "abnormalety": "abnormality",
    "braest": "breast"
}

Note that "ebm" was preserved as an abbreviation and not expanded or modified.

# Additional Example

**Input**:
"Pt with hx of HTN complaned of hedache and dizzness."

**Output**:
{
    "complaned": "complained",
    "hedache": "headache",
    "dizzness": "dizziness"
}

Note that "Pt", "hx", and "HTN" were all preserved as abbreviations and not expanded.
""".strip()