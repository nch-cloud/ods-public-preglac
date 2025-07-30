disambiguate_prompt = """
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