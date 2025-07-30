document_classification_prompt_v1 = """
Evaluate clinical notes to determine documented evidence of breastfeeding and assign the appropriate label.

Consider the following categories when reviewing the notes:

- **Exclusive breastfeeding**: Evidence shows only breastfeeding is documented.
- **Exclusive formula feeding**: Evidence shows only formula feeding is documented.
- **Mixed**: Evidence shows both breastfeeding and formula feeding are documented.
- **None**: No evidence of breastfeeding or formula feeding is documented.
- **Undetermined**: The notes are not clear about the type of feeding.

# Steps

1. **Read the Clinical Notes**: Carefully review the given clinical notes for any mentions of feeding types.
2. **Identify Evidence**: Look for specific mentions or terms that indicate breastfeeding, formula feeding, or both.
3. **Determine the Label**: Based on the identified evidence, classify the notes with one of the four labels.

# Output Format

- A JSON object with the following fields:
  - `Label`: The assigned label.
  - `Reasoning`: A brief, one sentence explanation for the assigned label.

# Examples

**Example 1**

- **Input**: "The infant was fed breast milk exclusively during the stay."
- **Reasoning**: The notes mention exclusive breastfeeding without any reference to formula.
- **Output**: Exclusive breastfeeding

**Example 2**

- **Input**: "The infant was given formula in addition to breastfeeding."
- **Reasoning**: The notes indicate both breastfeeding and formula feeding.
- **Output**: Mixed

**Example 3**

- **Input**: "The father reports the last feed was 2 hours ago via bottle."
- **Reasoning**: The notes mention feeding via bottle but does not clarify if it was expressed breast milk or formula.
- **Output**: Undetermined

# Notes

- Pay close attention to the language used in the notes — synonyms or indirect mentions may indicate feeding types.
- Assume common medical terminologies related to feeding practices are understood.
- Only output the JSON object, nothing else.
""".strip()

document_classification_prompt_v2 = """
Evaluate clinical notes to decide whether they contain **documented evidence of actual feeding behavior** involving breast milk, and assign one of three labels.

### Labels

| Label | Definition | Typical Triggers | Typical Non-Triggers |
|-------|------------|------------------|----------------------|
| **Breastfeeding Mentioned** | The note explicitly documents that the infant/patient **received breast milk** (directly at breast or via expressed milk). Mentions of both breast milk **and** formula still belong here. | “Breastfeeds every 3 h”, “Takes 30 mL expressed breast milk by bottle”, “Review of nutrition: breast and bottle” | — |
| **No Breastfeeding Mentioned** | The note contains **no evidence** that the infant/patient received breast milk. This includes: only formula feeding, hypothetical statements, education, recommendations, or best-practice guidance. | “Feeds Enfamil 60 mL q3h”, “Continue formula ad lib”, “Nutrition: breastfeeding is highly recommended” | Any direct documentation of breast milk received |
| **Undetermined** | Use **sparingly**. Feeding is clearly documented, but it is impossible to tell whether the milk was breast or formula. | “Last feed 2 h ago via bottle” (no milk type specified) | Ambiguous wording **when breast is explicitly mentioned** (“breast and bottle” → *Breastfeeding Mentioned*) |

### Context Rules

1. **Actual behaviour vs. guidance**  
   Recommendations, anticipatory guidance, or best-practice statements **do not count** as evidence of what was actually fed.  

2. **Nutrition-specific sections** often contain the relevant evidence. Pay extra attention to headings such as **“Review of Nutrition”, “Current Feeding”, “Feeding Difficulties”.**

3. **Context matters**  
   - If *breast milk* is mentioned as **part of current or past feeding behaviour**, choose **Breastfeeding Mentioned** (even if formula is also present).  
   - If *breast milk* is mentioned only as a **recommendation, plan, or intended future behaviour**, choose **No Breastfeeding Mentioned**.  

### Steps

1. **Read the clinical note** thoroughly.  
2. **Identify evidence** of current or past feeding behaviour.  
3. **Apply the context rules** to exclude purely hypothetical or advisory text.  
4. **Assign the appropriate label**.  
5. **Output** a JSON object.

### Output Format

```json
{
  "Label": "<Breastfeeding Mentioned | No Breastfeeding Mentioned | Undetermined>",
  "Reasoning": "<one concise sentence>"
}
```

### Examples

| **Input** | **Expected JSON Output** | **Explanation** |
|-----------|--------------------------|------------------|
| *“The infant was fed breast milk exclusively during the stay.”* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Exclusive breast-milk feeding documented."}` | Direct evidence of breast milk as actual feeding behavior. |
| *“The infant was given formula in addition to breastfeeding.”* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Both formula and breast milk documented; breast milk present."}` | Breast milk is part of current feeding behavior. |
| *“Last feed 2 h ago via bottle.”* | `{"Label":"Undetermined","Reasoning":"Feeding documented but milk type not specified."}` | The feeding event is documented, but it's unclear whether it was breast milk or formula. |
| *“Nutrition: breastfeeding is highly recommended.”* | `{"Label":"No Breastfeeding Mentioned","Reasoning":"Statement is guidance, not evidence of actual feeding."}` | This is a best-practice recommendation, not documentation of feeding. |
| *“Review of nutrition: breast and bottle.”* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Explicit mention of breast feeding behaviour."}` | "Breast" indicates breastfeeding was part of actual feeding behavior, regardless of the ambiguity in “bottle.” |
| *"The mom pumped breast milk this morning."* | `{"Label":"Undetermined","Reasoning":"The note does not mention the infant receiving breast milk."}` | "While the mom pumped this morning, it is unclear whether the milk was given to the baby or dumped." |

> **Note**: Only output the JSON object, and nothing else.
""".strip()

document_classification_prompt = """
Evaluate clinical notes to decide whether they contain **documented evidence of actual feeding behavior** involving breast milk, and assign one of four labels.

### Labels

| Label | Definition | Typical Triggers | Typical Non-Triggers |
|-------|------------|------------------|----------------------|
| **Breastfeeding Mentioned** | The note explicitly documents that the infant/patient **received breast milk** (directly at breast or via expressed milk). Mentions of both breast milk **and** formula still belong here. | "Breastfeeds every 3 h", "Takes 30 mL expressed breast milk by bottle", "Review of nutrition: breast and bottle" | — |
| **Exclusive Formula Mentioned** | The note explicitly documents that the infant/patient received **only formula** with no evidence of breast milk consumption. | "Feeds Enfamil 60 mL q3h", "Continue formula ad lib", "Exclusively formula-fed" | Any mention of breast milk being received |
| **Undetermined** | Use when feeding is clearly documented, but it is impossible to tell whether the milk was breast or formula. | "Last feed 2 h ago via bottle" (no milk type specified) | Ambiguous wording **when breast is explicitly mentioned** ("breast and bottle" → *Breastfeeding Mentioned*) |
| **None** | No feeding information is documented in the note at all. This includes hypothetical statements, education, recommendations, or best-practice guidance without actual feeding documentation. | "Nutrition: breastfeeding is highly recommended", "Discussed feeding options with parents" | Any documentation of actual feeding behavior |

### Context Rules

1. **Actual behaviour vs. guidance**  
   Recommendations, anticipatory guidance, or best-practice statements **do not count** as evidence of what was actually fed.  

2. **Nutrition-specific sections** often contain the relevant evidence. Pay extra attention to headings such as **“Review of Nutrition”, “Current Feeding”, “Feeding Difficulties”.**

3. **Context matters**  
   - If *breast milk* is mentioned as **part of current or past feeding behaviour**, choose **Breastfeeding Mentioned** (even if formula is also present).  
   - If *breast milk* is mentioned as part of **past feeding behavior**, explicitly mention "historical" in the reasoning.
   - If *breast milk* is mentioned only as a **recommendation, plan, or intended future behaviour**, choose **None**.  

### Steps

1. **Read the clinical note** thoroughly.  
2. **Identify evidence** of current or past feeding behaviour.  
3. **Apply the context rules** to exclude purely hypothetical or advisory text.  
4. **Assign the appropriate label**.  
5. **Output** a JSON object.

### Output Format

```json
{
  "Label": "<Breastfeeding Mentioned | No Breastfeeding Mentioned | Undetermined>",
  "Reasoning": "<one concise sentence>"
}
```

### Examples

| **Input** | **Expected JSON Output** | **Explanation** |
|-----------|--------------------------|------------------|
| *“The infant was fed breast milk exclusively during the stay.”* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Exclusive breast-milk feeding documented."}` | Direct evidence of breast milk as actual feeding behavior. |
| *“The infant was given formula in addition to breastfeeding.”* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Both formula and breast milk documented; breast milk present."}` | Breast milk is part of current feeding behavior. |
| *“Last feed 2 h ago via bottle.”* | `{"Label":"Undetermined","Reasoning":"Feeding documented but milk type not specified."}` | The feeding event is documented, but it's unclear whether it was breast milk or formula. |
| *“Nutrition: breastfeeding is highly recommended.”* | `{"Label":"No Breastfeeding Mentioned","Reasoning":"Statement is guidance, not evidence of actual feeding."}` | This is a best-practice recommendation, not documentation of feeding. |
| *“Review of nutrition: breast and bottle.”* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Explicit mention of breast feeding behaviour."}` | "Breast" indicates breastfeeding was part of actual feeding behavior, regardless of the ambiguity in “bottle.” |
| *"The mom pumped breast milk this morning."* | `{"Label":"Undetermined","Reasoning":"The note does not mention the infant receiving breast milk."}` | "While the mom pumped this morning, it is unclear whether the milk was given to the baby or dumped." |
| *"The infant was discharged on breast milk."* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Historical breast milk consumption documented."}` | Breast milk is part of past feeding behavior. |

> **Note**: Only output the JSON object, and nothing else.
""".strip()

document_classification_prompt_haiku_v1 = """
Evaluate clinical notes to decide whether they contain **documented evidence of actual feeding behavior** involving breast milk, and assign one of four labels.

### Labels

| Label | Definition | Typical Triggers | Typical Non-Triggers |
|-------|------------|------------------|----------------------|
| **Breastfeeding Mentioned** | The note explicitly documents that the infant/patient **received breast milk** (directly at breast or via expressed milk). Mentions of both breast milk **and** formula still belong here. | "Breastfeeds every 3 h", "Takes 30 mL expressed breast milk by bottle", "Review of nutrition: breast and bottle" | — |
| **Exclusive Formula Mentioned** | The note explicitly documents that the infant/patient received **only formula** with no evidence of breast milk consumption. | "Feeds Enfamil 60 mL q3h", "Continue formula ad lib", "Exclusively formula-fed" | Any mention of breast milk being received |
| **Undetermined** | Use when feeding is clearly documented, but it is impossible to tell whether the milk was breast or formula. | "Last feed 2 h ago via bottle" (no milk type specified) | Ambiguous wording **when breast is explicitly mentioned** ("breast and bottle" → *Breastfeeding Mentioned*) |
| **None** | No feeding information is documented in the note at all. This includes hypothetical statements, education, recommendations, or best-practice guidance without actual feeding documentation. | "Nutrition: breastfeeding is highly recommended", "Discussed feeding options with parents" | Any documentation of actual feeding behavior |

### Context Rules

1. **Actual behaviour vs. guidance**  
   Recommendations, anticipatory guidance, or best-practice statements **do not count** as evidence of what was actually fed.  

2. **Nutrition-specific sections** often contain the relevant evidence. Pay extra attention to headings such as **“Review of Nutrition”, “Current Feeding”, “Feeding Difficulties”.**

3. **Context matters**  
   - If *breast milk* is mentioned as **part of current or past feeding behaviour**, choose **Breastfeeding Mentioned** (even if formula is also present).  
   - If *breast milk* is mentioned as part of **past feeding behavior**, explicitly mention "historical" in the reasoning.
   - If *breast milk* is mentioned only as a **recommendation, plan, or intended future behaviour**, choose **None**.  

### Steps

1. **Read the clinical note** thoroughly.  
2. **Identify evidence** of current or past feeding behaviour.  
3. **Apply the context rules** to exclude purely hypothetical or advisory text.  
4. **Assign the appropriate label**.  
5. **Output** a JSON object.

### Output Format

```json
{
  "Label": "<Breastfeeding Mentioned | No Breastfeeding Mentioned | Undetermined>",
  "Reasoning": "<one concise sentence>"
}
```

### Examples

| **Input** | **Expected JSON Output** | **Explanation** |
|-----------|--------------------------|------------------|
| *“The infant was fed breast milk exclusively during the stay.”* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Exclusive breast-milk feeding documented."}` | Direct evidence of breast milk as actual feeding behavior. |
| *“The infant was given formula in addition to breastfeeding.”* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Both formula and breast milk documented; breast milk present."}` | Breast milk is part of current feeding behavior. |
| *“Last feed 2 h ago via bottle.”* | `{"Label":"Undetermined","Reasoning":"Feeding documented but milk type not specified."}` | The feeding event is documented, but it's unclear whether it was breast milk or formula. |
| *“Nutrition: breastfeeding is highly recommended.”* | `{"Label":"No Breastfeeding Mentioned","Reasoning":"Statement is guidance, not evidence of actual feeding."}` | This is a best-practice recommendation, not documentation of feeding. |
| *“Review of nutrition: breast and bottle.”* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Explicit mention of breast feeding behaviour."}` | "Breast" indicates breastfeeding was part of actual feeding behavior, regardless of the ambiguity in “bottle.” |
| *"The mom pumped breast milk this morning."* | `{"Label":"Undetermined","Reasoning":"The note does not mention the infant receiving breast milk."}` | "While the mom pumped this morning, it is unclear whether the milk was given to the baby or dumped." |
| *"The infant was discharged on breast milk."* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Historical breast milk consumption documented."}` | Breast milk is part of past feeding behavior. |

> **Note**: Only output the JSON object, and nothing else. Do not include any other text in your response.
""".strip()

document_classification_prompt_haiku = """
Evaluate clinical notes to decide whether they contain **documented evidence of actual feeding behavior** involving breast milk, and assign one of four labels.

### Labels

| Label | Definition | Typical Triggers | Typical Non-Triggers |
|-------|------------|------------------|----------------------|
| **Breastfeeding Mentioned** | The note explicitly documents that the infant/patient **received breast milk** (directly at breast or via expressed milk). Mentions of both breast milk **and** formula still belong here. | "Breastfeeds every 3 h", "Takes 30 mL expressed breast milk by bottle", "Review of nutrition: breast and bottle" | — |
| **Exclusive Formula Mentioned** | The note explicitly documents that the infant/patient received **only formula** with no evidence of breast milk consumption. | "Feeds Enfamil 60 mL q3h", "Continue formula ad lib", "Exclusively formula-fed" | Any mention of breast milk being received |
| **Undetermined** | Use when feeding is clearly documented, but it is impossible to tell whether the milk was breast or formula. | "Last feed 2 h ago via bottle" (no milk type specified) | Ambiguous wording **when breast is explicitly mentioned** ("breast and bottle" → *Breastfeeding Mentioned*) |
| **No Breastfeeding Mentioned** | No feeding information is documented in the note at all. This includes hypothetical statements, education, recommendations, or best-practice guidance without actual feeding documentation. | "Nutrition: breastfeeding is highly recommended", "Discussed feeding options with parents" | Any documentation of actual feeding behavior |

### Context Rules

1. **Actual behaviour vs. guidance**  
   Recommendations, anticipatory guidance, or best-practice statements **do not count** as evidence of what was actually fed.  

2. **Nutrition-specific sections** often contain the relevant evidence. Pay extra attention to headings such as **“Review of Nutrition”, “Current Feeding”, “Feeding Difficulties”.**

3. **Context matters**  
   - If *breast milk* is mentioned as **part of current or past feeding behaviour**, choose **Breastfeeding Mentioned** (even if formula is also present).  
   - If *breast milk* is mentioned only as a **recommendation, plan, or intended future behaviour**, choose **No Breastfeeding Mentioned**.  

4. **Temporality**
   - If the note mentions a history of one type of feeding and a current type of feeding, base your decision on the current type of feeding, but mention the history in the reasoning.
   - If *breast milk* is mentioned as part of **past feeding behavior**, explicitly mention "historical" in the reasoning.

### Steps

1. **Read the clinical note** thoroughly.  
2. **Identify evidence** of current feeding behaviour.  
3. **Apply the context rules** to exclude purely hypothetical or advisory text.  
4. **Assign the appropriate label**.  
5. **Output** a JSON object.

### Output Format

```json
{
  "Label": "<Breastfeeding Mentioned | Exclusive Formula Mentioned | Undetermined | No Breastfeeding Mentioned>",
  "Reasoning": "<one concise sentence>"
}
```

### Examples

| **Input** | **Expected JSON Output** | **Explanation** |
|-----------|--------------------------|------------------|
| *“The infant was fed breast milk exclusively during the stay.”* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Exclusive breast-milk feeding documented."}` | Direct evidence of breast milk as actual feeding behavior. |
| *“The infant was given formula in addition to breastfeeding.”* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Both formula and breast milk documented; breast milk present."}` | Breast milk is part of current feeding behavior. |
| *“Last feed 2 h ago via bottle.”* | `{"Label":"Undetermined","Reasoning":"Feeding documented but milk type not specified."}` | The feeding event is documented, but it's unclear whether it was breast milk or formula. |
| *“Nutrition: breastfeeding is highly recommended.”* | `{"Label":"No Breastfeeding Mentioned","Reasoning":"Statement is guidance, not evidence of actual feeding."}` | This is a best-practice recommendation, not documentation of feeding. |
| *“Review of nutrition: breast and bottle.”* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Explicit mention of breast feeding behaviour."}` | "Breast" indicates breastfeeding was part of actual feeding behavior, regardless of the ambiguity in “bottle.” |
| *"The mom pumped breast milk this morning."* | `{"Label":"Undetermined","Reasoning":"The note does not mention the infant receiving breast milk."}` | "While the mom pumped this morning, it is unclear whether the milk was given to the baby or dumped." |
| *"The infant was discharged on breast milk."* | `{"Label":"Breastfeeding Mentioned","Reasoning":"Historical breast milk consumption documented."}` | Breast milk is part of past feeding behavior. |
| *"The patient was breastfed for 2 months and then switched to formula."* | `{"Label":"Exclusive Formula Mentioned","Reasoning":"Breastfeeding history documented, but only currently on formula."}` | Breastfeeding history is documented, but patient is currently on formula. |

> **Note**: Only output the JSON object, and nothing else. Do not include any other text in your response.
""".strip()