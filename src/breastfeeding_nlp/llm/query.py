from breastfeeding_nlp.llm.agents import BedrockClient

def query_llm(system_prompt: str, text: str) -> str:
    agent = BedrockClient() # takes a model_id parameter. Make this a parameter with either 'haiku' or 'sonnet' as options. Default to sonnet.
    res = agent.invoke_model(
            system_message=system_prompt,
            messages=[{"role": "user", "content": text}],
            temperature=0.1,
            top_p=0.4,
        )
    return res