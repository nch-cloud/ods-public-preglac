import os
import json
import boto3
from typing import Optional, Dict, List, Any
from botocore.exceptions import ClientError

class BedrockClient:
    """Handles all interactions with AWS Bedrock."""
    def __init__(self, model_id=None, region=None):
        self.model_id = model_id or os.environ.get('MODEL_ID', "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        self.region = region or os.environ.get('REGION_NAME', "us-east-2")
        self.bedrock = boto3.client("bedrock-runtime", region_name=self.region)

    def invoke_model(
        self, 
        messages: List[Dict[str, str]], 
        system_message: Optional[str] = None, 
        max_tokens: int = 256,
        temperature: float = 0.3, # controls randomness of the output
        top_p: float = 0.6 # controls diversity of the output
    ) -> Optional[Dict[str, Any]]:
        """Generic method to invoke Bedrock model.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries containing 'role' and 'content' keys
            system_message (Optional[str]): Optional system message to include in the request
            max_tokens (int): Maximum number of tokens in the response. Defaults to 256.
        
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing:
                - 'text': The generated response text
                - 'input_tokens': Number of tokens in the input
                - 'output_tokens': Number of tokens in the output
                Returns None if an error occurs during the API call.
        
        Raises:
            ClientError: If there is an error with the AWS Bedrock API call
            Exception: For any other unexpected errors
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": messages
        }
        if system_message:
            body["system"] = system_message

        try:
            response = self.bedrock.invoke_model(
                body=json.dumps(body), 
                modelId=self.model_id
            )
            response_body = json.loads(response.get("body").read())
            
            return {
                'text': response_body.get("content")[0].get("text"),
                'input_tokens': response_body.get('usage', {}).get('input_tokens'),
                'output_tokens': response_body.get('usage', {}).get('output_tokens')
            }
        except ClientError as e:
            print(f"AWS Bedrock Error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


class CorrectionAgent:
    """
    A general purpose class for using Claude to look at text and return a dictionary of corrections.

    This can be used for a variety of tasks, including:
    - Resolving typos in text.
    - Disambiguating text.
    - Expanding abbreviations and acronyms.
    - And more! 

    Args:
        system_prompt (str): The system prompt to use for the agent.
    
    Returns:
        Dict[str, str]: A dictionary of corrections.
    """
    def __init__(self, system_prompt: str):
        self.agent = BedrockClient()
        self.system_prompt = system_prompt

    def get_corrections(self, text: str) -> Dict[str, str]:
        res = self.agent.invoke_model(
            system_message=self.system_prompt,
            messages=[{"role": "user", "content": text}],
            temperature=0.1,
            top_p=0.4,
        )
        # Extract the dictionary portion from the response.
        dict_text = (
            res['text']
                [res['text'].find('{\n')+2: res['text'].find('\n}')]
            .strip()
        )
        
        corrections: Dict[str, str] = {}
        for line in dict_text.split('\n'):
            line = line.strip()
            if line.endswith(','):
                line = line[:-1]  # Remove trailing comma.
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip().strip('"\'')
                value = parts[1].strip().strip('"\'')
                corrections[key] = value
        
        return corrections