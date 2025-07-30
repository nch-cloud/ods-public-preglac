import json
import time
from typing import Dict
from breastfeeding_nlp.llm.agents import BedrockClient

class LLMInterface:
    """
    A class that provides an interface for interacting with Large Language Models.
    
    This class handles:
    - Querying LLMs with appropriate prompts
    - Parsing structured JSON responses
    - Tracking token usage and calculating costs
    - Measuring query response time
    
    Attributes:
        client: The LLM client used for model invocation
        input_tokens: Number of tokens in the most recent input
        output_tokens: Number of tokens in the most recent output
        total_tokens: Total tokens used in the most recent query
        input_cost: Cost of the most recent input in USD
        output_cost: Cost of the most recent output in USD
        total_cost: Total cost of the most recent query in USD
        query_time: Time taken for the most recent query in seconds
    """
    
    def __init__(self, model_name: str):
        if model_name.lower() == 'sonnet':
            self.model_id = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0' # Why 3.5 and not 3.7?
            # self.model_id = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0' # 3.7 was REALLY bad
        elif model_name.lower() == 'haiku':
            self.model_id = 'us.anthropic.claude-3-5-haiku-20241022-v1:0'
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        self.client = BedrockClient(model_id=self.model_id)
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_tokens: int = 0
        self.input_cost: float = 0.0
        self.output_cost: float = 0.0
        self.total_cost: float = 0.0
        self.query_time: float = 0.0

        # Cost constants in USD per million tokens
        if model_name.lower() == 'sonnet':
            self._input_price_per_1M: float = 3
            self._output_price_per_1M: float = 15
        elif model_name.lower() == 'haiku':
            self._input_price_per_1M: float = 0.8
            self._output_price_per_1M: float = 4
    
    def _get_query_response(self, system_prompt: str, text: str) -> Dict[str, any]:
        """
        Send a query to the LLM and get the raw response.
        
        Args:
            system_prompt: The system prompt to guide the LLM's behavior
            text: The user query text
            
        Returns:
            The raw response from the LLM
        """
        return self.client.invoke_model(
            system_message=system_prompt,
            messages=[{"role": "user", "content": text}],
            temperature=0.1,
            top_p=0.4,
        )
    
    def _parse_json_response(self, response_text: str) -> Dict[str, str]:
        """
        Parse the JSON response from the LLM.
        
        Args:
            response_text: The text response from the LLM
            
        Returns:
            Parsed JSON as a dictionary
            
        Raises:
            Exception: If the response cannot be parsed as JSON
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON if it's embedded in other text
            json_start = response_text.find("{")
            if json_start >= 0:
                return json.loads(response_text[json_start:])
            raise Exception("Failed to parse response as JSON")
    
    def _calculate_costs(self, response: Dict[str, any]) -> None:
        """
        Calculate token usage and costs from the response.
        
        Args:
            response: The raw response from the LLM
        """
        self.input_tokens = response['input_tokens']
        self.output_tokens = response['output_tokens']
        self.total_tokens = self.input_tokens + self.output_tokens
        self.input_cost = self.input_tokens / 1000000 * self._input_price_per_1M
        self.output_cost = self.output_tokens / 1000000 * self._output_price_per_1M
        self.total_cost = self.input_cost + self.output_cost
    
    def query(self, system_prompt: str, text: str) -> Dict[str, str]:
        """
        Query the LLM and return a structured response.
        
        Args:
            system_prompt: The system prompt to guide the LLM's behavior
            text: The user query text
            
        Returns:
            A dictionary containing the 'Label' and 'Reasoning' from the LLM
            
        Raises:
            Exception: If the response cannot be parsed or lacks required fields
        """
        # Measure query time
        start_time = time.time()
        initial_response = self._get_query_response(system_prompt, text)
        self.query_time = time.time() - start_time
        
        # Parse response
        try:
            parsed_response = self._parse_json_response(initial_response['text'])
        except json.JSONDecodeError:
            # raise Exception(f"Failed to parse response as JSON: {e}")
            return json.loads(initial_response['text'][8:-3])
        
        # Calculate costs
        self._calculate_costs(initial_response)
        
        # Validate response structure
        if 'Label' not in parsed_response or 'Reasoning' not in parsed_response:
            raise Exception("Response missing required fields: must contain 'Label' and 'Reasoning'")
        
        return {
            'Label': parsed_response['Label'],
            'Reasoning': parsed_response['Reasoning']
        }