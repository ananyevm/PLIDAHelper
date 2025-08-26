import openai
from config import OPENAI_API_KEY, GPT_MODEL_NAME, SEED

class LLMClient:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = GPT_MODEL_NAME
        self.seed = SEED
    
    def complete(self, system_prompt, user_prompt):
        """Get completion from OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                seed=self.seed
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"LLM API error: {e}")