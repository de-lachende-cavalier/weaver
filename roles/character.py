from typing import Dict, Any
from .base import BaseRole


class Character(BaseRole):
    """
    A specialized language model for a character's actions and decisions.
    """

    def __init__(
        self,
        llm: str = "moonshotai/kimi-k2-instruct-0905",
        groq_kwargs: Dict[str, Any] = {},
    ):
        super().__init__(role="character", llm=llm, groq_kwargs=groq_kwargs)

    def decide_action(self, current_situation_prompt: str, multichar: bool = False) -> str:
        """
        Decides an action for the character based on the current situation.
        The generated output is saved to this model's memory.
        """
        if multichar:
            task_specific_prompt = f"Decide the next action for all the characters involved based on: '{current_situation_prompt}'"
        else:
            task_specific_prompt = f"Decide your next action based on: '{current_situation_prompt}'"
        return self.generate(user_prompt=task_specific_prompt)
