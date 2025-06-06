from typing import Dict, Any
from .base import BaseRole


class WorldSim(BaseRole):
    """
    A specialized language model for simulating world events.
    """

    def __init__(
        self,
        llm: str = "microsoft/phi-3-mini-4k-instruct",
        pipeline_kwargs: Dict[str, Any] = {},
    ):
        super().__init__(role="worldsim", llm=llm, pipeline_kwargs=pipeline_kwargs)

    def simulate_world_event(self, current_situation_prompt: str) -> str:
        """
        Simulates a world event based on the current situation.
        The generated output is saved to this model's memory.
        """
        task_specific_prompt = f"As the world simulator, describe what happens next given the situation: '{current_situation_prompt}'"
        return self.generate(task_specific_prompt)
