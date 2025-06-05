from .base import BaseRole


class WorldSim(BaseRole):
    """
    A specialized language model for simulating world events.
    """

    def __init__(
        self,
        system_prompt: str = "You are a detailed world simulator. Describe environments and events realistically based on the current situation.",
    ):
        super().__init__(role="worldsim", system_prompt=system_prompt)

    def simulate_world_event(self, current_situation_prompt: str) -> str:
        """
        Simulates a world event based on the current situation.
        The generated output is saved to this model's memory.
        """
        task_specific_prompt = f"As the world simulator, describe what happens next given the situation: '{current_situation_prompt}'"
        return self.generate(task_specific_prompt)
