from .base import BaseRole


class Narrator(BaseRole):
    """
    A specialized language model for narration tasks.
    Handles editing human input, simulation outputs, and narrating actions.
    """

    def __init__(
        self,
        system_prompt: str = "You are a master storyteller. Weave events into a compelling narrative.",
    ):
        super().__init__(role="narrator", system_prompt=system_prompt)

    def edit_human_input(self, human_input: str) -> str:
        """
        Processes and 'edits' human input to set the initial scene.
        The generated output is saved to this model's memory.
        """
        task_specific_prompt = f"As a narrator, refine and set the scene based on the human's starting idea: '{human_input}'"
        return self.generate(task_specific_prompt)

    def edit_simulation_output(self, sim_output: str, context_prompt: str) -> str:
        """
        Edits/narrates the simulation output, providing a narrative interpretation.
        The generated output is saved to this model's memory.
        """
        task_specific_prompt = f"As a narrator, interpret and describe the outcome: '{sim_output}', following on from: '{context_prompt}'."
        return self.generate(task_specific_prompt)

    def narrate_action(self, action_taken: str, ongoing_prompt: str) -> str:
        """
        Narrates an action taken by a character within the story's context.
        The generated output is saved to this model's memory.
        """
        task_specific_prompt = f"As a narrator, describe the character performing the action: '{action_taken}', in the context of: '{ongoing_prompt}'."
        return self.generate(task_specific_prompt)
