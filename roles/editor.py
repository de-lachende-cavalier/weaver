from typing import List, Dict

from .base import BaseRole


class Editor(BaseRole):
    """
    A specialized language model for editing and compiling the final story.
    """

    def __init__(
        self,
        system_prompt: str = "You are a meticulous editor. Combine all provided logs and narratives into a single, polished, and coherent story.",
    ):
        super().__init__(role="editor", system_prompt=system_prompt)

    def compile_story_from_logs(self, all_memories: Dict[str, List[str]]) -> str:
        """
        Compiles a cohesive story from the memory logs of various roles.
        The generated story is NOT saved to this editor model's primary memory by default,
        as it's a final product derived from other memories.
        """
        story_material = "Combine the following perspectives and events into a cohesive narrative story:\n\n"
        for role_name, memory_log in all_memories.items():
            story_material += f"--- Log from {role_name.upper()} ---\n"
            for entry in memory_log:
                story_material += f"{entry}\n"
            story_material += f"--- End of {role_name.upper()} Log ---\n\n"

        # the editor uses its own 'generate' capability to write the story
        return self.generate(story_material, save_to_memory=False)
