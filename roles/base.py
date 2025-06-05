import logging
from typing import List
from datetime import datetime
from pathlib import Path


class BaseRole:
    """
    A base class representing a role model with a specific role and memory.
    Automatically logs all interactions to a timestamped file in a 'logs' directory.

    Attributes:
        role (str): The role of the role model (e.g., 'narrator', 'character').
        system_prompt (str): An initial prompt or context for the model.
        memory (List[str]): A list to store the history of prompts and generated texts.
        log_file_path (Path): Path to the automatically generated log file.
    """

    def __init__(self, role: str, system_prompt: str = ""):
        """
        Initializes the RoleModel and creates a timestamped log file.

        Args:
            role: The role of the role model.
            system_prompt: An initial system prompt or guiding context for the model.
        """
        self.role: str = role
        self.system_prompt: str = system_prompt
        self.memory: List[str] = []

        # create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # generate timestamped log file path
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )  # added microseconds for more uniqueness
        self.log_file_path: Path = log_dir / f"{self.role}_{timestamp}.log"

        if self.system_prompt:
            entry = f"SYSTEM PROMPT: {self.system_prompt}"
            self.memory.append(entry)
            self._log_to_file(f"[INITIAL SETUP] {entry}")

        logging.info(
            f"RoleModel initialized for role: '{self.role}'. Logging to: {self.log_file_path}"
        )

    def _log_to_file(self, entry: str) -> None:
        """Appends a log entry to the automatically generated log file."""
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )  # added microseconds
                f.write(f"[{self.role} - {timestamp}] {entry}\n")
        except Exception as e:
            logging.error(
                f"Role '{self.role}': Failed to write to log file {self.log_file_path}: {e}"
            )

    def generate(self, prompt: str, save_to_memory: bool = True) -> str:
        """
        Generates text based on the given prompt using its role and system prompt.
        Logs interactions to the automatically generated log file.

        Args:
            prompt: The input prompt for the language model.
            save_to_memory: Whether to save the input prompt and output to memory.

        Returns:
            The generated text.
        """
        logging.info(f"Role '{self.role}' generating with prompt: '{prompt[:100]}...'")

        full_prompt_for_llm = f"Role: {self.role}\n"
        if self.system_prompt:
            full_prompt_for_llm += f"System Context: {self.system_prompt}\n"
        full_prompt_for_llm += f"User Prompt: {prompt}"

        generated_text = (
            f"[{self.role} based on '{prompt[:30]}...'] generated text placeholder."
        )

        if save_to_memory:
            input_entry = f"Input Prompt: {prompt}"
            output_entry = f"Generated Output: {generated_text}"

            self.memory.append(input_entry)
            self.memory.append(output_entry)

            self._log_to_file(input_entry)
            self._log_to_file(output_entry)

            logging.debug(
                f"Saved to memory for role '{self.role}'. Memory size: {len(self.memory)} entries."
            )

        return generated_text

    def get_memory(self) -> List[str]:
        """
        Retrieves the entire memory of the role model.

        Returns:
            A list of strings representing the memory log.
        """
        return self.memory.copy()  # return a copy to prevent external modification

    def clear_memory(self) -> None:
        """
        Clears the role model's memory.
        The system prompt, if set, will be re-added and re-logged.
        """
        self.memory = []
        if self.system_prompt:
            entry = f"SYSTEM PROMPT: {self.system_prompt}"
            self.memory.append(entry)
            self._log_to_file(f"[MEMORY CLEARED, RE-ADDED] {entry}")
        logging.info(f"Memory cleared for role '{self.role}'.")
