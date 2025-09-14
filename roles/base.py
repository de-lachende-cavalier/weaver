from typing import List, Any, Dict
import logging
from datetime import datetime
from pathlib import Path

from groq import Groq


class BaseRole:
    """A base class representing a role model with a specific role and memory.
    Automatically logs all interactions to a timestamped file in a 'logs' directory."""

    def __init__(self, role: str, llm: str, groq_kwargs: Dict[str, Any] = {}) -> None:
        """
        Initializes the BaseRole and creates a timestamped log file.

        Args:
            role: The role of the model.
            llm: The name of the Groq model to use for the text-generation pipeline.
        """
        with open("groqkey.api", "r") as f:
            api_key = f.read().strip()
        self.client = Groq(api_key=api_key)
        self.groq_kwargs = groq_kwargs
        self.model = llm

        self.role: str = role
        self.system_prompt: str = Path(f"system_prompts/{role}.txt").read_text()

        self.memory: List[str] = []

        # create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # generate timestamped log file path (microseconds for more uniqueness)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log_file_path: Path = log_dir / f"{timestamp}" / f"{self.role}.log"

        logging.info(
            f"BaseRole initialized for role: '{self.role}'. Logging to: {self.log_file_path}"
        )

    def _log_to_file(self, entry: str) -> None:
        """Appends a log entry to the automatically generated log file."""
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                f.write(f"[{self.role} - {timestamp}] {entry}\n")
        except Exception as e:
            logging.error(
                f"Role '{self.role}': Failed to write to log file {self.log_file_path}: {e}"
            )

    def generate(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        save_to_memory: bool = True,
    ) -> str:
        """
        Generates text based on the given prompt using its role and system prompt.
        Logs interactions to the automatically generated log file.

        Args:
            prompt: The input prompt for the language model.
            save_to_memory: Whether to save the input prompt and output to memory.

        Returns:
            The generated text.
        """
        logging.info(
            f"Role '{self.role}' - generating with prompt: '{user_prompt[:100]}...'"
        )

        messages = []
        if system_prompt:
            # if the caller provides a system prompt, use it instead of the one found in the txt files
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        chat_completion = self.client.chat.completions.create(
            messages=messages, model=self.model, **self.groq_kwargs
        )

        generated_text = chat_completion.choices[0].message.content
        if generated_text is None:
            raise ValueError("Generated text is None")

        logging.info(
            f"Role '{self.role}' - generated text: '{generated_text[:100]}...'"
        )

        if save_to_memory:
            # input_entry = f"Input Prompt: {user_prompt}"
            output_entry = f"Generated Output: {generated_text}"

            # self.memory.append(input_entry)
            self.memory.append(output_entry)

            # self._log_to_file(input_entry)
            self._log_to_file(output_entry)

            logging.debug(
                f"Saved memory for role '{self.role}'. Memory size: {len(self.memory)} entries."
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
        logging.info(f"Memory cleared for role '{self.role}'.")
