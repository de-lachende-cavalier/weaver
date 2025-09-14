# TODO could use a simple YAML config file to manage some of the hardcoded values here (hdyra is overkill)

import os
import argparse

from roles import Narrator, WorldSim, Character, Editor
from utils import write_story_to_file


def main(human_input: str, max_iterations: int) -> None:
    print("[+] Instantiating the roles...")
    # use one set of kwargs for simplicity
    # TODO use CLI to set these dynamically? or hydra after all?
    global_kwargs = {
        "temperature": 0.6,
        "max_completion_tokens": 4096,
        "top_p": 1,
        "stream": False,
        "stop": None,
    }
    narrator = Narrator(groq_kwargs=global_kwargs)
    worldsim = WorldSim(groq_kwargs=global_kwargs)
    character = Character(groq_kwargs=global_kwargs)
    editor = Editor(groq_kwargs=global_kwargs)

    print("[+] Initialising the narrator...")
    prompt = narrator.edit_human_input(human_input)
    for i in range(max_iterations):
        print(f"[+] Iteration {i+1}:")
        print("\t[+] WorldSim - Simulating...")
        sim_out = worldsim.simulate_world_event(prompt)
        print("\t[+] Narrator - Editing simulation output...")
        prompt = narrator.edit_simulation_output(sim_out, prompt)
        # TODO i currently allow for only one protagonist, what if i added multiple interacting characters?
        print("\t[+] Character - Deciding action...")
        action = character.decide_action(prompt)
        print("\t[+] Narrator - Narrating action...")
        prompt = narrator.narrate_action(action, prompt)

    print("[+] Finalising the story...")
    memory_dict = {
        "narrator": narrator.get_memory(),
        "worldsim": worldsim.get_memory(),
        "character": character.get_memory(),
    }
    story = editor.compile_story(memory_dict)

    # write to stories/ by default
    os.makedirs("stories", exist_ok=True)
    story_fname = os.path.join("stories", "story.txt")
    print(f"[+] Saving the story to {story_fname}...")
    write_story_to_file(story, story_fname)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-m",
        "--max_iterations",
        type=int,
        required=True,
        help="Maximum number of iterations to use for the story",
    )

    args = args.parse_args()

    human_input = input(">>> Insert initial prompt: ")
    main(human_input, args.max_iterations)
    print("[+] All done!")
