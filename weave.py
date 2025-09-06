# TODO should probably make this a CLI tool

import os

from roles import Narrator, WorldSim, Character, Editor

narrator = Narrator(
    llm="distilgpt2", pipeline_kwargs={"device": "cpu", "max_length": 512}
)
worldsim = WorldSim(
    llm="distilgpt2", pipeline_kwargs={"device": "cpu", "max_length": 512}
)
character = Character(
    llm="distilgpt2", pipeline_kwargs={"device": "cpu", "max_length": 512}
)
editor = Editor(llm="distilgpt2", pipeline_kwargs={"device": "cpu", "max_length": 512})

human_input = prompt_human()

end_condition = False
prompt = narrator.edit_human_input(human_input)
while not end_condition:
    sim_out = worldsim.simulate_world_event(prompt)
    prompt = narrator.edit_simulation_output(sim_out, prompt)
    action = character.decide_action(prompt)
    prompt = narrator.narrate_action(action, prompt)

    end_condition = check_end_condition(end_condition)

memory_dict = {
    "narrator": narrator.get_memory(),
    "worldsim": worldsim.get_memory(),
    "character": character.get_memory(),
}
story = editor.compile_story(memory_dict)

os.makedirs("stories", exist_ok=True)
write_story_to_file(story)
