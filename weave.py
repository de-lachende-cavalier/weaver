# TODO should probably make this a CLI tool

human_input = prompt_human()

end_condition = False
prompt = narrator.edit_human_input_and_save(human_input)
while not end_condition:
    sim_out = worldsim.simulate_and_save_summary(prompt)
    prompt = narrator.edit_sim_output_and_save(sim_out)
    action = character.act_and_save(prompt)
    prompt = narrator.edit_action(action, prompt)

    update_end_condition()

story = editor.build_story_from_memory()
write_story_to_file(story)
