# weaver

weaver is a project exploring the possibility of creating stories automatically, by using an entirely artificial LLM-based end-to-end pipeline[^1].

The human enters the loop in the beginning, specifying the initial conditions, and then the rest is _woven_ together in an automated fashion.

## Running the code

The code uses `python 3.13`. To run it, do the following:

1. (Optional) Set up a virtual environment;

2. Install all the requirements (`pip install -r requirements.txt`);

3. Install the project in editable mode (`pip install -e .`);

4. Run the `weave.py` script (`python weave.py`).

## Structure

`roles/` contains the classes corresponding to the various roles (character, editor, narrator and worldsim);
`system_prompts/` contains the system prompts for the various roles.

[^1]: quite the mouthful...
