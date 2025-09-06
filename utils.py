def write_story_to_file(story: str, filename: str) -> None:
    """Write the completed story to a file."""
    with open(filename, "w") as file:
        file.write(story)
