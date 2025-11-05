import os


def get_file_name(file_path: str) -> str:
    """
    Extract the file name from a given file path.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The name of the file without the extension.
    """
    return os.path.splitext(os.path.basename(file_path))[0]
