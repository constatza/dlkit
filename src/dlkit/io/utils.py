def complete_with_defaults(read: dict, default: dict) -> dict:
    """
    Recursively completes the 'read' dictionary by filling in any missing entries with values from 'default'.

    Args:
    - read (dict): The dictionary that was read and might have missing entries.
    - default (dict): The dictionary containing default values.

    Returns:
    - dict: The 'read' dictionary, completed with missing values from 'default'.
    """
    for key, value in default.items():
        if isinstance(value, dict):
            # If the key exists and is a dict, recurse
            read[key] = complete_with_defaults(read.get(key, {}), value)
        else:
            # If the key is missing, use the default value
            read.setdefault(key, value)
    return read
