from dlkit.io.settings import load_validated_settings
from dlkit.setup.model_state import build_model_state


def model_state_from_path(path):
    settings = load_validated_settings(path)
    model_state = build_model_state(settings)
    return model_state
