# utils/suggestion.py
from dlkit.io.logging import get_logger

logger = get_logger(__name__)


def suggest(trial, config, prefix=""):
    suggested_values = {}
    for table_name, table_config in config.items():
        suggested_values[table_name] = {}
        for key, value in table_config.items():
            suggestion_name = f"{prefix}{key}"
            if isinstance(value, dict):
                param_type = value.get("type")
                if "low" in value and "high" in value:
                    low = value["low"]
                    high = value["high"]
                    if param_type == "int":
                        suggested_values[table_name][key] = trial.suggest_int(
                            suggestion_name,
                            low,
                            high,
                            step=value.get("step", 1),
                            log=value.get("log", False),
                        )
                    elif param_type == "float":
                        suggested_values[table_name][key] = trial.suggest_float(
                            suggestion_name,
                            low,
                            high,
                            step=value.get("step", None),
                            log=value.get("log", False),
                        )
                    else:
                        raise ValueError(
                            f"Unsupported type '{param_type}' for range suggestion."
                        )
                elif "choice" in value:
                    choices = value["choice"]
                    suggested_values[table_name][key] = trial.suggest_categorical(
                        suggestion_name, choices
                    )
                else:
                    logger.warning(f"Parameter '{key}' is not recognized.")
            else:
                suggested_values[table_name][key] = value
    return suggested_values
