from dlkit.utils.system_utils import import_dynamically, filter_kwargs


def initialize_pruner(pruner_config):
    if not pruner_config:
        pruner_config = {"name": "NopPruner"}

    pruner_name = pruner_config.get("name", "NopPruner")
    pruner_class = import_dynamically(pruner_name, prepend="optuna.pruners")
    return pruner_class(**filter_kwargs(pruner_config))
