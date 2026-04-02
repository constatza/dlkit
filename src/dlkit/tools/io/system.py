import importlib
import importlib.util
import sys
from hashlib import sha1
from pathlib import Path
from types import ModuleType

from pydantic import DirectoryPath, validate_call

from dlkit.tools.utils.general import kwargs_compatible_with


def _dynamic_module_name(path: Path) -> str:
    """Generate a collision-resistant module name for filesystem imports."""
    digest = sha1(str(path).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"_dlkit_dynamic_{path.stem}_{digest}"


def _register_safe_alias(module_name: str, module: ModuleType, path: Path) -> None:
    """Expose the original module stem when that alias will not shadow another module."""
    existing = sys.modules.get(module_name)
    if existing is None:
        sys.modules[module_name] = module
        return

    existing_file = getattr(existing, "__file__", None)
    if existing_file is not None and Path(existing_file).resolve() == path.resolve():
        sys.modules[module_name] = module


def import_from_module(class_name: str, module_prefix: str = "") -> type:
    """Dynamically import a module, class, function, or attribute from a string path.

    Args:
        class_name: The name of the class to import.
        module_prefix: Optional path prefix to prepend to the class name. Defaults to an empty string.

    Returns:
        The imported module, class, function, or attribute.

    Raises:
        ImportError: If import or attribute lookup fails.
    """
    # ensure only the last part of the path is used
    full_path = f"{module_prefix}.{class_name}" if module_prefix else class_name
    module_name, attr_name = full_path.rsplit(".", 1)
    module: ModuleType = importlib.import_module(module_name)
    return getattr(module, attr_name)


def import_from_path(class_name: str, path: Path, base: Path) -> type:
    """Import a module from a filesystem path.

    If from_path resolves to a .py file, load it directly.
    If it resolves to a directory with __init__.py, import it as a package.

    Args:
        class_name: The name of the class to import.
        path: Absolute or relative path to file or package directory.
        base: Base directory to resolve relative paths against.

    Returns:
        The imported module.

    Raises:
        ImportError: On import failure or invalid structure.
    """
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()

    if path.is_file():
        spec = importlib.util.spec_from_file_location(_dynamic_module_name(path), str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _register_safe_alias(path.stem, module, path)
        return getattr(module, class_name)

    if path.is_dir():
        init_file = path / "__init__.py"
        if not init_file.exists():
            raise ImportError(f"Directory {path} is not a package (missing __init__.py)")
        spec = importlib.util.spec_from_file_location(
            _dynamic_module_name(path),
            str(init_file),
            submodule_search_locations=[str(path)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for package: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _register_safe_alias(path.name, module, init_file)
        return getattr(module, class_name)

    raise ImportError(f"Path is neither file nor package: {path}")


@validate_call()
def load_class(class_name: str, module_path: str, settings_dir: DirectoryPath | None = None):
    """High-level loader: parse config, import module or file, and return the model class.

    Args:

    Returns:
        The nn.Module subclass specified in the config.

    Raises:
        ImportError or ValueError for any lookup or validation errors.
    """
    if settings_dir is not None and (r"/" in module_path or "\\" in module_path):
        return import_from_path(class_name, Path(module_path), settings_dir)

    return import_from_module(class_name=class_name, module_prefix=module_path)


def init_class(
    *,
    name: str,
    module_path: str,
    settings_path: DirectoryPath | None = None,
    exclude: set[str] | None = None,
    **kwargs,
):
    """Construct a class or function from a name and module path.

    Args:
        name: The name of the class to load.
        module_path: The path to the module containing the class.
        settings_path: The path to the directory containing the settings file.
        exclude: A set of keys to exclude from the kwargs.
        **kwargs: Keyword arguments to pass to the class constructor.

    Returns:
        An instance of the class.
    """
    class_name = load_class(name, module_path, settings_path)
    kwargs = kwargs_compatible_with(class_name, exclude=exclude, **kwargs)
    if isinstance(class_name, type):
        return class_name(**kwargs)
    return class_name
