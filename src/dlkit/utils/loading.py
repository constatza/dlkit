import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from pydantic import FilePath, validate_call

from dlkit.settings.base_settings import ClassSettings


def import_from_module(class_name: str, module_prefix: str = '') -> type:
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
	full_path = f'{module_prefix}.{class_name}'
	module_name, attr_name = full_path.rsplit('.', 1)
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
		spec = importlib.util.spec_from_file_location(path.stem, str(path))
		if spec is None or spec.loader is None:
			raise ImportError(f'Cannot load spec for file: {path}')
		module = importlib.util.module_from_spec(spec)
		sys.modules[spec.name] = module
		spec.loader.exec_module(module)  # type: ignore
		return getattr(module, class_name)

	if path.is_dir():
		init_file = path / '__init__.py'
		if not init_file.exists():
			raise ImportError(f'Directory {path} is not a package (missing __init__.py)')
		sys.path.insert(0, str(path.parent))
		try:
			module = importlib.import_module(path.name)
		finally:
			sys.path.pop(0)
		return getattr(module, class_name)

	raise ImportError(f'Path is neither file nor package: {path}')


@validate_call()
def load_class(class_name: str, module_path: str, settings_path: FilePath | None = None):
	"""High-level loader: parse config, import module or file, and return the model class.

	Args:

	Returns:
	    The nn.Module subclass specified in the config.

	Raises:
	    ImportError or ValueError for any lookup or validation errors.
	"""
	if settings_path is not None and (r'/' in module_path or '\\' in module_path):
		return import_from_path(class_name, Path(module_path), settings_path.parent)

	return import_from_module(class_name=class_name, module_prefix=module_path)


def init_class[T](cls_settings: ClassSettings[T], settings_path: FilePath | None = None, **kwargs ) -> T:
	"""Initialize a class instance from ClassSettings.
	
	Args:
		cls_settings:
		settings_path:
		**kwargs:

	Returns:
		An instance of the class specified in cls_settings.
	"""
	class_name = load_class(cls_settings.name, cls_settings.module_path, settings_path)
	if isinstance(class_name, type):
		return class_name(**cls_settings.to_dict_compatible_with(class_name, **kwargs))
	return class_name
