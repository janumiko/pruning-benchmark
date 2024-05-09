from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules

# get the current package name
package_name = __name__

# get the path of the current package
package_path = str(Path(__file__).parent)

# iterate over all modules in the current package
for _, module_name, _ in iter_modules([package_path]):
    # import the module
    import_module(f"{package_name}.{module_name}")
