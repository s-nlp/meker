import os
import sys
import importlib
def import_source_as_module(source_path):
    'Importing module from a specified path.'
    'See https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly'
    _, file_name = os.path.split(source_path)
    module_name = os.path.splitext(file_name)[0]
    module_spec = importlib.util.spec_from_file_location(module_name, source_path)
    if module_name in sys.modules:
        print(f'Module {module_name} is already imported!')
        module = sys.modules[module_name]
    else:
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
    return module