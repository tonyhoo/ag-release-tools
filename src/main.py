import argparse
from test_tabular import test_tabular

def done():
    pass

module_tests_map = {
        "tabular": test_tabular,
        "done": done
    }

if __name__ == "__main__":
    module = input(f"Please select module in the list of {module_tests_map.keys()}")
    while module != "done":
        test_func = module_tests_map.get(module, None)
        if test_func is not None:
            print(f"Running module {module}...")
            test_func()
        else:
            print(f"Module passed {module} is not supported and should be in the list of {module_tests_map.keys()}")



