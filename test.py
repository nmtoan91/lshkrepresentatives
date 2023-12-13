import numpy as np
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("LSHkRepresentatives", "D:/TOANWORKSPACE/lshkrepresentatives/LSHkRepresentatives/LSHkRepresentatives.py")

foo = importlib.util.module_from_spec(spec)

sys.modules["LSHkRepresentatives"] = foo
spec.loader.exec_module(foo)