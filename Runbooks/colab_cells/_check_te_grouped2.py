import inspect

import torch
import transformer_engine.pytorch as te

print("=== GroupedLinear.__init__ signature ===")
print(inspect.signature(te.GroupedLinear.__init__))
print()
print("=== GroupedLinear.__init__ docstring ===")
print(te.GroupedLinear.__init__.__doc__)
print()
print("=== GroupedLinear.forward signature ===")
print(inspect.signature(te.GroupedLinear.forward))
print()
print("=== GroupedLinear.forward docstring ===")
print(te.GroupedLinear.forward.__doc__)
