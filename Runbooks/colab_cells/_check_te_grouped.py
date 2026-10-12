import torch
import transformer_engine
import transformer_engine.pytorch as te

print("TE version:", getattr(transformer_engine, "__version__", "unknown"))
print("Grouped-related names:", [x for x in dir(te) if "roup" in x])
print("All te.pytorch top-level names:", sorted(dir(te)))
