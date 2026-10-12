import torch  # noqa: F401 -- must import before transformer_engine so its bundled cublas is resident
import transformer_engine.common.recipe as r
print([n for n in dir(r) if not n.startswith("_")])
