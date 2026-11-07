from .arch_a import LlamaLM
from .arch_b import EncDecLM
from .arch_c import HybridLM
from .arch_d import EfficientLM

MODEL_CLASSES = {c.__name__: c for c in (LlamaLM, EncDecLM, HybridLM, EfficientLM)}
