try:
    import transformer_engine
    print("TE_AVAILABLE")
except ImportError as e:
    print("TE_MISSING:", e)
