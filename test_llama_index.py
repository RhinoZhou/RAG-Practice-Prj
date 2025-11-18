print("Testing llama_index import...")
try:
    from llama_index.core.node_parser import SentenceSplitter
    print("✓ Successfully imported SentenceSplitter")
except ImportError as e:
    print(f"✗ ImportError: {e}")
except Exception as e:
    print(f"✗ Exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()