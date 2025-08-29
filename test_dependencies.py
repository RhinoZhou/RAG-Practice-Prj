try:
    import pandas
    print(f"✅ pandas installed (version: {pandas.__version__})")
except ImportError:
    print("❌ pandas not installed")

try:
    import pyarrow
    print(f"✅ pyarrow installed (version: {pyarrow.__version__})")
except ImportError:
    print("❌ pyarrow not installed")

try:
    import pydantic
    print(f"✅ pydantic installed (version: {pydantic.__version__})")
except ImportError:
    print("❌ pydantic not installed")

try:
    import pdfplumber
    print(f"✅ pdfplumber installed (version: {pdfplumber.__version__})")
except ImportError:
    print("❌ pdfplumber not installed")

try:
    import docx
    print(f"✅ python-docx installed (version: {docx.__version__})")
except ImportError:
    print("❌ python-docx not installed")

try:
    import pptx
    print(f"✅ python-pptx installed")
except ImportError:
    print("❌ python-pptx not installed")

try:
    import hashlib
    print(f"✅ hashlib installed")
except ImportError:
    print("❌ hashlib not installed")

try:
    import uuid
    print(f"✅ uuid installed")
except ImportError:
    print("❌ uuid not installed")