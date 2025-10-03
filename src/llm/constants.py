from pathlib import Path

# ==== CONFIG ====
OLLAMA_URL  = "http://localhost:11434"
CHAT_MODEL  = "deepseek-r1:7b"       # or your model, e.g. "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"        # embedding model for RAG
INDEX_PATH  = "index.npz"
MEM_DIR     = Path("docs")
DISCU_DIR   = Path("discussion")
MEM_TXT     = MEM_DIR / DISCU_DIR / "discussion.txt"  # human-readable log
MEM_JSONL   = MEM_DIR / "memory.jsonl"    # structured log
HISTORY_TURNS = 6  # how many prior user↔assistant pairs to include
TOP_K = 4           # RAG top-k
# ===============