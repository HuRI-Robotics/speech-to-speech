import sys, json
import numpy as np
import requests
from datetime import datetime, timezone

from constants import OLLAMA_URL, EMBED_MODEL, CHAT_MODEL, INDEX_PATH, TOP_K, MEM_DIR, MEM_TXT, MEM_JSONL, HISTORY_TURNS

class LLM:
    def __init__(self, model=CHAT_MODEL):
        self.model = model

    def now_iso(self):
        """Get time from machine"""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def ensure_mem_files(self):
        MEM_DIR.mkdir(parents=True, exist_ok=True)
        if not MEM_TXT.exists():
            MEM_TXT.touch()
        if not MEM_JSONL.exists():
            MEM_JSONL.touch()

    def log_message(self, role, content):
        """Append the message to docs/discussion.txt and docs/memory.jsonl."""
        self.ensure_mem_files()
        ts = self.now_iso()
        with open(MEM_TXT, "a", encoding="utf-8") as f:
            f.write(f"{ts} [{role}]: {content}\n\n")
        with open(MEM_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": ts, "role": role, "content": content}) + "\n")

    def load_history(self,max_turns=HISTORY_TURNS):
        """Load last N user/assistant turns as chat messages."""
        self.ensure_mem_files()
        msgs = []
        with open(MEM_JSONL, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        lines = lines[-(2*max_turns):]
        for ln in lines:
            try:
                item = json.loads(ln)
                role = item.get("role", "")
                content = item.get("content", "")
                if role in ("user", "assistant"):
                    msgs.append({"role": role, "content": content})
            except Exception:
                continue
        return msgs

    def embed_one(self, text):
        """Get the embeddings"""
        r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                        json={"model": EMBED_MODEL, "input": text})
        r.raise_for_status()
        v = np.array(r.json()["embedding"], dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)
        return v

    def top_k(self, query_vec, mat, k=TOP_K):
        """Transfrom a text into a vector table"""
        scores = mat @ query_vec  # cosine via normalized dot
        idx = np.argsort(-scores)[:k]
        return idx, scores[idx]

    def load_index(self, path=INDEX_PATH):
        """Load vectore, texts and metadata from the rag"""
        data = np.load(path, allow_pickle=True)
        return data["vectors"], data["texts"], data["meta"]

    def build_user_prompt(self, question, contexts):
        """Ensence the user prompt with more data"""
        ctx_block = "\n\n---\n\n".join(contexts) if contexts else "(no retrieved context)"
        user = (
            f"Context:\n{ctx_block}\n\n"
            f"Question: {question}\n"
            "Answer as precisely and concisely as possible. If the answer isn't in the context, say you don't know."
        )
        return user


    def get_rag_contexts(self, question):
        """Get the information from the rag"""
        contexts = []
        try:
            V, texts, meta = self.load_index(INDEX_PATH) # For now no need of the metadata, maybe later
            qv = self.embed_one(question)
            idxs, scores = self.top_k(qv, V, k=TOP_K) # For now no need of the scores, maybe later
            contexts = [texts[i] for i in idxs]
        except FileNotFoundError:
            contexts = []
        except Exception as e:
            print(f"[RAG warning] {e}", file=sys.stderr)
        return contexts

    def memory(self, contexts, question, no_history=False):
        """Load the context, the history and rework the prompt"""
        system_prompt = (
            "Based on your context try to always give the best answer, and if you are searching something try to get it at any cost"
            "All answer has to be in a model like an answer from a conversation"
        )
        messages = [{"role": "system", "content": system_prompt}]

        if not no_history:
            messages.extend(self.load_history(HISTORY_TURNS))

        user_prompt = self.build_user_prompt(question, contexts)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def stream_chat(self, messages, think=True):
        """Call Ollama with streaming, return the full assistant text while printing tokens live."""
        payload = {"model": CHAT_MODEL, "stream": True, "messages": messages}
        assistant_text = []
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    chunk = data["message"]["content"]
                    assistant_text.append(chunk)
                    print(chunk, end="", flush=True)
                if data.get("done"):
                    break
        print() 
        return "".join(assistant_text)

    def chat_loop_written(self, no_history=False, think=True):
        """Create a chat bot"""
        while True:
            question = input("You (exit or quit to quit): ")
            if question.lower() in ("exit", "quit"):
                break
            contexts = self.get_rag_contexts(question)
            messages = self.memory(contexts, question, no_history)

            print(messages)
            self.log_message("user", question)
            assistant_reply = self.stream_chat(messages, think)
            self.log_message("assistant", assistant_reply)

def main():
    if len(sys.argv) == 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        print('Usage:   python rag_stream.py [--no-history]')
        print('             loop')
        print('                 ["your prompt"')
        print('                 "Answer from the assistant"]')
        return

    args = sys.argv[0:]
    no_history = False
    if "--no-history" in args:
        no_history = True
        args.remove("--no-history")

    llm = LLM(model=CHAT_MODEL)
    llm.chat_loop_written(no_history=no_history , think=true)


if __name__ == "__main__":
    main()
