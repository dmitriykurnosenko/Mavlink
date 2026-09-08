# AGENTS.md

Standalone git repo rooted at `tensor-test/` scripts plus `opencode.json`. Originally a "Mavlink test" repo (git remote named `Acode` points at a Mavlink repo), but all current content is Python LLM-teaching scripts. Ignore the remote name/Mavlink history; it is not representative of what this repo does.

## Layout
- `tensor-test/*.py` — five self-contained, runnable scripts teaching causal self-attention / mini-GPT:
  - `llm_from_scratch.py` — char-level tokenizer + GPT in PyTorch (run in Jupyter).
  - `llm_modern.py` — "LLaMA-style": RoPE, RMSNorm, SwiGLU, KV-cache, top-p (runnable directly).
  - `llm_numpy_demo.py` — pure-NumPy causal self-attention with a causality self-check (no PyTorch, works on Termux/Android).
  - `attention.py` — minimal single-head scaled dot-product attention example (shapes printed; torch).
  - `MultiHeadAttention.py` — reusable `MultiHeadAttention(nn.Module)` class + shape-check example (torch).
- The two newer files (`attention.py`, `MultiHeadAttention.py`) are duplicated verbatim in the separate scratch repo at `/public/project` (outside this git repo, no commits). They are not part of this repo's history yet.
- `opencode.json` — bare `$schema` placeholder, no instructions.
- `.opencode/` — opencode plugin tooling only (only its `.gitignore` is tracked; `package.json`/`package-lock.json`/`node_modules` are gitignored).

## Running
- No build, tests, or lint exist. No package manifest at repo root.
- Torch scripts need `torch`; `llm_from_scratch.py` also needs `requests`. `llm_numpy_demo.py` needs only `numpy` (matplotlib optional).
- /public/project is a scratch checkout: `/public/project/my_gpt/train.py` is incomplete — it imports `from mingpt.model` (not vendored) and uses a stub URL; do not treat it as runnable.
- All scripts download the Tiny Shakespeare dataset from a hardcoded GitHub URL at runtime — they require network access on first run. Do not "fix" this by vendoring data unless asked.
- Scripts auto-select `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`; nothing to configure.
- Dev context is Termux/Android where PyTorch often won't install — prefer `llm_numpy_demo.py` for verification there.

## Conventions
- Comments, docstrings, and print output are in Russian; keep that style when editing.
- Each file is a standalone tutorial script with numbered sections; add tutorial-style scripts under `tensor-test/`.
