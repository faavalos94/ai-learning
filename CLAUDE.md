# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A learning repo of small, standalone Python scripts that build up from a basic Claude chatbot to RAG, tool use, an agent, and evals. Each step is its own commit. There is no package, build step, or linter: every file is a script with a `main()`. Offline unit tests use pytest (`test_*.py`). The `eval_*.py` scripts are separate and call live APIs.

## Setup and commands

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

`.env` (gitignored, loaded with `python-dotenv`) must provide `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` (the second is used by the `voyageai.Client()` in `rag_embed.py`).

Run any script directly, for example `python agent.py`. Scripts must be run from the repo root because they open `handbook.txt` by relative path.

Unit tests (no API calls):
- `python -m pytest`: run all tests
- `python -m pytest test_tools.py::test_divide_by_zero`: run a single test

Evals (each one is a single runnable "test suite"; to run a single case, edit that file's `TEST_CASES`):
- `python eval_extract.py`: field checks for `extract.extract_job`
- `python eval_rag.py`: retrieval hit rate, keyword vs. embedding
- `python eval_answers.py`: LLM-as-judge on RAG answers (`judge()` lives here)
- `python eval_agent.py`: trajectory checks (`must_use` / `must_not_use` tools, step limit) plus the judge on the final answer

## Architecture

- **Corpus:** `handbook.txt` is split on `## ` headings into `{title, text}` chunks by `rag.load_chunks`. Every RAG/agent script uses this function.
- **Retrieval:** `rag.py` has keyword-overlap retrieval. `rag_embed.py` adds Voyage embeddings (`EMBED_MODEL`, `vo`): `embed_chunks` stores an `"embedding"` on each chunk dict in place, and `retrieve` ranks chunks by cosine similarity. Both modules expose the same `retrieve(question, chunks, k)` and `answer(question, chunks)` names.
- **Tools:** `tools_chat.py` defines `TOOLS` (`get_current_datetime`, `calculate`) and `run_tool(name, input)`.
- **Agent:** `agent.py` combines these. It adds a `search_handbook` tool (backed by `rag_embed.retrieve`) to the basic tools, sends tool calls it doesn't handle itself to `tools_chat.run_tool`, and loops in `run_agent(messages, chunks)` for up to `MAX_STEPS`. It returns `(answer, tool_call_names)`, and `eval_agent.py` relies on that shape and on the `"Stopped after {MAX_STEPS} steps"` message.
- **Imports between scripts:** evals import functions from the scripts (for example, `eval_agent` imports `agent.run_agent` and `eval_answers.judge`). Renaming or changing the signature of these functions breaks the evals.
- Every module creates its own `Anthropic()` client at import time and uses the model `claude-haiku-5-5`.

## Rules

- Never read, print, or edit .env (it contains API keys)
