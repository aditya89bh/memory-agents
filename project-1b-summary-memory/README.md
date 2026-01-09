# Project 1B – Summary Memory Agent

This project extends short-term memory with a second layer:
a running summary that compresses older conversation context.

## What it shows
- Two-tier memory: recent buffer + summary
- Explicit forgetting and compression
- Prevention of recursive memory growth
- Colab-safe agent loop without frameworks

## Notes
The summarizer is heuristic (no LLM). It can be replaced later
with an LLM-based summarization call.
