"""Preserves `python -m scripts.generated_knowledge_stage0` as the package's CLI entry point."""

import asyncio
import sys

from scripts.generated_knowledge_stage0.run import main

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
