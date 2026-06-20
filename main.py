"""Thin entry point that delegates to the CLI in app.py.

Kept so `python main.py ...` and the `speechsep` console script both work.
"""

from app import main

if __name__ == "__main__":
    main()
