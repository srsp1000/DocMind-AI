"""DocMind AI — production entry point.

Launches the FastAPI server with the modern web SPA interface.
For the optional Streamlit dev UI, run:  streamlit run frontend.py
"""

import sys
import uvicorn

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    print()
    print("  [DocMind AI] Learn Smarter")
    print("  ─────────────────────────────")
    print("  Open in browser -> http://127.0.0.1:8000")
    print("  API docs        -> http://127.0.0.1:8000/docs")
    print()
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
