#!/usr/bin/env bash
# Launch the Stock Portfolio Analyzer.
# DYLD_LIBRARY_PATH works around Homebrew Python 3.14 failing to load the system libexpat.
cd "$(dirname "$0")"
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
exec .venv/bin/streamlit run app.py "$@"
