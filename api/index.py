"""
Vercel entry point for AIU Smart Resume Analyzer.
This file must live at api/index.py for Vercel to pick it up.
"""
import sys
import os

# Make the project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: F401 — Vercel expects `app` to be importable here
