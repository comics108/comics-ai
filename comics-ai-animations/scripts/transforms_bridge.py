"""Thin adapter reusing apps/comics-ai/comics-multimodal/scripts/ instead of duplicating it --
same "read/import, never modify" convention as comics-positioning's own positioning_bridge.py.

This flow needs live functions (resolve_reveal_animation, infer_kinds_for_file, ComicsArchive), not
just comics-multimodal's already-materialized JSON outputs (unlike positioning_bridge.py's Phase
1-4 path) -- so this bridges via sys.path injection, same mechanism comics-multimodal's own
baloons_bridge.py already uses for comics-ai-baloons.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
MULTIMODAL_SCRIPTS_DIR = REPO_ROOT / "apps" / "comics-ai" / "comics-multimodal" / "scripts"

if not MULTIMODAL_SCRIPTS_DIR.is_dir():
    raise RuntimeError(f"comics-multimodal scripts dir not found: {MULTIMODAL_SCRIPTS_DIR}")

# append, not insert(0, ...): our own modules stay authoritative on any name collision, same
# defensive convention baloons_bridge.py/positioning_bridge.py already established.
if str(MULTIMODAL_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(MULTIMODAL_SCRIPTS_DIR))

import baloons_bridge  # noqa: E402
import resting_position  # noqa: E402
from kind_heuristic import infer_kinds_for_file  # noqa: E402

find_comics_files = baloons_bridge.find_comics_files
ComicsArchive = baloons_bridge.ComicsArchive
resolve_reveal_animation = resting_position.resolve_reveal_animation
RevealAnimation = resting_position.RevealAnimation
PropertyReveal = resting_position.PropertyReveal
