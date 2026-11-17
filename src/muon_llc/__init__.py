"""Learning dynamics under Muon through the lens of singular learning theory.

Shared core:  optim (Muon family, SGD, AdamW), slt (samplers and LLC estimators), config, style.
Testbeds:     dln (deep linear networks), tms (toy model of superposition).
"""

__version__ = "0.2.0"


def _utf8_stdio() -> None:
    """Progress lines contain λ, β, γ; consoles and pipes on Windows default to a legacy code page."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_utf8_stdio()
