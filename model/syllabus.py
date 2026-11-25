from __future__ import annotations

NMC_REDUCED_SYLLABUS_EXCLUSIONS = frozenset(
    {
        "States of Matter",
        "s-Block Elements",
        "Environmental Chemistry",
        "General Principles & Metallurgy",
        "Surface Chemistry",
    }
)
SYLLABUS_MODES = ("all", "nmc_reduced")
DEFAULT_SYLLABUS_MODE = "all"


def normalise_syllabus_mode(mode: str | None) -> str:
    value = (mode or DEFAULT_SYLLABUS_MODE).strip().lower()
    if value not in SYLLABUS_MODES:
        raise ValueError(f"syllabus_mode must be one of {SYLLABUS_MODES}, got {mode!r}")
    return value


def in_syllabus(chapter: str, mode: str) -> bool:
    return not (mode == "nmc_reduced" and chapter in NMC_REDUCED_SYLLABUS_EXCLUSIONS)
