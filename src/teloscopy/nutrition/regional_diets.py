"""Country-wise and state-wise regional dietary profiles.

Maps frontend region names → backend diet_advisor region_ids, and provides
hierarchical country → state lookups for localised meal planning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateProfile:
    """Dietary profile for a state / province within a country."""
    name: str
    staple_foods: list[str] = field(default_factory=list)
    proteins: list[str] = field(default_factory=list)
    vegetables: list[str] = field(default_factory=list)
    fruits: list[str] = field(default_factory=list)
    spices: list[str] = field(default_factory=list)
    traditional_dishes: list[str] = field(default_factory=list)
    dietary_notes: str = ""
    telomere_relevant_foods: list[str] = field(default_factory=list)
    common_deficiencies: list[str] = field(default_factory=list)
    cooking_methods: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CountryProfile:
    """Dietary profile for a country, with optional state-level detail."""
    name: str
    region_id: str  # maps to diet_advisor GEOGRAPHIC_FOOD_DB region_id
    staple_foods: list[str] = field(default_factory=list)
    proteins: list[str] = field(default_factory=list)
    vegetables: list[str] = field(default_factory=list)
    fruits: list[str] = field(default_factory=list)
    spices: list[str] = field(default_factory=list)
    traditional_dishes: list[str] = field(default_factory=list)
    dietary_notes: str = ""
    states: dict[str, StateProfile] = field(default_factory=dict)
    common_deficiencies: list[str] = field(default_factory=list)
    cultural_restrictions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# JSON data loading
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "json"


def _load_json(name: str) -> Any:
    """Load and return a JSON file from the data directory."""
    with open(_DATA_DIR / name) as f:
        return json.load(f)


def _build_state_profile(data: dict[str, Any]) -> StateProfile:
    """Convert a raw JSON dict into a *StateProfile* dataclass instance."""
    return StateProfile(**data)


def _build_country_profile(data: dict[str, Any]) -> CountryProfile:
    """Convert a raw JSON dict into a *CountryProfile* dataclass instance."""
    raw = dict(data)
    states_data = raw.pop("states", {})
    states = {name: _build_state_profile(sp) for name, sp in states_data.items()}
    return CountryProfile(**raw, states=states)


# ---------------------------------------------------------------------------
# Frontend region name → backend region_id mapping
# (fixes the mismatch between the select-option labels and diet_advisor IDs)
# ---------------------------------------------------------------------------

FRONTEND_REGION_MAP: dict[str, str] = _load_json("frontend_region_map.json")

# ---------------------------------------------------------------------------
# Country → more-specific region_id overrides
# (when a country clearly maps to a sub-region of the broad frontend region)
# ---------------------------------------------------------------------------

_COUNTRY_REGION_OVERRIDE: dict[str, str] = _load_json("country_region_override.json")

# ---------------------------------------------------------------------------
# State-level region_id overrides for countries with diverse sub-regions
# ---------------------------------------------------------------------------

_STATE_REGION_OVERRIDE: dict[str, dict[str, str]] = _load_json("state_region_override.json")


# =========================================================================
# Region → countries / Country → states lookup tables
# =========================================================================

REGION_COUNTRIES: dict[str, list[str]] = _load_json("region_countries.json")

COUNTRY_STATES: dict[str, list[str]] = _load_json("country_states.json")

# All country profiles keyed by name
_raw_profiles: dict[str, Any] = _load_json("country_profiles.json")
_ALL_COUNTRIES: dict[str, CountryProfile] = {
    name: _build_country_profile(profile_data)
    for name, profile_data in _raw_profiles.items()
}

_ALL: list[str] = list(REGION_COUNTRIES.keys())


# =========================================================================
# Public API functions
# =========================================================================

def resolve_region(
    frontend_region: str,
    *,
    country: str | None = None,
    state: str | None = None,
) -> str:
    """Resolve a frontend region label + optional country/state to a diet_advisor region_id.

    Priority: state override > country override > frontend map > pass-through.
    """
    # 1. State-level override (most specific)
    if country and state and country in _STATE_REGION_OVERRIDE:
        state_map = _STATE_REGION_OVERRIDE[country]
        if state in state_map:
            return state_map[state]

    # 2. Country-level override
    if country and country in _COUNTRY_REGION_OVERRIDE:
        return _COUNTRY_REGION_OVERRIDE[country]

    # 3. Frontend label → region_id mapping
    if frontend_region in FRONTEND_REGION_MAP:
        return FRONTEND_REGION_MAP[frontend_region]

    # 4. Pass-through (assume it's already a valid region_id)
    return frontend_region


def get_country_profile(country: str) -> CountryProfile | None:
    """Return the full dietary profile for a country, or ``None``."""
    return _ALL_COUNTRIES.get(country)


def get_state_profile(country: str, state: str) -> StateProfile | None:
    """Return the dietary profile for a state within a country, or ``None``."""
    cp = _ALL_COUNTRIES.get(country)
    if cp is None:
        return None
    return cp.states.get(state)


def list_countries_for_region(frontend_region: str) -> list[str]:
    """Return the list of countries available for a given frontend region label."""
    return REGION_COUNTRIES.get(frontend_region, [])


def list_states_for_country(country: str) -> list[str]:
    """Return the list of states with detailed profiles for a given country."""
    return COUNTRY_STATES.get(country, [])
