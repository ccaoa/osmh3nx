from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional

import networkx as nx

from . import network_h3 as hnetx


@dataclass(frozen=True)
class CalibrationProfile:
    """
    Named preset for H3 timing behavior.

    The profile registry is just a dictionary of these presets keyed by name.
    It lets downstream code ask for a known timing configuration such as
    "default" or "rez10_default" instead of copying a dozen kwargs by hand.
    """

    name: str
    description: str
    default_h3_res: int = 10
    build_osm_weight_attr: str = "travel_time"
    default_query_weight_attr: str = "travel_time_route"
    route_weight_attr: str = "travel_time_route"
    report_weight_attr: str = "travel_time_postcalibrated"
    sample_miles: float = 0.1
    combine_parallel: str = "min"
    directional: bool = True
    enforce_min_step_time: bool = True
    v_max_mph: float = 50.0
    floor_speed_source: str = "vmax"
    min_osm_speed_mph: float = 15.0
    preserve_way_geometry: bool = True
    way_cell_refine_max_depth: int = 18
    route_floor_penalty_weight: float = 0.35
    report_floor_penalty_weight: float = 1.0


DEFAULT_PROFILE_NAME: str = "default"


PROFILE_REGISTRY: Dict[str, CalibrationProfile] = {
    "default": CalibrationProfile(
        name="default",
        description=(
            "Default calibrated H3 timing profile. Uses the current shared"
            " route-choice calibration settings with rez 10 as the default resolution."
        ),
    ),
    "rez10_default": CalibrationProfile(
        name="rez10_default",
        description=(
            "Explicit rez 10 version of the default calibrated timing profile."
        ),
        default_h3_res=10,
    ),
    "rez9_default": CalibrationProfile(
        name="rez9_default",
        description=(
            "Rez 9 variant of the current default calibrated timing profile."
        ),
        default_h3_res=9,
    ),
}


def available_profile_names() -> list[str]:
    return sorted(PROFILE_REGISTRY)


def get_calibration_profile(
    profile_name: str = DEFAULT_PROFILE_NAME,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> CalibrationProfile:
    if profile_name not in PROFILE_REGISTRY:
        raise ValueError(
            f"Unknown calibration profile '{profile_name}'. "
            f"Available: {available_profile_names()}"
        )

    profile = PROFILE_REGISTRY[profile_name]
    if not overrides:
        return profile
    return replace(profile, **dict(overrides))


def get_profile_registry() -> Dict[str, CalibrationProfile]:
    return dict(PROFILE_REGISTRY)


def get_default_h3_res(profile: CalibrationProfile) -> int:
    return int(profile.default_h3_res)


def get_default_query_weight_attr(profile: CalibrationProfile) -> str:
    return str(profile.default_query_weight_attr)


def get_default_report_weight_attr(profile: CalibrationProfile) -> str:
    return str(profile.report_weight_attr)


def get_network_h3_build_kwargs(profile: CalibrationProfile) -> Dict[str, Any]:
    return {
        "weight_attr": str(profile.build_osm_weight_attr),
        "sample_miles": float(profile.sample_miles),
        "combine_parallel": str(profile.combine_parallel),
        "directional": bool(profile.directional),
        "enforce_min_step_time": bool(profile.enforce_min_step_time),
        "v_max_mph": float(profile.v_max_mph),
        "floor_speed_source": str(profile.floor_speed_source),
        "min_osm_speed_mph": float(profile.min_osm_speed_mph),
        "preserve_way_geometry": bool(profile.preserve_way_geometry),
        "way_cell_refine_max_depth": int(profile.way_cell_refine_max_depth),
        "route_weight_attr": str(profile.route_weight_attr),
        "route_floor_penalty_weight": float(profile.route_floor_penalty_weight),
        "report_weight_attr": str(profile.report_weight_attr),
        "report_floor_penalty_weight": float(profile.report_floor_penalty_weight),
    }


def build_calibrated_h3_graph_from_osm(
    G_osm: nx.MultiDiGraph,
    *,
    h3_res: Optional[int] = None,
    profile_name: str = DEFAULT_PROFILE_NAME,
    profile_overrides: Optional[Mapping[str, Any]] = None,
) -> tuple[nx.Graph, CalibrationProfile]:
    profile = get_calibration_profile(profile_name, overrides=profile_overrides)
    resolved_h3_res = int(h3_res if h3_res is not None else profile.default_h3_res)
    H_h3 = hnetx.build_h3_travel_graph_from_osm(
        G_osm,
        h3_res=resolved_h3_res,
        **get_network_h3_build_kwargs(profile),
    )
    H_h3.graph["calibration_profile_name"] = str(profile.name)
    H_h3.graph["default_query_weight_attr"] = str(profile.default_query_weight_attr)
    return H_h3, profile
