"""Configuration for the shock finder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShockFinderConfig:
    """Configuration for the uniform-grid shock finder."""

    div_v_threshold: float = 0.0
    grad_T_min: float = 1e-30
    require_gradT_gradS_alignment: bool = True
    require_gradT_gradRho_alignment: bool = True
    normal_field: str = "temperature"
    shock_width_cells: int = 1
    require_pressure_jump: bool = True
    require_temperature_jump: bool = True
    require_density_jump: bool = True
    min_mach: float = 1.1
    mach_max: float = 1000.0
    reduce_to_centers: bool = True
    center_score: str = "compression"
    sampling_method: str = "nearest_axis"
    upstream_pressure_floor: float | None = None
    upstream_temperature_floor: float | None = None
    upstream_density_floor: float | None = None

    def __post_init__(self) -> None:
        """Validate supported configuration values."""

        if self.normal_field not in {"temperature", "pressure"}:
            raise ValueError("normal_field must be 'temperature' or 'pressure'.")
        if self.grad_T_min < 0.0:
            raise ValueError("grad_T_min must be >= 0.")
        if self.min_mach < 1.0:
            raise ValueError("min_mach must be >= 1.")
        if self.shock_width_cells < 1:
            raise ValueError("shock_width_cells must be >= 1.")
        if self.mach_max <= 1.0:
            raise ValueError("mach_max must be > 1.")
        if self.center_score not in {"compression", "mach_pressure", "mach_temperature", "combined"}:
            raise ValueError("center_score must be one of: compression, mach_pressure, mach_temperature, combined.")
        if self.sampling_method not in {"nearest_axis", "trilinear"}:
            raise ValueError("sampling_method must be either 'nearest_axis' or 'trilinear'.")
        for name, value in (
            ("upstream_pressure_floor", self.upstream_pressure_floor),
            ("upstream_temperature_floor", self.upstream_temperature_floor),
            ("upstream_density_floor", self.upstream_density_floor),
        ):
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be >= 0 when provided.")
