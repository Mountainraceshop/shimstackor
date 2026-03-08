from __future__ import annotations

from math import pi, sqrt
from typing import List

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


app = FastAPI(title="Shim Calculator Engineering MVP")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


STEEL_E = 210e9
POISSON = 0.30
DEFAULT_GAMMA = 1.4
ATM_PA = 101325.0


class Shim(BaseModel):
    diameter_mm: float = Field(..., gt=0)
    thickness_mm: float = Field(..., gt=0)
    qty: int = Field(1, ge=1, le=20)


class SimulationInput(BaseModel):
    mode: str = "fork"
    piston_diameter_mm: float = Field(24.0, gt=1)
    shaft_diameter_mm: float = Field(12.0, gt=0)
    port_count: int = Field(4, ge=1, le=20)
    port_diameter_mm: float = Field(6.0, gt=0)
    bleed_diameter_mm: float = Field(0.9, gt=0)
    rebound_orifice_diameter_mm: float = Field(1.2, gt=0)
    low_speed_clicks_out: int = Field(12, ge=0, le=30)
    high_speed_turns_out: float = Field(1.5, ge=0, le=6)
    oil_density_kg_m3: float = Field(850.0, gt=100)
    oil_viscosity_cst: float = Field(16.0, gt=1)
    discharge_coefficient: float = Field(0.70, gt=0.1, lt=1.0)
    spring_rate_n_per_mm: float = Field(4.6, gt=0)
    preload_mm: float = Field(5.0, ge=0)
    travel_mm: float = Field(300.0, gt=1)
    motion_ratio: float = Field(1.0, gt=0.1)
    rider_mass_kg: float = Field(85.0, gt=1)
    bike_mass_kg: float = Field(110.0, gt=1)
    oil_height_mm: float = Field(110.0, gt=1)
    tube_inner_diameter_mm: float = Field(48.0, gt=1)
    initial_gas_pressure_bar: float = Field(6.0, gt=0.1)
    reservoir_volume_cc: float = Field(120.0, gt=1)
    shaft_velocity_max_m_s: float = Field(1.0, gt=0.05, le=5)
    steps: int = Field(41, ge=11, le=401)
    compression_stack: List[Shim] = Field(default_factory=lambda: [
        Shim(diameter_mm=24, thickness_mm=0.15, qty=6),
        Shim(diameter_mm=22, thickness_mm=0.15, qty=2),
        Shim(diameter_mm=20, thickness_mm=0.10, qty=1),
    ])
    rebound_stack: List[Shim] = Field(default_factory=lambda: [
        Shim(diameter_mm=22, thickness_mm=0.15, qty=5),
        Shim(diameter_mm=20, thickness_mm=0.15, qty=2),
        Shim(diameter_mm=18, thickness_mm=0.10, qty=1),
    ])


def mm2_to_m2(v: float) -> float:
    return v * 1e-6


def mm_to_m(v: float) -> float:
    return v / 1000.0


def bar(v: float) -> float:
    return v * 1e5


def round_port_area_m2(count: int, diameter_mm: float) -> float:
    return count * pi * (mm_to_m(diameter_mm) ** 2) / 4.0


def single_orifice_area_m2(diameter_mm: float) -> float:
    return pi * (mm_to_m(diameter_mm) ** 2) / 4.0


def annular_area_m2(outer_d_mm: float, inner_d_mm: float) -> float:
    return max(pi * (mm_to_m(outer_d_mm) ** 2 - mm_to_m(inner_d_mm) ** 2) / 4.0, 0.0)


def piston_effective_area(inp: SimulationInput) -> float:
    piston_area = single_orifice_area_m2(inp.piston_diameter_mm)
    shaft_area = single_orifice_area_m2(inp.shaft_diameter_mm)
    return piston_area - shaft_area if inp.mode == "shock" else piston_area


def displaced_flow_m3_s(inp: SimulationInput, shaft_velocity_m_s: float) -> float:
    return abs(shaft_velocity_m_s) * piston_effective_area(inp)


def click_factor(clicks_out: int, max_clicks: int = 30) -> float:
    # More clicks out => more open.
    return 0.15 + 0.85 * min(max(clicks_out, 0), max_clicks) / max_clicks


def hs_factor(turns_out: float, max_turns: float = 6.0) -> float:
    # More turns out => lower preload => opens easier.
    open_factor = min(max(turns_out, 0.0), max_turns) / max_turns
    return 1.25 - 0.65 * open_factor


def shim_stack_stiffness(stack: List[Shim], clamp_d_mm: float = 8.0) -> float:
    # Transparent but simplified plate-bending-inspired stiffness proxy.
    k_total = 0.0
    clamp_r = mm_to_m(clamp_d_mm) / 2.0
    for shim in stack:
        active_r = max(mm_to_m(shim.diameter_mm) / 2.0 - clamp_r, 1e-4)
        t = mm_to_m(shim.thickness_mm)
        single_k = (STEEL_E * t**3) / (12.0 * (1 - POISSON**2) * active_r**3)
        k_total += single_k * shim.qty
    return k_total


def shim_opening_pressure_pa(stack: List[Shim], port_area_m2: float, hs_turns_out: float) -> float:
    k = shim_stack_stiffness(stack)
    nominal_lift = 0.08e-3  # 0.08 mm first meaningful opening
    preload_force = k * nominal_lift * hs_factor(hs_turns_out)
    return max(preload_force / max(port_area_m2, 1e-9), 5e3)


def shim_lift_m(delta_p_pa: float, stack: List[Shim], port_area_m2: float, hs_turns_out: float) -> float:
    k = shim_stack_stiffness(stack)
    opening_p = shim_opening_pressure_pa(stack, port_area_m2, hs_turns_out)
    if delta_p_pa <= opening_p:
        return 0.0
    force = (delta_p_pa - opening_p) * port_area_m2
    lift = force / max(k, 1e-6)
    return min(max(lift, 0.0), 0.60e-3)


def shim_curtain_area_m2(port_count: int, port_d_mm: float, lift_m: float) -> float:
    perimeter_each = pi * mm_to_m(port_d_mm)
    return port_count * perimeter_each * lift_m


def orifice_delta_p_pa(q_m3_s: float, area_m2: float, cd: float, rho: float) -> float:
    if area_m2 <= 1e-12:
        return 1e9
    return 0.5 * rho * (q_m3_s / (cd * area_m2)) ** 2


def solve_velocity_point(inp: SimulationInput, v_m_s: float, stack: List[Shim], direction: str) -> dict:
    q = displaced_flow_m3_s(inp, v_m_s)
    rho = inp.oil_density_kg_m3
    cd = inp.discharge_coefficient
    main_port_area = round_port_area_m2(inp.port_count, inp.port_diameter_mm)

    bleed_area = single_orifice_area_m2(inp.bleed_diameter_mm) * click_factor(inp.low_speed_clicks_out)
    rebound_area = single_orifice_area_m2(inp.rebound_orifice_diameter_mm) * click_factor(inp.low_speed_clicks_out)
    fixed_bypass_area = bleed_area if direction == "compression" else rebound_area

    delta_p = orifice_delta_p_pa(q, fixed_bypass_area + main_port_area * 0.15, cd, rho)

    for _ in range(40):
        lift = shim_lift_m(delta_p, stack, main_port_area, inp.high_speed_turns_out)
        curtain = shim_curtain_area_m2(inp.port_count, inp.port_diameter_mm, lift)
        effective_main_area = min(main_port_area, curtain + main_port_area * 0.08)

        total_area = fixed_bypass_area + effective_main_area
        new_delta_p = orifice_delta_p_pa(q, total_area, cd, rho)

        if abs(new_delta_p - delta_p) / max(delta_p, 1.0) < 1e-4:
            delta_p = new_delta_p
            break
        delta_p = 0.65 * delta_p + 0.35 * new_delta_p

    lift = shim_lift_m(delta_p, stack, main_port_area, inp.high_speed_turns_out)
    curtain = shim_curtain_area_m2(inp.port_count, inp.port_diameter_mm, lift)
    effective_main_area = min(main_port_area, curtain + main_port_area * 0.08)
    total_area = fixed_bypass_area + effective_main_area

    q_bypass = min(q, cd * fixed_bypass_area * sqrt(max(2 * delta_p / rho, 0.0)))
    q_main = max(q - q_bypass, 0.0)

    force_n = delta_p * piston_effective_area(inp)
    force_n = force_n if direction == "compression" else -force_n

    return {
        "velocity_m_s": v_m_s if direction == "compression" else -v_m_s,
        "pressure_pa": delta_p,
        "force_n": force_n,
        "shim_lift_mm": lift * 1000.0,
        "q_total_cc_s": q * 1e6,
        "flow_bypass_pct": 100.0 * q_bypass / max(q, 1e-12),
        "flow_main_pct": 100.0 * q_main / max(q, 1e-12),
        "effective_main_area_mm2": effective_main_area * 1e6,
        "fixed_bypass_area_mm2": fixed_bypass_area * 1e6,
        "port_total_area_mm2": main_port_area * 1e6,
    }


def air_spring_curve(inp: SimulationInput) -> dict:
    tube_area = single_orifice_area_m2(inp.tube_inner_diameter_mm)
    initial_gap_m = mm_to_m(inp.oil_height_mm)
    initial_volume = tube_area * initial_gap_m
    spring_forces = []
    travel_points = np.linspace(0.0, inp.travel_mm, 41)
    p1 = ATM_PA
    gamma = DEFAULT_GAMMA

    for travel_mm in travel_points:
        compressed_gap = max(initial_gap_m - mm_to_m(travel_mm), initial_gap_m * 0.08)
        v2 = tube_area * compressed_gap
        p2 = p1 * (initial_volume / v2) ** gamma
        air_force = (p2 - p1) * tube_area
        spring_force = (inp.preload_mm + travel_mm) * inp.spring_rate_n_per_mm
        spring_forces.append({
            "travel_mm": float(travel_mm),
            "air_force_n": float(air_force),
            "spring_force_n": float(spring_force),
            "combined_force_n": float(air_force + spring_force),
            "air_pressure_bar_abs": float(p2 / 1e5),
        })
    return {"curve": spring_forces}


def damping_ratio_estimate(inp: SimulationInput, comp_force_03: float, reb_force_03: float) -> dict:
    mass = 0.5 * (inp.rider_mass_kg + inp.bike_mass_kg)
    wheel_rate = inp.spring_rate_n_per_mm * 1000.0 / (inp.motion_ratio**2)
    c_crit = 2.0 * sqrt(max(wheel_rate * mass, 1.0))
    v = 0.3
    c_comp = abs(comp_force_03) / max(v, 1e-6)
    c_reb = abs(reb_force_03) / max(v, 1e-6)
    return {
        "critical_damping_n_s_m": c_crit,
        "compression_zeta": c_comp / c_crit,
        "rebound_zeta": c_reb / c_crit,
    }


@app.get("/")
def root() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.post("/api/simulate")
def simulate(inp: SimulationInput) -> JSONResponse:
    velocities = np.linspace(0.01, inp.shaft_velocity_max_m_s, inp.steps)
    compression = [solve_velocity_point(inp, float(v), inp.compression_stack, "compression") for v in velocities]
    rebound = [solve_velocity_point(inp, float(v), inp.rebound_stack, "rebound") for v in velocities]
    spring_curve = air_spring_curve(inp)

    comp_force_03 = min(compression, key=lambda x: abs(x["velocity_m_s"] - 0.3))["force_n"]
    reb_force_03 = min(rebound, key=lambda x: abs(abs(x["velocity_m_s"]) - 0.3))["force_n"]
    ratios = damping_ratio_estimate(inp, comp_force_03, reb_force_03)

    port_area_mm2 = round_port_area_m2(inp.port_count, inp.port_diameter_mm) * 1e6
    result = {
        "summary": {
            "piston_effective_area_mm2": piston_effective_area(inp) * 1e6,
            "total_port_area_mm2": port_area_mm2,
            "bleed_area_mm2": single_orifice_area_m2(inp.bleed_diameter_mm) * 1e6,
            "rebound_orifice_area_mm2": single_orifice_area_m2(inp.rebound_orifice_diameter_mm) * 1e6,
            "initial_gas_pressure_bar": inp.initial_gas_pressure_bar,
            **ratios,
        },
        "compression": compression,
        "rebound": rebound,
        "spring": spring_curve["curve"],
    }
    return JSONResponse(result)
