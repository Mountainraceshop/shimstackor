from __future__ import annotations

import csv
import io
import json
import os
import secrets
import time
from base64 import b64encode
from math import exp, pi, sqrt
from threading import Lock
from typing import Literal
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


app = FastAPI(title="Shim Calculator Engineering MVP")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


STEEL_E = 210e9
POISSON = 0.30
DEFAULT_GAMMA = 1.4
ATM_PA = 101325.0
PAYMENT_AMOUNT_USD = "100.00"
PAYMENT_CURRENCY = "USD"

PAYMENT_LOCK = Lock()
PENDING_PAYMENTS: dict[str, dict] = {}
PAID_PAYMENTS: dict[str, dict] = {}


class Shim(BaseModel):
    diameter_mm: float = Field(..., gt=0)
    thickness_mm: float = Field(..., gt=0)
    qty: int = Field(1, ge=1, le=20)


class SimulationInput(BaseModel):
    # Backward compatibility: frontend may still send `mode`.
    mode: Literal["fork", "shock"] = "fork"
    machine_type: Literal["fork", "shock"] | None = None
    fork_platform: Literal["coil", "kyb_psf1", "kyb_psf2", "showa_air", "wp_air"] = "coil"
    shock_architecture: Literal["monotube", "twin_tube"] = "monotube"

    piston_diameter_mm: float = Field(24.0, gt=1)
    shaft_diameter_mm: float = Field(12.0, gt=0)
    port_count: int = Field(4, ge=1, le=24)
    port_diameter_mm: float = Field(6.0, gt=0)
    base_port_count: int = Field(4, ge=1, le=24)
    base_port_diameter_mm: float = Field(5.8, gt=0)
    mid_port_count: int = Field(3, ge=1, le=24)
    mid_port_diameter_mm: float = Field(4.8, gt=0)

    bleed_diameter_mm: float = Field(0.9, gt=0)
    rebound_orifice_diameter_mm: float = Field(1.2, gt=0)
    low_speed_clicks_out: int = Field(12, ge=0, le=30)
    rebound_clicks_out: int = Field(12, ge=0, le=30)
    high_speed_turns_out: float = Field(1.5, ge=0, le=6)
    hsc_poppet_diameter_mm: float = Field(6.0, gt=0)
    hsc_spring_rate_n_per_mm: float = Field(18.0, gt=0)
    hsc_preload_mm: float = Field(1.0, ge=0)
    hsc_max_lift_mm: float = Field(1.0, gt=0)

    clamp_diameter_mm: float = Field(8.0, gt=0.1)
    include_base_valve: bool = True
    include_mid_valve: bool = True

    oil_density_kg_m3: float = Field(850.0, gt=100)
    oil_viscosity_cst: float = Field(16.0, gt=1)
    oil_temperature_c: float = Field(35.0, ge=-20, le=180)
    viscosity_temp_sensitivity: float = Field(0.022, gt=0.001, lt=0.2)
    discharge_coefficient: float = Field(0.70, gt=0.1, lt=1.0)
    vapor_pressure_bar_abs: float = Field(0.03, gt=0.005, lt=2.0)

    spring_rate_n_per_mm: float = Field(4.6, ge=0)
    preload_mm: float = Field(5.0, ge=0)
    travel_mm: float = Field(300.0, gt=1)
    motion_ratio: float = Field(1.0, gt=0.1)
    rider_mass_kg: float = Field(85.0, gt=1)
    bike_mass_kg: float = Field(110.0, gt=1)

    # Chassis measurement / geometry inputs
    wheelbase_mm: float = Field(1480.0, gt=500)
    steering_head_angle_deg: float = Field(24.0, gt=10, lt=40)  # rake from vertical
    triple_clamp_offset_mm: float = Field(32.0, gt=0, lt=100)
    front_tire_radius_mm: float = Field(340.0, gt=100)
    rear_tire_radius_mm: float = Field(340.0, gt=100)
    front_ride_height_mm: float = Field(0.0, ge=-100, le=100)
    rear_ride_height_mm: float = Field(0.0, ge=-100, le=100)
    swingarm_length_mm: float = Field(610.0, gt=200)
    swingarm_pivot_height_mm: float = Field(360.0, gt=100)
    rear_axle_height_mm: float = Field(330.0, gt=100)
    countershaft_height_mm: float = Field(350.0, gt=100)
    countershaft_to_rear_axle_mm: float = Field(580.0, gt=100)
    cg_height_mm: float = Field(600.0, gt=100)
    cg_from_front_axle_mm: float = Field(780.0, gt=0)

    # Fork spring / chamber inputs
    oil_height_mm: float = Field(110.0, gt=1)
    tube_inner_diameter_mm: float = Field(48.0, gt=1)
    fork_positive_air_bar_abs: float = Field(8.0, gt=0.5)
    fork_negative_air_bar_abs: float = Field(6.0, gt=0.3)
    fork_secondary_volume_cc: float = Field(55.0, gt=1)

    # Shock reservoir / gas inputs
    initial_gas_pressure_bar: float = Field(6.0, gt=0.1)
    reservoir_volume_cc: float = Field(120.0, gt=5)

    shaft_velocity_max_m_s: float = Field(1.0, gt=0.05, le=5)
    steps: int = Field(41, ge=11, le=401)
    payment_token: str | None = None
    compression_stack: list[Shim] = Field(
        default_factory=lambda: [
            Shim(diameter_mm=24, thickness_mm=0.15, qty=6),
            Shim(diameter_mm=22, thickness_mm=0.15, qty=2),
            Shim(diameter_mm=20, thickness_mm=0.10, qty=1),
        ]
    )
    rebound_stack: list[Shim] = Field(
        default_factory=lambda: [
            Shim(diameter_mm=22, thickness_mm=0.15, qty=5),
            Shim(diameter_mm=20, thickness_mm=0.15, qty=2),
            Shim(diameter_mm=18, thickness_mm=0.10, qty=1),
        ]
    )
    base_valve_stack: list[Shim] = Field(
        default_factory=lambda: [
            Shim(diameter_mm=22, thickness_mm=0.15, qty=4),
            Shim(diameter_mm=20, thickness_mm=0.15, qty=2),
        ]
    )
    mid_valve_stack: list[Shim] = Field(
        default_factory=lambda: [
            Shim(diameter_mm=20, thickness_mm=0.10, qty=3),
            Shim(diameter_mm=18, thickness_mm=0.10, qty=2),
        ]
    )


class DynoCsvInput(BaseModel):
    csv_text: str = Field(..., min_length=5)


class CreateOrderRequest(BaseModel):
    customer_reference: str | None = None


class CaptureOrderRequest(BaseModel):
    order_id: str = Field(..., min_length=5)
    payment_token: str = Field(..., min_length=8)


class ValveStage(BaseModel):
    name: str
    port_count: int
    port_diameter_mm: float
    stack: list[Shim]
    area_multiplier: float = 1.0
    hs_multiplier: float = 1.0


def mm_to_m(v: float) -> float:
    return v / 1000.0


def bar_to_pa(v: float) -> float:
    return v * 1e5


def round_port_area_m2(count: int, diameter_mm: float) -> float:
    return count * pi * (mm_to_m(diameter_mm) ** 2) / 4.0


def single_orifice_area_m2(diameter_mm: float) -> float:
    return pi * (mm_to_m(diameter_mm) ** 2) / 4.0


def resolved_machine_type(inp: SimulationInput) -> str:
    return inp.machine_type or inp.mode


def piston_effective_area(inp: SimulationInput) -> float:
    piston_area = single_orifice_area_m2(inp.piston_diameter_mm)
    shaft_area = single_orifice_area_m2(inp.shaft_diameter_mm)
    return piston_area - shaft_area if resolved_machine_type(inp) == "shock" else piston_area


def displaced_flow_m3_s(inp: SimulationInput, shaft_velocity_m_s: float) -> float:
    return abs(shaft_velocity_m_s) * piston_effective_area(inp)


def click_factor(clicks_out: int, max_clicks: int = 30) -> float:
    return 0.12 + 0.88 * min(max(clicks_out, 0), max_clicks) / max_clicks


def hs_factor(turns_out: float, max_turns: float = 6.0) -> float:
    open_factor = min(max(turns_out, 0.0), max_turns) / max_turns
    return 1.3 - 0.75 * open_factor


def temperature_corrected_cd(inp: SimulationInput) -> float:
    # Coarse correction: lower viscosity at higher temp effectively increases Cd.
    mu_ref = inp.oil_viscosity_cst
    mu_t = mu_ref * exp(-inp.viscosity_temp_sensitivity * (inp.oil_temperature_c - 40.0))
    visc_factor = (mu_ref / max(mu_t, 1e-6)) ** 0.14
    return min(max(inp.discharge_coefficient * visc_factor, 0.30), 0.96)


def shim_stack_stiffness(stack: list[Shim], clamp_d_mm: float) -> float:
    k_total = 0.0
    clamp_r = mm_to_m(clamp_d_mm) / 2.0
    for shim in stack:
        active_r = max(mm_to_m(shim.diameter_mm) / 2.0 - clamp_r, 1e-4)
        t = mm_to_m(shim.thickness_mm)
        single_k = (STEEL_E * t**3) / (12.0 * (1 - POISSON**2) * active_r**3)
        k_total += single_k * shim.qty
    return max(k_total, 1e-6)


def shim_opening_pressure_pa(
    stack: list[Shim], port_area_m2: float, hs_turns_out: float, clamp_d_mm: float, hs_multiplier: float
) -> float:
    k = shim_stack_stiffness(stack, clamp_d_mm)
    nominal_lift = 0.08e-3
    preload_force = k * nominal_lift * hs_factor(hs_turns_out) * hs_multiplier
    return max(preload_force / max(port_area_m2, 1e-9), 4e3)


def shim_lift_m(
    delta_p_pa: float, stack: list[Shim], port_area_m2: float, hs_turns_out: float, clamp_d_mm: float, hs_multiplier: float
) -> float:
    k = shim_stack_stiffness(stack, clamp_d_mm)
    opening_p = shim_opening_pressure_pa(stack, port_area_m2, hs_turns_out, clamp_d_mm, hs_multiplier)
    if delta_p_pa <= opening_p:
        return 0.0
    force = (delta_p_pa - opening_p) * port_area_m2
    lift = force / max(k, 1e-6)
    return min(max(lift, 0.0), 0.90e-3)


def shim_curtain_area_m2(port_count: int, port_d_mm: float, lift_m: float) -> float:
    perimeter_each = pi * mm_to_m(port_d_mm)
    return port_count * perimeter_each * lift_m


def orifice_delta_p_pa(q_m3_s: float, area_m2: float, cd: float, rho: float) -> float:
    if area_m2 <= 1e-12:
        return 1e9
    return 0.5 * rho * (q_m3_s / (cd * area_m2)) ** 2


def flow_from_delta_p_pa(delta_p_pa: float, area_m2: float, cd: float, rho: float) -> float:
    if area_m2 <= 1e-12 or delta_p_pa <= 0:
        return 0.0
    return cd * area_m2 * sqrt(2.0 * delta_p_pa / rho)


def hsc_poppet_area_m2(inp: SimulationInput, delta_p_pa: float, direction: str) -> float:
    if direction != "compression":
        return 0.0
    seat_area = single_orifice_area_m2(inp.hsc_poppet_diameter_mm)
    k_spring_n_m = inp.hsc_spring_rate_n_per_mm * 1000.0
    preload_n = inp.hsc_preload_mm * inp.hsc_spring_rate_n_per_mm * hs_factor(inp.high_speed_turns_out)
    threshold_pa = preload_n / max(seat_area, 1e-9)
    if delta_p_pa <= threshold_pa:
        return 0.0
    force_excess_n = (delta_p_pa - threshold_pa) * seat_area
    lift_m = min(force_excess_n / max(k_spring_n_m, 1e-6), mm_to_m(inp.hsc_max_lift_mm))
    circumference = pi * mm_to_m(inp.hsc_poppet_diameter_mm)
    return circumference * lift_m


def build_stages(inp: SimulationInput, direction: str) -> list[ValveStage]:
    machine = resolved_machine_type(inp)
    stages: list[ValveStage] = [
        ValveStage(
            name="piston",
            port_count=inp.port_count,
            port_diameter_mm=inp.port_diameter_mm,
            stack=inp.compression_stack if direction == "compression" else inp.rebound_stack,
            area_multiplier=1.0,
            hs_multiplier=1.0,
        )
    ]
    if inp.include_base_valve:
        stages.append(
            ValveStage(
                name="base",
                port_count=inp.base_port_count,
                port_diameter_mm=inp.base_port_diameter_mm,
                stack=inp.base_valve_stack,
                area_multiplier=0.95,
                hs_multiplier=1.08,
            )
        )
    if inp.include_mid_valve:
        stages.append(
            ValveStage(
                name="mid",
                port_count=inp.mid_port_count,
                port_diameter_mm=inp.mid_port_diameter_mm,
                stack=inp.mid_valve_stack,
                area_multiplier=0.90,
                hs_multiplier=1.15,
            )
        )

    if machine == "shock" and inp.shock_architecture == "twin_tube":
        for stage in stages:
            if stage.name == "base":
                stage.area_multiplier *= 1.08
            if stage.name == "piston":
                stage.area_multiplier *= 1.22 if direction == "compression" else 0.95
    return stages


def stage_dynamic_area_m2(inp: SimulationInput, stage: ValveStage, delta_p_pa: float) -> tuple[float, float]:
    raw_port_area = round_port_area_m2(stage.port_count, stage.port_diameter_mm) * stage.area_multiplier
    lift = shim_lift_m(
        delta_p_pa,
        stage.stack,
        raw_port_area,
        inp.high_speed_turns_out,
        inp.clamp_diameter_mm,
        stage.hs_multiplier,
    )
    curtain = shim_curtain_area_m2(stage.port_count, stage.port_diameter_mm, lift)
    effective_area = min(raw_port_area, raw_port_area * 0.07 + curtain)
    return max(effective_area, 1e-12), lift


def solve_series_flow_for_dp(
    inp: SimulationInput, stages: list[ValveStage], target_dp_pa: float, cd: float, rho: float
) -> tuple[float, list[dict]]:
    def series_dp_for_q(q_guess: float) -> tuple[float, list[dict]]:
        total_dp = 0.0
        stage_data = []
        initial_stage_dp = target_dp_pa / max(len(stages), 1)
        for stage in stages:
            dp_stage = initial_stage_dp
            area = 1e-12
            lift = 0.0
            for _ in range(8):
                area, lift = stage_dynamic_area_m2(inp, stage, dp_stage)
                dp_next = orifice_delta_p_pa(q_guess, area, cd, rho)
                if abs(dp_next - dp_stage) <= max(dp_next, 1.0) * 1e-3:
                    dp_stage = dp_next
                    break
                dp_stage = 0.5 * (dp_stage + dp_next)
            total_dp += dp_stage
            stage_data.append({
                "name": stage.name,
                "area_m2": area,
                "lift_m": lift,
                "dp_pa": dp_stage,
            })
        return total_dp, stage_data

    max_area = sum(round_port_area_m2(s.port_count, s.port_diameter_mm) * s.area_multiplier for s in stages)
    q_hi = flow_from_delta_p_pa(target_dp_pa, max(max_area, 1e-12), cd, rho)
    lo, hi = 0.0, max(q_hi * 2.2, 1e-8)
    best_stage_data: list[dict] = []
    for _ in range(42):
        mid = 0.5 * (lo + hi)
        dp_mid, stage_data = series_dp_for_q(mid)
        best_stage_data = stage_data
        if dp_mid > target_dp_pa:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) <= 1e-10:
            break
    return 0.5 * (lo + hi), best_stage_data


def reservoir_pressure_bar_abs(inp: SimulationInput, stroke_mm: float) -> float:
    p1 = bar_to_pa(inp.initial_gas_pressure_bar)
    v1 = inp.reservoir_volume_cc * 1e-6
    shaft_area = single_orifice_area_m2(inp.shaft_diameter_mm)
    displaced = shaft_area * mm_to_m(max(stroke_mm, 0.0))
    v2 = max(v1 - displaced, 0.2 * v1)
    p2 = p1 * (v1 / v2) ** DEFAULT_GAMMA
    return p2 / 1e5


def solve_velocity_point(inp: SimulationInput, v_m_s: float, direction: str) -> dict:
    q_required = displaced_flow_m3_s(inp, v_m_s)
    rho = inp.oil_density_kg_m3
    cd = temperature_corrected_cd(inp)
    stages = build_stages(inp, direction)

    bleed_area = single_orifice_area_m2(inp.bleed_diameter_mm) * click_factor(inp.low_speed_clicks_out)
    rebound_area = single_orifice_area_m2(inp.rebound_orifice_diameter_mm) * click_factor(inp.rebound_clicks_out)
    bypass_base_area = bleed_area if direction == "compression" else rebound_area

    lo_dp, hi_dp = 500.0, 3.0e7
    converged_stage_data: list[dict] = []
    converged_q_shim = 0.0
    converged_q_bypass = 0.0
    converged_bypass_area = bypass_base_area
    dp = 0.0

    for _ in range(55):
        dp = 0.5 * (lo_dp + hi_dp)
        hsc_area = hsc_poppet_area_m2(inp, dp, direction)
        bypass_area = bypass_base_area + hsc_area
        q_bypass = flow_from_delta_p_pa(dp, bypass_area, cd, rho)
        q_shim, stage_data = solve_series_flow_for_dp(inp, stages, dp, cd, rho)
        q_total = q_bypass + q_shim

        converged_stage_data = stage_data
        converged_q_shim = q_shim
        converged_q_bypass = q_bypass
        converged_bypass_area = bypass_area

        if q_total > q_required:
            hi_dp = dp
        else:
            lo_dp = dp
        if abs(hi_dp - lo_dp) <= 1.0:
            break

    main_lift_mm = 0.0
    for stage in converged_stage_data:
        if stage["name"] == "piston":
            main_lift_mm = stage["lift_m"] * 1000.0
            break

    total_flow = max(converged_q_shim + converged_q_bypass, 1e-12)
    machine = resolved_machine_type(inp)
    stroke_probe_mm = min(inp.travel_mm * 0.60, 180.0)
    reservoir_abs_bar = reservoir_pressure_bar_abs(inp, stroke_probe_mm)
    if machine == "shock" and inp.shock_architecture == "twin_tube":
        reservoir_abs_bar *= 1.04
    cav_ref_bar = max(reservoir_abs_bar - (dp / 1e5), 0.0)
    cav_margin_bar = cav_ref_bar - inp.vapor_pressure_bar_abs

    force_n = dp * piston_effective_area(inp)
    force_n = force_n if direction == "compression" else -force_n

    return {
        "velocity_m_s": v_m_s if direction == "compression" else -v_m_s,
        "pressure_pa": dp,
        "force_n": force_n,
        "shim_lift_mm": main_lift_mm,
        "q_total_cc_s": q_required * 1e6,
        "flow_bypass_pct": 100.0 * converged_q_bypass / total_flow,
        "flow_main_pct": 100.0 * converged_q_shim / total_flow,
        "effective_main_area_mm2": converged_stage_data[0]["area_m2"] * 1e6 if converged_stage_data else 0.0,
        "fixed_bypass_area_mm2": bypass_base_area * 1e6,
        "hsc_area_mm2": (converged_bypass_area - bypass_base_area) * 1e6,
        "port_total_area_mm2": round_port_area_m2(inp.port_count, inp.port_diameter_mm) * 1e6,
        "stage_pressures_pa": {s["name"]: s["dp_pa"] for s in converged_stage_data},
        "reservoir_pressure_bar_abs": reservoir_abs_bar,
        "cavitation_margin_bar": cav_margin_bar,
    }


def fork_spring_curve(inp: SimulationInput) -> list[dict]:
    area = single_orifice_area_m2(inp.tube_inner_diameter_mm)
    travel_points = np.linspace(0.0, inp.travel_mm, 41)
    curve = []

    v_pos0 = area * mm_to_m(inp.oil_height_mm)
    v_neg0 = max(inp.fork_secondary_volume_cc * 1e-6, area * 0.010)
    p_pos0 = bar_to_pa(inp.fork_positive_air_bar_abs)
    p_neg0 = bar_to_pa(inp.fork_negative_air_bar_abs)

    for travel_mm in travel_points:
        disp = area * mm_to_m(travel_mm)
        v_pos = max(v_pos0 - disp, 0.1 * v_pos0)
        p_pos = p_pos0 * (v_pos0 / v_pos) ** DEFAULT_GAMMA
        p_neg = ATM_PA

        if inp.fork_platform == "coil":
            p_pos = ATM_PA * (v_pos0 / v_pos) ** DEFAULT_GAMMA
            p_neg = ATM_PA
        elif inp.fork_platform == "kyb_psf1":
            p_neg = ATM_PA
        elif inp.fork_platform == "kyb_psf2":
            v_neg = v_neg0 + disp * 0.90
            p_neg = p_neg0 * (v_neg0 / v_neg) ** DEFAULT_GAMMA
        elif inp.fork_platform == "showa_air":
            v_bal = v_neg0 + disp * 0.45
            p_bal = p_neg0 * (v_neg0 / v_bal) ** DEFAULT_GAMMA
            p_neg = ATM_PA + 0.55 * (p_bal - ATM_PA)
        elif inp.fork_platform == "wp_air":
            v_neg = v_neg0 + disp * 0.60
            p_neg_raw = p_neg0 * (v_neg0 / v_neg) ** DEFAULT_GAMMA
            transfer = exp(-travel_mm / 180.0)
            p_neg = ATM_PA + transfer * (p_neg_raw - ATM_PA)

        air_force = max((p_pos - p_neg) * area, 0.0)
        spring_force = (inp.preload_mm + travel_mm) * inp.spring_rate_n_per_mm
        curve.append(
            {
                "travel_mm": float(travel_mm),
                "air_force_n": float(air_force),
                "spring_force_n": float(spring_force),
                "combined_force_n": float(air_force + spring_force),
                "air_pressure_bar_abs": float(p_pos / 1e5),
                "negative_pressure_bar_abs": float(p_neg / 1e5),
            }
        )
    return curve


def shock_spring_curve(inp: SimulationInput) -> list[dict]:
    travel_points = np.linspace(0.0, inp.travel_mm, 41)
    curve = []
    shaft_area = single_orifice_area_m2(inp.shaft_diameter_mm)
    p1 = bar_to_pa(inp.initial_gas_pressure_bar)
    v1 = inp.reservoir_volume_cc * 1e-6
    gas_scale = 0.62 if inp.shock_architecture == "twin_tube" else 1.0

    for travel_mm in travel_points:
        displaced = shaft_area * mm_to_m(travel_mm)
        v2 = max(v1 - displaced, 0.22 * v1)
        p2 = p1 * (v1 / v2) ** DEFAULT_GAMMA
        gas_force = max((p2 - p1) * shaft_area * gas_scale, 0.0)
        spring_force = (inp.preload_mm + travel_mm) * inp.spring_rate_n_per_mm
        curve.append(
            {
                "travel_mm": float(travel_mm),
                "air_force_n": float(gas_force),
                "spring_force_n": float(spring_force),
                "combined_force_n": float(gas_force + spring_force),
                "air_pressure_bar_abs": float(p2 / 1e5),
                "negative_pressure_bar_abs": 0.0,
            }
        )
    return curve


def spring_curve(inp: SimulationInput) -> list[dict]:
    return fork_spring_curve(inp) if resolved_machine_type(inp) == "fork" else shock_spring_curve(inp)


def damping_ratio_estimate(inp: SimulationInput, comp_force_03: float, reb_force_03: float) -> dict:
    mass = 0.5 * (inp.rider_mass_kg + inp.bike_mass_kg)
    wheel_rate = max(inp.spring_rate_n_per_mm * 1000.0 / (inp.motion_ratio**2), 1.0)
    c_crit = 2.0 * sqrt(max(wheel_rate * mass, 1.0))
    v = 0.3
    c_comp = abs(comp_force_03) / max(v, 1e-6)
    c_reb = abs(reb_force_03) / max(v, 1e-6)
    return {
        "critical_damping_n_s_m": c_crit,
        "compression_zeta": c_comp / c_crit,
        "rebound_zeta": c_reb / c_crit,
    }


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def chassis_measurement(inp: SimulationInput) -> dict:
    # Chassis pitch: positive => rear higher than front.
    pitch_rad = np.arctan2(inp.rear_ride_height_mm - inp.front_ride_height_mm, max(inp.wheelbase_mm, 1.0))
    pitch_deg = float(np.degrees(pitch_rad))

    # Effective rake from vertical with ride-height attitude included.
    rake_deg_eff = float(inp.steering_head_angle_deg + pitch_deg)
    rake_deg_eff = float(clamp(rake_deg_eff, 8.0, 45.0))
    rake_rad_eff = np.radians(rake_deg_eff)

    # Motorcycle convention proxy: trail ~= (R*sin(rake)-offset)/cos(rake), rake from vertical.
    trail_mm = (
        (inp.front_tire_radius_mm * np.sin(rake_rad_eff) - inp.triple_clamp_offset_mm)
        / max(np.cos(rake_rad_eff), 1e-6)
    )

    swingarm_angle_rad = np.arctan2(
        inp.rear_axle_height_mm - inp.swingarm_pivot_height_mm,
        max(inp.swingarm_length_mm, 1.0),
    )
    swingarm_angle_deg = float(np.degrees(swingarm_angle_rad))

    chain_angle_rad = np.arctan2(
        inp.rear_axle_height_mm - inp.countershaft_height_mm,
        max(inp.countershaft_to_rear_axle_mm, 1.0),
    )
    chain_angle_deg = float(np.degrees(chain_angle_rad))

    # Practical anti-squat proxy for setup guidance.
    denom = max(abs(np.tan(swingarm_angle_rad)), 0.05)
    geom_factor = max(inp.cg_height_mm / max(inp.rear_tire_radius_mm, 1.0), 0.6)
    anti_squat_pct = float(100.0 * (np.tan(chain_angle_rad) / denom) / geom_factor)
    anti_squat_pct = float(clamp(anti_squat_pct, 20.0, 180.0))

    cg_x = clamp(inp.cg_from_front_axle_mm, 1.0, max(inp.wheelbase_mm - 1.0, 2.0))
    rear_pct = 100.0 * cg_x / inp.wheelbase_mm
    front_pct = 100.0 - rear_pct

    wheel_rate_front_n_mm = inp.spring_rate_n_per_mm
    wheel_rate_rear_n_mm = inp.spring_rate_n_per_mm / (inp.motion_ratio**2)

    trail_state = "balanced"
    if trail_mm < 95:
        trail_state = "quick / nervous"
    elif trail_mm > 115:
        trail_state = "stable / slow-turning"

    anti_squat_state = "balanced"
    if anti_squat_pct < 85:
        anti_squat_state = "low anti-squat (more squat)"
    elif anti_squat_pct > 120:
        anti_squat_state = "high anti-squat (less squat / harsher drive)"

    return {
        "trail_mm": float(trail_mm),
        "effective_rake_deg": rake_deg_eff,
        "chassis_pitch_deg": pitch_deg,
        "swingarm_angle_deg": swingarm_angle_deg,
        "chain_angle_deg": chain_angle_deg,
        "anti_squat_pct_proxy": anti_squat_pct,
        "front_weight_pct": float(front_pct),
        "rear_weight_pct": float(rear_pct),
        "wheel_rate_front_n_mm": float(wheel_rate_front_n_mm),
        "wheel_rate_rear_n_mm": float(wheel_rate_rear_n_mm),
        "handling_flags": {
            "trail_balance": trail_state,
            "anti_squat_balance": anti_squat_state,
        },
    }


def parse_dyno_csv(csv_text: str) -> dict:
    fh = io.StringIO(csv_text.strip())
    sample = fh.read(1024)
    fh.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.get_dialect("excel")
    reader = csv.DictReader(fh, dialect=dialect)
    if not reader.fieldnames:
        raise ValueError("CSV must contain a header row.")

    fields = [f.strip() for f in reader.fieldnames]
    lower = [f.lower() for f in fields]

    vel_idx = next((i for i, name in enumerate(lower) if "vel" in name or "speed" in name), None)
    force_idx = next((i for i, name in enumerate(lower) if "force" in name or "load" in name or "damp" in name), None)
    dir_idx = next((i for i, name in enumerate(lower) if "dir" in name or "stroke" in name), None)
    if vel_idx is None or force_idx is None:
        raise ValueError("CSV needs velocity and force columns.")

    vel_col = fields[vel_idx]
    force_col = fields[force_idx]
    dir_col = fields[dir_idx] if dir_idx is not None else None
    vel_key = reader.fieldnames[vel_idx]
    force_key = reader.fieldnames[force_idx]
    dir_key = reader.fieldnames[dir_idx] if dir_idx is not None else None

    compression = []
    rebound = []
    for row in reader:
        try:
            vel = float(str(row[vel_key]).strip())
            force = float(str(row[force_key]).strip())
        except (TypeError, ValueError):
            continue
        if dir_key:
            d = str(row.get(dir_key, "")).strip().lower()
            if "reb" in d:
                rebound.append({"velocity_m_s": -abs(vel), "force_n": -abs(force)})
            elif "comp" in d:
                compression.append({"velocity_m_s": abs(vel), "force_n": abs(force)})
            else:
                (compression if vel >= 0 else rebound).append({"velocity_m_s": vel, "force_n": force})
        else:
            if vel >= 0:
                compression.append({"velocity_m_s": vel, "force_n": force})
            else:
                rebound.append({"velocity_m_s": vel, "force_n": force})

    compression.sort(key=lambda x: x["velocity_m_s"])
    rebound.sort(key=lambda x: x["velocity_m_s"])
    return {
        "compression": compression,
        "rebound": rebound,
        "columns": {"velocity": vel_col, "force": force_col, "direction": dir_col},
        "points": len(compression) + len(rebound),
    }


def payment_gate_bypass() -> bool:
    return os.getenv("SIMULATOR_BYPASS_PAYMENT", "").strip().lower() in {"1", "true", "yes", "on"}


def paypal_settings() -> dict:
    client_id = os.getenv("PAYPAL_CLIENT_ID", "").strip()
    client_secret = os.getenv("PAYPAL_CLIENT_SECRET", "").strip()
    api_base = os.getenv("PAYPAL_API_BASE", "https://api-m.sandbox.paypal.com").strip().rstrip("/")
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "api_base": api_base,
        "enabled": bool(client_id and client_secret),
    }


def _http_json(method: str, url: str, headers: dict[str, str], payload: dict | None = None) -> dict:
    data = None
    req_headers = dict(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urlrequest.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urlrequest.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"PayPal HTTP {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"PayPal connection error: {exc.reason}") from exc


def paypal_access_token(settings: dict) -> str:
    creds = f"{settings['client_id']}:{settings['client_secret']}".encode("utf-8")
    auth = b64encode(creds).decode("ascii")
    form_data = urlparse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
    req = urlrequest.Request(
        f"{settings['api_base']}/v1/oauth2/token",
        data=form_data,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
            body = json.loads(raw)
            token = body.get("access_token", "")
            if not token:
                raise RuntimeError("PayPal token response did not include an access token.")
            return token
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"PayPal token request failed ({exc.code}): {detail}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"PayPal token request connection error: {exc.reason}") from exc


def cleanup_payment_state(max_age_sec: int = 24 * 3600) -> None:
    now = time.time()
    with PAYMENT_LOCK:
        stale_pending = [k for k, v in PENDING_PAYMENTS.items() if now - v["created_at"] > max_age_sec]
        stale_paid = [k for k, v in PAID_PAYMENTS.items() if now - v["created_at"] > max_age_sec]
        for key in stale_pending:
            PENDING_PAYMENTS.pop(key, None)
        for key in stale_paid:
            PAID_PAYMENTS.pop(key, None)


def register_pending_payment(order_id: str) -> str:
    token = secrets.token_urlsafe(24)
    with PAYMENT_LOCK:
        PENDING_PAYMENTS[token] = {
            "order_id": order_id,
            "created_at": time.time(),
        }
    return token


def payment_is_paid(payment_token: str | None) -> bool:
    if not payment_token:
        return False
    with PAYMENT_LOCK:
        paid = PAID_PAYMENTS.get(payment_token)
        return bool(paid and paid.get("remaining_uses", 0) > 0)


def mark_payment_paid(payment_token: str, order_id: str, capture_id: str) -> None:
    with PAYMENT_LOCK:
        pending = PENDING_PAYMENTS.pop(payment_token, None)
        created_at = pending["created_at"] if pending else time.time()
        PAID_PAYMENTS[payment_token] = {
            "order_id": order_id,
            "capture_id": capture_id,
            "created_at": created_at,
            "remaining_uses": 1,
        }


def get_pending_payment(payment_token: str) -> dict | None:
    with PAYMENT_LOCK:
        return PENDING_PAYMENTS.get(payment_token)


def consume_paid_token(payment_token: str) -> int:
    with PAYMENT_LOCK:
        paid = PAID_PAYMENTS.get(payment_token)
        if not paid:
            return 0
        remaining = int(paid.get("remaining_uses", 0))
        if remaining <= 0:
            return 0
        remaining -= 1
        paid["remaining_uses"] = remaining
        return remaining


def create_paypal_order(settings: dict, customer_reference: str | None = None) -> dict:
    access_token = paypal_access_token(settings)
    payload = {
        "intent": "CAPTURE",
        "purchase_units": [
            {
                "amount": {
                    "currency_code": PAYMENT_CURRENCY,
                    "value": PAYMENT_AMOUNT_USD,
                },
                "description": "Shim Calculator Engineering single analysis unlock",
            }
        ],
        "application_context": {
            "shipping_preference": "NO_SHIPPING",
            "user_action": "PAY_NOW",
        },
    }
    if customer_reference:
        payload["purchase_units"][0]["custom_id"] = customer_reference[:127]
    return _http_json(
        "POST",
        f"{settings['api_base']}/v2/checkout/orders",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
        payload=payload,
    )


def capture_paypal_order(settings: dict, order_id: str) -> dict:
    access_token = paypal_access_token(settings)
    return _http_json(
        "POST",
        f"{settings['api_base']}/v2/checkout/orders/{order_id}/capture",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
        payload={},
    )


def validate_capture_result(capture_response: dict) -> tuple[bool, str]:
    if capture_response.get("status") != "COMPLETED":
        return False, "PayPal order was not completed."
    purchase_units = capture_response.get("purchase_units") or []
    if not purchase_units:
        return False, "PayPal capture did not include purchase units."
    captures = (((purchase_units[0] or {}).get("payments") or {}).get("captures")) or []
    if not captures:
        return False, "PayPal capture payload did not include captures."
    capture = captures[0] or {}
    amount = capture.get("amount") or {}
    if capture.get("status") != "COMPLETED":
        return False, "PayPal capture status is not COMPLETED."
    if amount.get("currency_code") != PAYMENT_CURRENCY or amount.get("value") != PAYMENT_AMOUNT_USD:
        return False, "Payment amount mismatch."
    capture_id = capture.get("id") or ""
    if not capture_id:
        return False, "Missing capture ID in PayPal response."
    return True, capture_id


def stack_to_rows(stack: list[Shim]) -> list[dict]:
    return [
        {
            "diameter_mm": s.diameter_mm,
            "thickness_mm": s.thickness_mm,
            "qty": s.qty,
        }
        for s in stack
    ]


def recommended_stack(stack: list[Shim], direction: str, zeta: float) -> tuple[list[dict], str]:
    target = 0.32 if direction == "compression" else 0.75
    delta = zeta - target
    note = f"{direction.title()} zeta near target."
    if not stack:
        return [], "No input stack provided."

    rec = [
        {
            "diameter_mm": s.diameter_mm,
            "thickness_mm": s.thickness_mm,
            "qty": s.qty,
        }
        for s in stack
    ]

    if delta > 0.08:
        rec[0]["qty"] = max(1, rec[0]["qty"] - 1)
        note = f"{direction.title()} damping appears high; reduce top shim quantity by 1."
    elif delta < -0.08:
        rec[0]["qty"] = min(20, rec[0]["qty"] + 1)
        note = f"{direction.title()} damping appears low; increase top shim quantity by 1."
    return rec, note


@app.get("/")
def root() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/api/public-config")
def public_config() -> JSONResponse:
    settings = paypal_settings()
    return JSONResponse(
        {
            "paypal_client_id": settings["client_id"],
            "payment_amount_usd": PAYMENT_AMOUNT_USD,
            "payment_currency": PAYMENT_CURRENCY,
            "payment_enabled": settings["enabled"],
            "payment_bypass": payment_gate_bypass(),
        }
    )


@app.post("/api/payment/create-order")
def payment_create_order(req: CreateOrderRequest) -> JSONResponse:
    cleanup_payment_state()
    settings = paypal_settings()
    if payment_gate_bypass():
        token = register_pending_payment("BYPASS_ORDER")
        mark_payment_paid(token, "BYPASS_ORDER", "BYPASS_CAPTURE")
        return JSONResponse(
            {
                "order_id": "BYPASS_ORDER",
                "payment_token": token,
                "currency": PAYMENT_CURRENCY,
                "amount": PAYMENT_AMOUNT_USD,
                "bypass": True,
            }
        )
    if not settings["enabled"]:
        return JSONResponse({"error": "PayPal is not configured on the server."}, status_code=503)
    try:
        created = create_paypal_order(settings, req.customer_reference)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)

    order_id = str(created.get("id") or "")
    if not order_id:
        return JSONResponse({"error": "PayPal order ID missing from response."}, status_code=502)

    payment_token = register_pending_payment(order_id)
    return JSONResponse(
        {
            "order_id": order_id,
            "payment_token": payment_token,
            "currency": PAYMENT_CURRENCY,
            "amount": PAYMENT_AMOUNT_USD,
            "bypass": False,
        }
    )


@app.post("/api/payment/capture-order")
def payment_capture_order(req: CaptureOrderRequest) -> JSONResponse:
    cleanup_payment_state()
    settings = paypal_settings()
    if payment_gate_bypass():
        mark_payment_paid(req.payment_token, req.order_id, "BYPASS_CAPTURE")
        return JSONResponse({"paid": True, "payment_token": req.payment_token, "capture_id": "BYPASS_CAPTURE"})

    pending = get_pending_payment(req.payment_token)
    if not pending:
        return JSONResponse({"error": "Unknown or expired payment token."}, status_code=400)
    if pending["order_id"] != req.order_id:
        return JSONResponse({"error": "Order ID does not match pending payment token."}, status_code=400)
    if not settings["enabled"]:
        return JSONResponse({"error": "PayPal is not configured on the server."}, status_code=503)
    try:
        capture_resp = capture_paypal_order(settings, req.order_id)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)

    valid, capture_or_error = validate_capture_result(capture_resp)
    if not valid:
        return JSONResponse({"error": capture_or_error}, status_code=400)

    mark_payment_paid(req.payment_token, req.order_id, capture_or_error)
    return JSONResponse({"paid": True, "payment_token": req.payment_token, "capture_id": capture_or_error})


@app.post("/api/parse-dyno-csv")
def parse_dyno(req: DynoCsvInput) -> JSONResponse:
    try:
        return JSONResponse(parse_dyno_csv(req.csv_text))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/simulate")
def simulate(inp: SimulationInput) -> JSONResponse:
    if not payment_gate_bypass() and not payment_is_paid(inp.payment_token):
        return JSONResponse(
            {
                "error": "Payment required. Complete USD 100.00 PayPal payment to unlock calculations.",
                "payment_required": True,
                "amount": PAYMENT_AMOUNT_USD,
                "currency": PAYMENT_CURRENCY,
            },
            status_code=402,
        )

    velocities = np.linspace(0.01, inp.shaft_velocity_max_m_s, inp.steps)
    compression = [solve_velocity_point(inp, float(v), "compression") for v in velocities]
    rebound = [solve_velocity_point(inp, float(v), "rebound") for v in velocities]
    spring = spring_curve(inp)

    comp_force_03 = min(compression, key=lambda x: abs(x["velocity_m_s"] - 0.3))["force_n"]
    reb_force_03 = min(rebound, key=lambda x: abs(abs(x["velocity_m_s"]) - 0.3))["force_n"]
    ratios = damping_ratio_estimate(inp, comp_force_03, reb_force_03)

    comp_03 = min(compression, key=lambda x: abs(x["velocity_m_s"] - 0.3))
    chassis = chassis_measurement(inp)
    comp_rec, comp_note = recommended_stack(inp.compression_stack, "compression", ratios["compression_zeta"])
    reb_rec, reb_note = recommended_stack(inp.rebound_stack, "rebound", ratios["rebound_zeta"])
    result = {
        "summary": {
            "machine_type": resolved_machine_type(inp),
            "fork_platform": inp.fork_platform,
            "shock_architecture": inp.shock_architecture,
            "piston_effective_area_mm2": piston_effective_area(inp) * 1e6,
            "total_port_area_mm2": round_port_area_m2(inp.port_count, inp.port_diameter_mm) * 1e6,
            "base_port_area_mm2": round_port_area_m2(inp.base_port_count, inp.base_port_diameter_mm) * 1e6,
            "mid_port_area_mm2": round_port_area_m2(inp.mid_port_count, inp.mid_port_diameter_mm) * 1e6,
            "bleed_area_mm2": single_orifice_area_m2(inp.bleed_diameter_mm) * 1e6,
            "rebound_orifice_area_mm2": single_orifice_area_m2(inp.rebound_orifice_diameter_mm) * 1e6,
            "temperature_corrected_cd": temperature_corrected_cd(inp),
            "initial_gas_pressure_bar": inp.initial_gas_pressure_bar,
            "reservoir_pressure_bar_abs_at_probe": comp_03["reservoir_pressure_bar_abs"],
            "cavitation_margin_bar_at_03": comp_03["cavitation_margin_bar"],
            **ratios,
        },
        "chassis": chassis,
        "stacks": {
            "input": {
                "compression": stack_to_rows(inp.compression_stack),
                "rebound": stack_to_rows(inp.rebound_stack),
            },
            "recommended": {
                "compression": comp_rec,
                "rebound": reb_rec,
            },
            "notes": [comp_note, reb_note],
        },
        "compression": compression,
        "rebound": rebound,
        "spring": spring,
    }
    if not payment_gate_bypass() and inp.payment_token:
        consume_paid_token(inp.payment_token)
    return JSONResponse(result)
