"""Synthetic utility-site data with embedded scenarios, for demonstrating the engine.

``generate_demo_data(input_path, days=7, seed=7, start="2026-10-01")`` writes
one CSV per source into ``<input_path>/<SOURCE>/`` (5-minute telemetry with
daily demand cycles, AR(1) noise and deliberately messy exports) and the
ground truth to ``<parent of input_path>/demo_ground_truth.json``.  The output
is fully deterministic for a given seed.

Files
-----
* ``TRIO/telemetry_site_a.csv``: ISO ``timestamp`` (UTC, "Z"), outlet_pressure_bar,
  inlet_pressure_bar, rssi_dbm.
* ``TCL/tank_pumps.csv``: messy headers ("Tank Level A (%)", "P2 Current (A)"),
  day-first "Time" (DD/MM/YYYY HH:MM), duplicated rows, re-sent rows with a
  corrected value, blank cells, sensor error markers, quoted numbers.
* ``Meters/zone_flow.csv``: "DateTime", zone_flow_lps, customer_bulk_m3h.
* ``AX/work_orders.csv``: created_at, work_order, asset, category, priority,
  description (routine entries plus the scenario ones).

Scenarios (day 1 is the start date; all times UTC)
--------------------------------------------------
1. Pump trip: day 3 14:00-17:00 P2 current is exactly 0.0; tank A falls 5 pct
   points/h during the outage and recovers by 23:00; TRIO outlet pressure drops
   0.45 bar from 14:10 to 17:10; zone flow dips 20% from 14:15 to 17:15; a
   work order "P2 tripped on overload, reset by operator" is logged at 16:00.
2. Stuck meter: customer_bulk_m3h frozen at one value, day 5 10:00-18:00.
3. Radio outage: no TRIO rows at all, day 4 10:00-12:00.
4. Slow sensor drift: tank level B rises 0.5 pct points/day from day 2 00:00.
5. Leak: zone flow gains a constant 25% of the pre-leak minimum night flow,
   ramping in between day 4 00:00 and 01:00 and persisting; a customer
   complains about low pressure on day 5 10:40.
6. Two isolated single-sample glitches in TRIO inlet pressure (day 2 09:35
   +1.8 bar, day 6 03:15 -2.2 bar).

Scenarios 1, 2, 3 and 6 are visible to the deterministic pipeline (flatline /
step / gap / spike events); 4 and 5 are trends that only show up in the
features and are left for the Mind to discover.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

INTERVAL_MINUTES = 5
DEFAULT_START = "2026-10-01"
DEFAULT_DAYS = 7
DEFAULT_SEED = 7
GROUND_TRUTH_FILE = "demo_ground_truth.json"

ROUTINE_WORK_ORDERS = [
    ("Tank A", "inspection", "low", "Routine inspection of Tank A level sensor and overflow"),
    ("Pump P1", "maintenance", "low", "Grease P1 bearings and check seal leak-off"),
    ("Zone 3 PRV", "maintenance", "medium", "Monthly PRV exercise and setpoint check"),
    ("Site A RTU", "maintenance", "low", "Replace RTU backup battery"),
    ("Outlet main", "calibration", "medium", "Calibrate outlet pressure transmitter against test gauge"),
    ("Site A", "sampling", "low", "Chlorine residual and turbidity sample"),
    ("Inlet screen", "maintenance", "low", "Clear debris from inlet screen"),
    ("Tank B", "inspection", "low", "Visual inspection of Tank B hatch seals"),
    ("Pump P2", "inspection", "low", "Vibration survey on P2 motor"),
    ("Site A", "security", "low", "Perimeter fence and gate lock check"),
    ("Zone 3 network", "inspection", "medium", "Hydrant flushing programme, Zone 3 north"),
    ("Pump P1", "inspection", "low", "Thermal imaging of P1 starter panel"),
    ("Site A RTU", "inspection", "low", "Antenna mount and cable inspection"),
]


def demand_profile(hour_of_day: np.ndarray) -> np.ndarray:
    """Diurnal demand multiplier: ~0.5 at 02:00, peaks ~1.24 around 08:00 and 19:00."""
    h = np.asarray(hour_of_day, dtype=float)
    return 1.0 + 0.3 * np.cos(2 * np.pi * (h - 13.5) / 24.0) + 0.2 * np.cos(4 * np.pi * (h - 8.0) / 24.0)


def _ar1(rng: np.random.Generator, n: int, sigma: float, phi: float = 0.6) -> np.ndarray:
    """Gaussian AR(1) noise with marginal std ``sigma``."""
    eps = rng.normal(0.0, sigma * np.sqrt(1.0 - phi * phi), n)
    out = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = phi * acc + eps[i]
        out[i] = acc
    return out


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).isoformat()


def _write_csv(path: str, header: list[str], rows: list[list[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(row) + "\n")


def generate_demo_data(input_path: str, days: int = DEFAULT_DAYS, seed: int = DEFAULT_SEED,
                       start: str = DEFAULT_START) -> dict:
    """Write the demo site into ``input_path`` and return (and save) the ground truth."""
    if days < 1:
        raise ValueError("days must be >= 1")
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    n = int(days * 24 * 60 / INTERVAL_MINUTES)
    t = pd.date_range(start_ts, periods=n, freq=f"{INTERVAL_MINUTES}min")
    end_ts = t[-1]
    hours = np.arange(n) * INTERVAL_MINUTES / 60.0
    hod = hours % 24.0
    days_f = hours / 24.0
    demand = demand_profile(hod) * (1.0 + 0.03 * np.sin(2 * np.pi * days_f / 7.0 + 0.5))

    def at(day: int, hour: int, minute: int = 0) -> pd.Timestamp:
        return start_ts + pd.Timedelta(days=day - 1, hours=hour, minutes=minute)

    def between(a: pd.Timestamp, b: pd.Timestamp) -> np.ndarray:
        return np.asarray((t >= a) & (t < b), dtype=bool)

    def ramp(points: list[tuple[pd.Timestamp, float]]) -> np.ndarray:
        xs = [(p - start_ts) / pd.Timedelta(hours=1) for p, _ in points]
        ys = [v for _, v in points]
        out = np.interp(hours, xs, ys)
        out[hours < xs[0]] = ys[0]
        out[hours > xs[-1]] = ys[-1]
        return out

    scenarios: list[dict] = []
    messiness: dict = {}

    # ------------------------------------------------------------------ TRIO
    outlet = 4.2 - 0.6 * (demand - 1.0) + _ar1(rng, n, 0.012)
    inlet = 5.6 - 0.25 * (demand - 1.0) + _ar1(rng, n, 0.010)
    rssi = np.round(-67.0 + 2.0 * np.cos(2 * np.pi * (hod - 15.0) / 24.0) + rng.normal(0.0, 1.3, n))

    # ------------------------------------------------------------------- TCL
    tank_a = 62.0 + 6.0 * np.cos(2 * np.pi * (hod - 6.0) / 24.0) + _ar1(rng, n, 0.08)
    tank_b = 48.0 + 5.0 * np.cos(2 * np.pi * (hod - 7.0) / 24.0) + _ar1(rng, n, 0.08)
    p1 = 6.8 + 1.3 * (demand - 1.0) + _ar1(rng, n, 0.07)
    p2 = 7.1 + 1.4 * (demand - 1.0) + _ar1(rng, n, 0.07)

    # ---------------------------------------------------------------- Meters
    flow = 30.0 * demand + _ar1(rng, n, 0.5)
    bulk = 8.0 * demand + _ar1(rng, n, 0.12)

    work_orders: list[tuple[pd.Timestamp, str, str, str, str]] = []

    # ---------------------------------------------------- scenario 1: pump trip
    trip_start, trip_end = at(3, 14, 0), at(3, 17, 0)
    if trip_start <= end_ts:
        trip = between(trip_start, trip_end)
        p2[trip] = 0.0
        tank_a += ramp([(trip_start, 0.0), (trip_end, -15.0), (at(3, 23, 0), 0.0)])
        outlet -= 0.45 * between(at(3, 14, 10), at(3, 17, 10))
        flow *= 1.0 - 0.2 * ramp([(at(3, 14, 15), 0.0), (at(3, 14, 25), 1.0), (at(3, 17, 15), 1.0), (at(3, 17, 25), 0.0)])
        wo_time = at(3, 16, 0)
        work_orders.append((wo_time, "Pump P2", "fault", "high", "P2 tripped on overload, reset by operator"))
        scenarios.append({
            "id": "pump_trip",
            "name": "Pump P2 trip with downstream effects",
            "start": _iso(trip_start), "end": _iso(trip_end),
            "detectable_by_pipeline": True,
            "description": "P2 current is exactly 0.0 for 3 hours; tank A drains, outlet pressure and zone flow drop shortly after; a work order is raised 2 hours later.",
            "effects": [
                {"dataset": "TCL/tank_pumps", "signal": "p2_current_a", "start": _iso(trip_start), "end": _iso(trip_end - pd.Timedelta(minutes=INTERVAL_MINUTES)),
                 "description": "current exactly 0.0 (36 samples)", "expected_pipeline_events": ["flatline", "step_change"]},
                {"dataset": "TCL/tank_pumps", "signal": "tank_level_a_pct", "start": _iso(trip_start), "end": _iso(at(3, 23, 0)),
                 "description": "level falls 5 points/h for 3 h then recovers by 23:00 (trend, no event expected)", "expected_pipeline_events": []},
                {"dataset": "TRIO/telemetry_site_a", "signal": "outlet_pressure_bar", "start": _iso(at(3, 14, 10)), "end": _iso(at(3, 17, 10)),
                 "description": "pressure 0.45 bar lower, 10 minutes after the trip", "expected_pipeline_events": ["step_change"]},
                {"dataset": "Meters/zone_flow", "signal": "zone_flow_lps", "start": _iso(at(3, 14, 15)), "end": _iso(at(3, 17, 15)),
                 "description": "flow 20% lower, 15 minutes after the trip", "expected_pipeline_events": ["step_change"]},
                {"dataset": "AX/work_orders", "signal": None, "start": _iso(wo_time), "end": _iso(wo_time),
                 "description": "work order: P2 tripped on overload, reset by operator", "expected_pipeline_events": []},
            ],
        })

    # -------------------------------------------------- scenario 2: stuck meter
    stuck_start, stuck_end = at(5, 10, 0), at(5, 18, 0)
    if stuck_start <= end_ts:
        stuck = between(stuck_start, stuck_end)
        frozen_value = round(float(bulk[stuck][0]), 2)
        bulk[stuck] = frozen_value
        scenarios.append({
            "id": "stuck_meter",
            "name": "Customer bulk meter frozen",
            "start": _iso(stuck_start), "end": _iso(stuck_end - pd.Timedelta(minutes=INTERVAL_MINUTES)),
            "detectable_by_pipeline": True,
            "description": f"customer_bulk_m3h reports exactly {frozen_value} for 8 hours while demand moves.",
            "effects": [
                {"dataset": "Meters/zone_flow", "signal": "customer_bulk_m3h", "start": _iso(stuck_start), "end": _iso(stuck_end - pd.Timedelta(minutes=INTERVAL_MINUTES)),
                 "description": f"frozen at {frozen_value} (96 samples)", "expected_pipeline_events": ["flatline"], "stuck_value": frozen_value},
            ],
        })

    # ------------------------------------------------- scenario 3: radio outage
    outage_start, outage_end = at(4, 10, 0), at(4, 12, 0)
    trio_keep = np.ones(n, dtype=bool)
    if outage_start <= end_ts:
        trio_keep = ~between(outage_start, outage_end)
        scenarios.append({
            "id": "radio_outage",
            "name": "TRIO radio outage",
            "start": _iso(outage_start), "end": _iso(outage_end - pd.Timedelta(minutes=INTERVAL_MINUTES)),
            "detectable_by_pipeline": True,
            "description": "no TRIO rows at all for 2 hours (last row 09:55, next row 12:00); every TRIO signal is offline.",
            "effects": [
                {"dataset": "TRIO/telemetry_site_a", "signal": "__all__", "start": _iso(outage_start), "end": _iso(outage_end - pd.Timedelta(minutes=INTERVAL_MINUTES)),
                 "description": "24 samples missing", "expected_pipeline_events": ["gap"]},
            ],
        })

    # ------------------------------------------------- scenario 4: sensor drift
    drift_start = at(2, 0, 0)
    if drift_start <= end_ts:
        drift_days = np.maximum(0.0, days_f - (drift_start - start_ts) / pd.Timedelta(days=1))
        tank_b += 0.5 * drift_days
        scenarios.append({
            "id": "sensor_drift",
            "name": "Tank B level sensor drifting upward",
            "start": _iso(drift_start), "end": _iso(end_ts),
            "detectable_by_pipeline": False,
            "description": "tank_level_b_pct gains 0.5 percentage points per day from day 2; no spikes or steps, only a trend.",
            "effects": [
                {"dataset": "TCL/tank_pumps", "signal": "tank_level_b_pct", "start": _iso(drift_start), "end": _iso(end_ts),
                 "description": "+0.5 points/day linear drift", "expected_pipeline_events": []},
            ],
        })

    # --------------------------------------------------------- scenario 5: leak
    leak_start = at(4, 1, 0)
    if leak_start <= end_ts:
        night = (hod >= 1.0) & (hod < 4.0) & (days_f < 3.0)
        mnf_before = float(np.min(flow[night])) if night.any() else float(np.min(flow))
        leak_lps = round(0.25 * mnf_before, 3)
        flow += leak_lps * ramp([(at(4, 0, 0), 0.0), (leak_start, 1.0)])
        complaint_time = at(5, 10, 40)
        work_orders.append((complaint_time, "Zone 3 network", "complaint", "medium",
                            "Customer complaint: low pressure at 14 Mill Lane, Zone 3 - intermittent since yesterday"))
        scenarios.append({
            "id": "leak",
            "name": "Leak raising the minimum night flow",
            "start": _iso(leak_start), "end": _iso(end_ts),
            "detectable_by_pipeline": False,
            "description": f"zone_flow_lps carries an extra {leak_lps} L/s (25% of the pre-leak minimum night flow {mnf_before:.2f} L/s) from day 4 01:00, ramping in from 00:00; a low-pressure complaint follows on day 5.",
            "effects": [
                {"dataset": "Meters/zone_flow", "signal": "zone_flow_lps", "start": _iso(leak_start), "end": _iso(end_ts),
                 "description": f"minimum night flow (01:00-04:00) ~25% higher than before (+{leak_lps} L/s)", "expected_pipeline_events": [],
                 "mnf_before_lps": round(mnf_before, 3), "leak_lps": leak_lps},
                {"dataset": "AX/work_orders", "signal": None, "start": _iso(complaint_time), "end": _iso(complaint_time),
                 "description": "work order: customer complaint about low pressure in Zone 3", "expected_pipeline_events": []},
            ],
        })

    # ------------------------------------------------ scenario 6: glitch spikes
    glitches = [(at(2, 9, 35), 1.8), (at(6, 3, 15), -2.2)]
    glitch_effects = []
    for when, delta in glitches:
        if when <= end_ts:
            i = int((when - start_ts) / pd.Timedelta(minutes=INTERVAL_MINUTES))
            inlet[i] += delta
            glitch_effects.append({
                "dataset": "TRIO/telemetry_site_a", "signal": "inlet_pressure_bar", "start": _iso(when), "end": _iso(when),
                "description": f"single sample {delta:+.1f} bar", "expected_pipeline_events": ["spike_high" if delta > 0 else "spike_low"],
            })
    if glitch_effects:
        scenarios.append({
            "id": "glitch_spikes",
            "name": "Isolated glitch spikes in inlet pressure",
            "start": glitch_effects[0]["start"], "end": glitch_effects[-1]["end"],
            "detectable_by_pipeline": True,
            "description": "two single-sample telemetry glitches in inlet_pressure_bar, one high and one low.",
            "effects": glitch_effects,
        })

    # ---------------------------------------------------------- write TRIO
    files: dict[str, str] = {}
    trio_path = os.path.join(input_path, "TRIO", "telemetry_site_a.csv")
    trio_rows = [
        [t[i].strftime("%Y-%m-%dT%H:%M:%SZ"), f"{outlet[i]:.3f}", f"{inlet[i]:.3f}", f"{int(rssi[i])}"]
        for i in range(n) if trio_keep[i]
    ]
    _write_csv(trio_path, ["timestamp", "outlet_pressure_bar", "inlet_pressure_bar", "rssi_dbm"], trio_rows)
    files["TRIO/telemetry_site_a"] = trio_path

    # ----------------------------------------------------------- write TCL
    tcl_header = ["Time", "Tank Level A (%)", "Tank Level B (%)", "P1 Current (A)", "P2 Current (A)"]
    tcl_rows = [
        [t[i].strftime("%d/%m/%Y %H:%M"), f"{tank_a[i]:.2f}", f"{tank_b[i]:.2f}", f"{p1[i]:.2f}", f"{p2[i]:.2f}"]
        for i in range(n)
    ]
    quiet = np.ones(n, dtype=bool)  # keep the mess away from the pump-trip window
    if trip_start <= end_ts:
        quiet &= ~between(trip_start - pd.Timedelta(hours=3), trip_end + pd.Timedelta(hours=3))
    quiet[:60] = False
    quiet[-60:] = False
    candidates = np.flatnonzero(quiet)
    picks = rng.choice(candidates, size=min(len(candidates), 22), replace=False)
    picks = [int(p) for p in picks]
    tcl_mess = {"duplicate_rows": [], "resent_rows": [], "blank_cells": [], "error_cells": [], "quoted_number_cells": [], "rows_without_time": []}
    columns = tcl_header[1:]
    for p in picks[0:6]:  # blank cells
        col = int(rng.integers(0, len(columns)))
        tcl_rows[p][col + 1] = ""
        tcl_mess["blank_cells"].append([tcl_rows[p][0], columns[col]])
    for p in picks[6:9]:  # error markers in P1 current
        tcl_rows[p][3] = "ERR"
        tcl_mess["error_cells"].append([tcl_rows[p][0], "P1 Current (A)", "ERR"])
    tcl_rows[picks[9]][2] = "n/a"
    tcl_mess["error_cells"].append([tcl_rows[picks[9]][0], "Tank Level B (%)", "n/a"])
    for p in picks[10:15]:  # numbers stored as quoted strings
        col = int(rng.integers(0, len(columns)))
        tcl_rows[p][col + 1] = f'"{tcl_rows[p][col + 1]}"'
        tcl_mess["quoted_number_cells"].append([tcl_rows[p][0], columns[col]])
    tcl_rows[picks[15]][0] = ""
    tcl_mess["rows_without_time"].append(t[picks[15]].strftime("%d/%m/%Y %H:%M"))
    inserts: list[tuple[int, list[str], str]] = []
    for p in picks[16:20]:  # exact duplicate rows
        inserts.append((p, list(tcl_rows[p]), "duplicate"))
        tcl_mess["duplicate_rows"].append(tcl_rows[p][0])
    for p in picks[20:22]:  # a stale reading re-sent before the correct one (same Time)
        stale = list(tcl_rows[p])
        stale[1] = f"{tank_a[p] + 0.9:.2f}"
        inserts.append((p, stale, "stale"))
        tcl_mess["resent_rows"].append({"time": tcl_rows[p][0], "stale_tank_level_a_pct": stale[1], "kept_tank_level_a_pct": tcl_rows[p][1]})
    for p, row, kind in sorted(inserts, key=lambda item: -item[0]):
        tcl_rows.insert(p if kind == "stale" else p + 1, row)
    tcl_path = os.path.join(input_path, "TCL", "tank_pumps.csv")
    _write_csv(tcl_path, tcl_header, tcl_rows)
    files["TCL/tank_pumps"] = tcl_path
    tcl_mess["time_format"] = "DD/MM/YYYY HH:MM (day-first, naive, UTC)"
    messiness["TCL/tank_pumps"] = tcl_mess

    # -------------------------------------------------------- write Meters
    meters_path = os.path.join(input_path, "Meters", "zone_flow.csv")
    meters_rows = [[t[i].strftime("%Y-%m-%d %H:%M:%S"), f"{flow[i]:.3f}", f"{bulk[i]:.2f}"] for i in range(n)]
    _write_csv(meters_path, ["DateTime", "zone_flow_lps", "customer_bulk_m3h"], meters_rows)
    files["Meters/zone_flow"] = meters_path

    # ------------------------------------------------------------ write AX
    n_routine = min(len(ROUTINE_WORK_ORDERS), max(4, int(round(12 * days / 7))))
    routine_times = sorted(rng.uniform(6.0, 18.0, n_routine) + rng.choice(np.arange(days), n_routine) * 24.0)
    for k, hour in enumerate(routine_times):
        when = start_ts + pd.Timedelta(minutes=int(round(hour * 60 / 5)) * 5)
        asset, category, priority, description = ROUTINE_WORK_ORDERS[k % len(ROUTINE_WORK_ORDERS)]
        work_orders.append((when, asset, category, priority, description))
    work_orders.sort(key=lambda item: item[0])
    ax_rows = []
    for k, (when, asset, category, priority, description) in enumerate(work_orders):
        ax_rows.append([when.strftime("%Y-%m-%d %H:%M:%S"), str(100231 + k), asset, category, priority, f'"{description}"'])
    ax_path = os.path.join(input_path, "AX", "work_orders.csv")
    _write_csv(ax_path, ["created_at", "work_order", "asset", "category", "priority", "description"], ax_rows)
    files["AX/work_orders"] = ax_path

    ground_truth = {
        "generator": "InSITE_Engine.demo",
        "seed": int(seed),
        "days": int(days),
        "start": _iso(start_ts),
        "end": _iso(end_ts),
        "interval_minutes": INTERVAL_MINUTES,
        "timezone": "UTC",
        "input_path": os.path.abspath(input_path),
        "files": files,
        "routine_work_orders": n_routine,
        "scenarios": scenarios,
        "messiness": messiness,
    }
    gt_path = os.path.join(os.path.dirname(os.path.abspath(input_path)), GROUND_TRUTH_FILE)
    with open(gt_path, "w", encoding="utf-8") as fh:
        json.dump(ground_truth, fh, indent=2)
    ground_truth["ground_truth_path"] = gt_path
    return ground_truth
