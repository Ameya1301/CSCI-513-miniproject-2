"""Offline smoke test of controller.py with a crude kinematic model (no CARLA).

Not a substitute for the real evaluation; checks the 5 specs against true distance
while feeding the controller noisy / dropped-out distance estimates."""
import sys, types, math, random
from enum import Enum
from typing import NamedTuple

stub = types.ModuleType("mp2_simulator.simulator")
class Mode(Enum):
    CRUISING = 0
    FOLLOWING = 1
class Observation(NamedTuple):
    ego_velocity: float
    desired_speed: float
    distance_to_lead: float
stub.Mode, stub.Observation = Mode, Observation
sys.modules["mp2_simulator"] = types.ModuleType("mp2_simulator")
sys.modules["mp2_simulator.simulator"] = stub
sys.path.insert(0, sys.argv[1])
from mp2_controller.controller import Controller

DT = 0.1
import os; LAG = float(os.environ.get("LAG", "0.3"))
import os
EXACT = os.environ.get('EXACT') == '1'  # MP1 setting: exact distance every step
import json, glob
# Defaults calibrated so the MP1 controller (passed all 15 in CARLA) passes here with exact distance.
EGO_BRAKE = float(os.environ.get("EGO_BRAKE", "0.8")); LEAD_BRAKE = float(os.environ.get("LEAD_BRAKE", "6.0"))
def ego_accel(acc):
    return 0.5 * acc if acc > 0 else EGO_BRAKE * acc

def run(scn, seed, noise=0.10, drop=0.10, spike=0.0):
    rng = random.Random(seed)
    d0 = scn['ego']['position']['x'] - scn['lead']['position']['x']
    acts = scn['ado_actions']
    ctrl = Controller(4.0)
    v, vl, d, a_act = 0.0, 0.0, d0, 0.0
    tr = [(0.0, v, d, vl, Mode.CRUISING)]
    for k in range(200):
        t = (k + 1) * DT
        p_det = 1.0 if (EXACT or d < 60) else (0.64 if d < 80 else 0.05)
        z = d * (1 + rng.gauss(0, noise)) if rng.random() < p_det else math.inf
        r = rng.random()
        if r < drop: z = math.inf
        elif r < drop + spike and z != math.inf: z = d * rng.choice([0.6, 1.5])
        acc, mode = ctrl.run_step(Observation(v, 20.0, z))
        a_act += (ego_accel(acc) - a_act) * DT / LAG
        v = max(v + a_act * DT, 0.0)
        a = acts[min(k, len(acts) - 1)]
        vl = max(vl + (5.0 * a['throttle'] - LEAD_BRAKE * a["brake"]) * DT, 0.0)
        d += (vl - v) * DT
        if d < 4.7:
            tr.append((t, v, 0.0, vl, mode)); break
        tr.append((t, v, d, vl, mode))
    return tr

def check(tr):
    res = {}
    res["safeFollowing"] = all(d >= 4 for _, _, d, _, _ in tr)
    res["forwardProgress"] = all(v >= 0 for _, v, _, _, _ in tr)
    # stop while lead moving after 3 s
    # G[3,3] not((v_lead > 0.1) U (v < 0.1)): violated iff ego stops at some t'>=3 while lead moved throughout [3, t')
    viol = False
    for j, (t, v, _, vl, _) in enumerate(tr):
        if t < 3: continue
        if v < 0.1:
            viol = all(x[3] > 0.1 for x in tr[:j] if x[0] >= 3); break
        if vl <= 0.1: break
    res["dontStop"] = not viol
    ok = True
    for i, (t, v, d, vl, m) in enumerate(tr):
        if t <= 18 and d > 40:
            win = tr[i:i + 21]
            if not any(mm == Mode.CRUISING or dd < 40 for _, _, dd, _, mm in win): ok = False
    res["notFollowFar"] = ok
    ok = True
    for i, (t, v, d, vl, m) in enumerate(tr):
        if t <= 14 and m == Mode.CRUISING:
            win = tr[i:i + 61]
            if not any(abs(vv - 20) < 2 or mm == Mode.FOLLOWING for _, vv, _, _, mm in win): ok = False
    res["reachTarget"] = ok
    return res, min(d for _, _, d, _, _ in tr)

fails = 0; total = 0
scns = sorted(glob.glob(sys.argv[1] + '/test_data/*.json'))
CONFIGS = ((0.0, 0.0, 0.0),) if EXACT else ((0.0, 0.0, 0.0), (0.06, 0.03, 0.0), (0.10, 0.10, 0.02))
for noise, drop, spike in CONFIGS:
    worst = {}
    for f in scns:
        scn = json.load(open(f))
        for seed in range(10):
            total += 1
            res, dmin = check(run(scn, seed, noise, drop, spike))
            name = f.split('/')[-1][:-5]
            worst[name] = min(worst.get(name, 1e9), dmin)
            if not all(res.values()):
                fails += 1
                print(f"FAIL noise={noise} drop={drop} {name} seed={seed} dmin={dmin:.1f}", [k for k, ok in res.items() if not ok])
    print(f"noise={noise} drop={drop} min gap per scenario:", {k: round(v, 1) for k, v in worst.items()})
print("total failing runs:", fails, "of", total)
