"""This file is the main controller file.

Adaptive cruise control that works from a *perceived* (noisy) distance to the lead car.

Pipeline per step (dt = 0.1 s):
  1. Sanitize the raw estimate (NaN -> dropout, negative -> 0).
  2. Dropout handling: if the estimate is `inf` right after we were tracking a car that
     was reasonably close, assume it is a missed detection and coast the track forward
     for up to HOLD_STEPS steps before believing the road is empty.
  3. Outlier rejection: median of the last 3 measurements.
  4. Alpha-beta filter -> smoothed distance and closing rate (-> lead speed estimate).
  5. Control = min(cruise law, gap-keeping law, emergency brake), computed on a
     *pessimistic* distance (filtered distance minus an error margin that grows with
     range), because the regressor is less accurate far away.

Mode is FOLLOWING when the (filtered) lead car is within FOLLOW_DIST, else CRUISING.
"""

from collections import deque
import math
from typing import Tuple

from mp2_simulator.simulator import Observation, Mode


DT = 0.1                 # simulator step (s)
A_MAX, A_MIN = 10.0, -10.0

# --- perception robustness ---------------------------------------------------
HOLD_STEPS = 10          # coast a lost track this many steps (1.0 s) before trusting `inf`
MEDIAN_WINDOW = 3
ALPHA, BETA = 0.5, 0.15  # alpha-beta filter gains
GATE_ABS, GATE_REL = 3.0, 0.10   # max correction per frame: 3 m + 10% of range
MAX_DV = 1.5             # max lead-speed estimate change per step (15 m/s^2)
MARGIN_ABS = 1.5         # pessimistic margin: MARGIN_ABS + MARGIN_REL * d
MARGIN_REL = 0.08

# --- control ------------------------------------------------------------------
FOLLOW_DIST = 40.0       # spec: CRUISING whenever lead is > 40 m
TRACK_DIST = 150.0       # use every plausible estimate: at 20 m/s we need ~57 m just to stop
D0 = 11.0                # standstill gap, centre-to-centre (spec min is 4 m; cars ~5 m long)
TIME_GAP = 1.6           # s
K_GAP = 0.25             # gap error gain
K_REL = 0.9              # relative speed gain
K_CRUISE = 1.2           # cruise speed gain
D_EMERGENCY = 7.0        # never plan to get closer than this (centre-to-centre)
A_COMFORT_DECEL = 3.0    # if avoiding collision needs more than this, brake fully
# Worst-case stopping constraint. Our strongest command (-10) maps to brake 0.5, while the
# lead car in the test scenarios brakes with brake 1.0, so it can stop much harder than we can.
A_EGO_BRAKE = 3.5        # deceleration we can count on (m/s^2), conservative
A_LEAD_BRAKE = 8.0       # assumed worst-case lead deceleration (m/s^2)
T_REACT = 0.5            # perception + filter + actuator delay (s)
D_STOP = 7.0             # centre-to-centre gap to keep after both cars stop
K_SAFE = 2.0             # gain toward the safe speed


# NOTE: Very important that the class name remains the same
class Controller:
    def __init__(self, distance_threshold: float):
        self.distance_threshold = distance_threshold
        self._reset_track()

    # ------------------------------------------------------------------ tracking
    # The track state is the distance and the lead car's *absolute* speed. Ego speed is
    # known exactly, so while coasting through a dropout the gap keeps shrinking correctly
    # even if we brake or accelerate.
    def _reset_track(self):
        self.d_hat = None        # filtered distance (m)
        self.v_lead_hat = 0.0    # estimated lead speed (m/s)
        self.v_lead_fast = 0.0   # quicker, noisier estimate used only to detect hard braking
        self.n_updates = 0
        self.lost_steps = 0
        self.recent = deque(maxlen=MEDIAN_WINDOW)
        self.d_prev = None

    def _update_track(self, z: float, ego_v: float):
        """Update the lead-car track with measurement z (may be inf).

        Returns the filtered distance (inf when no lead car is tracked)."""
        if z is None or math.isnan(z):
            z = math.inf
        z = max(float(z), 0.0)

        if self.d_hat is not None:
            pred = max(self.d_hat + (self.v_lead_hat - ego_v) * DT, 0.0)

        if math.isinf(z):
            if self.d_hat is None:
                return math.inf
            self.lost_steps += 1
            if self.lost_steps > HOLD_STEPS:
                self._reset_track()
                return math.inf
            # Missed detection: coast the track forward.
            self.d_hat = pred
            return self.d_hat

        self.lost_steps = 0
        self.recent.append(z)
        if self.d_hat is None:
            # New track: assume the lead moves at our speed until measurements say otherwise.
            self.d_hat, self.v_lead_hat, self.v_lead_fast, self.n_updates = z, ego_v, ego_v, 0
            return self.d_hat

        self.n_updates += 1
        z_med = sorted(self.recent)[len(self.recent) // 2]
        # A sudden much-closer reading is trusted immediately (safety first).
        if z < z_med - 5.0 and z < pred:
            z_med = z

        # Faster gains right after acquiring a track, settling to the steady-state ones.
        k = self.n_updates + 3
        alpha = max(ALPHA, 2.0 * (2 * k - 1) / (k * (k + 1)))
        beta = max(BETA, 6.0 / (k * (k + 1)))

        # Gate: one frame can only move the track so far (true distance cannot jump);
        # a consistent change still gets through over a few frames.
        gate = GATE_ABS + GATE_REL * pred
        resid = min(max(z_med - pred, -gate), gate)
        self.d_hat = max(pred + alpha * resid, 0.0)
        # Lead speed can only change as fast as a car can brake / accelerate.
        dv = min(max(beta * resid / DT, -MAX_DV), MAX_DV)
        self.v_lead_hat = min(max(self.v_lead_hat + dv, 0.0), 40.0)
        # Fast estimate: raw (median-filtered) range rate, low-passed lightly.
        v_raw = ego_v + (z_med - self.d_prev) / DT if self.d_prev is not None else self.v_lead_hat
        self.v_lead_fast += 0.35 * (min(max(v_raw, 0.0), 40.0) - self.v_lead_fast)
        self.d_prev = z_med
        return self.d_hat

    # ------------------------------------------------------------------ control
    def run_step(self, obs: Observation) -> Tuple[float, Mode]:
        """Return (acceleration in [-10, 10], operation mode)."""
        v = max(float(obs.ego_velocity), 0.0)
        v_des = float(obs.desired_speed)

        d = self._update_track(obs.distance_to_lead, v)

        a_cruise = K_CRUISE * (v_des - v)
        accel = a_cruise

        if d < TRACK_DIST:
            # Pessimistic distance: assume the estimate is too long by a range-dependent margin.
            d_eff = max(d - (MARGIN_ABS + MARGIN_REL * d), 0.0)
            v_lead = self.v_lead_hat

            d_target = D0 + TIME_GAP * v
            a_gap = K_GAP * (d_eff - d_target) + K_REL * (v_lead - v)
            accel = min(accel, a_gap)

            # Worst case: lead slams on its brakes now. Largest speed v_safe from which we
            # can still stop D_STOP behind it, given our weaker brakes and a reaction delay:
            #   v*T + v^2/(2 a_e) <= d_eff - D_STOP + v_lead^2/(2 a_l)
            v_lead_cons = min(v_lead, self.v_lead_fast)  # pessimistic: whichever says 'slower'
            budget = d_eff - D_STOP + v_lead_cons * v_lead_cons / (2.0 * A_LEAD_BRAKE)
            if budget <= 0:
                v_safe = 0.0
            else:
                a_e = A_EGO_BRAKE
                v_safe = a_e * (-T_REACT + math.sqrt(T_REACT ** 2 + 2.0 * budget / a_e))
            accel = min(accel, K_SAFE * (v_safe - v))

            # Emergency check: deceleration needed to match the lead's speed before
            # the gap shrinks to D_EMERGENCY.
            closing = v - v_lead
            room = d_eff - D_EMERGENCY
            if closing > 0:
                if room <= 0:
                    accel = A_MIN
                else:
                    a_req = closing * closing / (2.0 * room)
                    if a_req > A_COMFORT_DECEL:
                        accel = A_MIN
                    elif a_req > 0.5:
                        accel = min(accel, -2.0 * a_req)
            elif room <= 0:
                # Too close even though not closing: back off.
                accel = min(accel, -3.0)

            # Don't come to a stop while the lead is clearly still moving and there is room.
            if v_lead > 1.0 and v < 1.0 and room > 3.0 and v_safe > 1.5:
                accel = max(accel, 1.0)

        accel = min(max(accel, A_MIN), A_MAX)
        mode = Mode.FOLLOWING if d < FOLLOW_DIST else Mode.CRUISING
        return accel, mode
