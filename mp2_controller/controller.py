"""Adaptive cruise controller: integrated cruising and following.

Ported from our Mini-Project 1 controller (same cruising P-controller, safe-gap
following law, emergency gap and mode hysteresis). Mini-Project 2 changes:

  1. Import from mp2_simulator.
  2. Perception front end. The distance is now a noisy neural-network estimate and is
     `inf` whenever YOLO misses the car or the lane model says "not in lane". Instead of
     the raw distance and a raw finite-difference closing speed, the MP1 logic now gets:
       - median-of-3 outlier rejection,
       - an alpha-beta tracker for distance and lead speed (gated, rate-limited),
       - dropout handling: a lost car is coasted for up to 1 s before trusting `inf`.
  3. Extra safety margin: the MP1 law works on a pessimistic distance (estimate minus
     1.5 m + 8 %), and the minimum gap is raised from 4 m to 10 m centre-to-centre
     (cars are ~5 m long, so 4 m centre-to-centre is already contact).
  4. Worst-case braking floor. In scenario 3 the lead brakes with brake=1.0, but our
     strongest command (-10) only maps to brake=0.5. A speed cap guarantees we can
     still stop behind the lead if it brakes as hard as it can.
"""

from collections import deque
import math
from typing import Tuple

from mp2_simulator.simulator import Observation, Mode


DT = 0.1                 # simulator step (s)

# --- perception robustness ---------------------------------------------------
HOLD_STEPS = 10          # coast a lost track this many steps (1.0 s) before trusting `inf`
MEDIAN_WINDOW = 3
ALPHA, BETA = 0.5, 0.15  # alpha-beta filter gains
GATE_ABS, GATE_REL = 3.0, 0.10   # max correction per frame: 3 m + 10% of range
MAX_DV = 1.5             # max lead-speed estimate change per step (15 m/s^2)
MARGIN_ABS = 1.5         # pessimistic distance margin: MARGIN_ABS + MARGIN_REL * d
MARGIN_REL = 0.08

# --- MP1 following law ---------------------------------------------------------
KP_CRUISE = 3.0          # MP1 cruising gain
MIN_GAP = 10.0           # MP1 used max(4, threshold); raised for noisy estimates
REACTION_TIME = 1.0      # MP1
RELATIVE_DECEL = 3.0     # MP1
FOLLOW_DIST = 40.0       # spec: report CRUISING whenever the lead is > 40 m away

# --- worst-case braking floor (new) -----------------------------------------
# Calibrated so that the unmodified MP1 controller (which passed all 15 scenarios in
# CARLA with exact distances) is also collision-free in our offline model: that needs
# our brake=0.5 to stop about as hard as the lead's brake=1.0.
A_EGO_BRAKE = 5.5        # deceleration we count on at full command (m/s^2), with margin
A_LEAD_BRAKE = 8.0       # assumed worst-case lead deceleration (m/s^2), pessimistic
T_REACT = 0.5            # perception + filter + actuator delay (s)
D_STOP = 7.0             # centre-to-centre gap to keep after both cars stop
K_SAFE = 2.0             # gain toward the safe speed


# NOTE: Very important that the class name remains the same
class Controller:
    def __init__(self, distance_threshold: float):
        self.distance_threshold = distance_threshold
        self.kp = KP_CRUISE
        self.dt = DT
        self.is_following = False
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

    # ------------------------------------------------------------------ MP1 logic
    def cruising_control(self, ego_velocity: float, desired_speed: float) -> float:
        speed_error = desired_speed - ego_velocity
        acceleration = self.kp * speed_error

        if ego_velocity <= 0.0:
            acceleration = max(0.0, acceleration)

        acceleration = max(-10.0, min(10.0, acceleration))
        return float(acceleration)

    def _safe_speed(self, d_eff: float, v_lead: float) -> float:
        """Largest speed from which we can still stop D_STOP behind the lead if it brakes
        as hard as it can right now:  v*T + v^2/(2 a_e) <= d_eff - D_STOP + v_lead^2/(2 a_l)."""
        budget = d_eff - D_STOP + v_lead * v_lead / (2.0 * A_LEAD_BRAKE)
        if budget <= 0.0:
            return 0.0
        a_e = A_EGO_BRAKE
        return a_e * (-T_REACT + math.sqrt(T_REACT ** 2 + 2.0 * budget / a_e))

    def run_step(self, obs: Observation) -> Tuple[float, Mode]:
        ego_velocity = max(float(obs.ego_velocity), 0.0)
        desired_speed = obs.desired_speed

        # Perception front end (replaces MP1's raw distance / raw differencing).
        d_filt = self._update_track(obs.distance_to_lead, ego_velocity)

        if not math.isfinite(d_filt):
            self.is_following = False
            return (
                self.cruising_control(ego_velocity, desired_speed),
                Mode.CRUISING
            )

        distance = max(d_filt - (MARGIN_ABS + MARGIN_REL * d_filt), 0.0)
        v_lead = self.v_lead_hat
        v_lead_cons = min(v_lead, self.v_lead_fast)
        closing_speed = max(0.0, ego_velocity - v_lead)

        # ---- MP1 following logic (unchanged apart from MIN_GAP) ----
        minimum_gap = max(MIN_GAP, self.distance_threshold)
        braking_distance = (
            REACTION_TIME * closing_speed
            + closing_speed ** 2 / (2.0 * RELATIVE_DECEL)
        )
        safe_gap = minimum_gap + braking_distance

        # Enter following mode earlier when approaching a lead vehicle.
        entry_gap = max(safe_gap, minimum_gap + 2.0 * closing_speed)
        if distance <= entry_gap:
            self.is_following = True

        # Stay in following mode until there is a comfortable gap.
        if self.is_following:
            exit_gap = max(35.0, safe_gap + 10.0)
            if distance > exit_gap and closing_speed < 1.0:
                self.is_following = False

        if self.is_following:
            gap_error = distance - safe_gap
            acceleration = 0.5 * gap_error - 1.5 * closing_speed

            # Apply maximum braking earlier when closing rapidly.
            emergency_gap = (
                minimum_gap
                + 1.5 * closing_speed
                + closing_speed ** 2 / 16.0
            )
            if distance <= emergency_gap:
                acceleration = -10.0
        else:
            acceleration = self.cruising_control(ego_velocity, desired_speed)

        # ---- New: worst-case braking floor, applied in both modes ----
        v_safe = self._safe_speed(distance, v_lead_cons)
        acceleration = min(acceleration, K_SAFE * (v_safe - ego_velocity))

        # Don't stall behind a lead that is clearly moving if there is room to creep.
        if v_lead > 1.0 and ego_velocity < 1.0 and v_safe > 1.5 and distance > minimum_gap:
            acceleration = max(acceleration, 1.0)

        acceleration = max(-10.0, min(10.0, acceleration))

        # Spec: the controller must report CRUISING when the lead is > 40 m away.
        following = self.is_following and d_filt <= FOLLOW_DIST
        return (float(acceleration), Mode.FOLLOWING if following else Mode.CRUISING)
