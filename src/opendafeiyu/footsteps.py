"""Measure footfalls from contact and sole motion, independently of planner counters."""
import math
import numpy as np


class FootstepTracker:
    """Confirm a step only after lift, forward travel and sustained loaded landing.

    Inputs are measured simulation quantities. Contact chatter shorter than 30 ms
    cannot confirm a landing. Samples more than 50 ms apart discard a pending step
    because the unobserved interval cannot establish continuous support.
    """
    def __init__(self):
        self.last_time = None
        self.feet = [self._empty(), self._empty()]
        self.events = []
        self.gaps = 0

    @staticmethod
    def _empty():
        return dict(anchor=None, loaded_since=None, swing=None)

    def sample(self, now, positions, clearance, loaded, forward, walking):
        positions = np.asarray(positions, dtype=float)
        clearance = np.asarray(clearance, dtype=float)
        forward = np.asarray(forward, dtype=float)
        loaded = np.asarray(loaded)
        if (not math.isfinite(now) or positions.shape != (2, 3) or
                clearance.shape != (2,) or forward.shape != (2,) or
                loaded.shape != (2,) or loaded.dtype != np.dtype(bool) or
                not all(np.isfinite(x).all() for x in (positions, clearance, forward)) or
                np.linalg.norm(forward) < 1e-8):
            raise ValueError('Invalid footstep measurement')
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('Footstep timestamps must increase')
        if self.last_time is not None and now - self.last_time > .050001:
            self.feet = [self._empty(), self._empty()]
            self.gaps += 1
        self.last_time = now
        heading = forward / np.linalg.norm(forward)
        for side, state in enumerate(self.feet):
            if loaded[side]:
                if state['loaded_since'] is None:
                    state['loaded_since'] = now
                swing = state['swing']
                if swing is not None:
                    if swing['landing'] is None:
                        swing['landing'] = (now, positions[side].copy())
                    landed, point = swing['landing']
                    if now - landed >= .03 - 1e-8:
                        advance = float((point[:2] - swing['start'][:2]) @ swing['heading'])
                        if (landed - swing['time'] >= .03 - 1e-8 and
                                swing['peak'] >= .003 and advance >= .005):
                            self.events.append(dict(side=('left', 'right')[side],
                                takeoff_s=swing['time'], landing_s=landed, confirmed_s=now,
                                peak_clearance_m=swing['peak'], advance_m=advance))
                        state['swing'] = None
                if now - state['loaded_since'] >= .03 - 1e-8:
                    state['anchor'] = positions[side].copy()
            else:
                state['loaded_since'] = None
                if state['swing'] is None and state['anchor'] is not None and walking:
                    state['swing'] = dict(time=now, start=state['anchor'].copy(),
                                          heading=heading.copy(), peak=0., landing=None)
                if state['swing'] is not None:
                    state['swing']['landing'] = None
                    state['swing']['peak'] = max(state['swing']['peak'], float(clearance[side]))

    def report(self):
        sides = [event['side'] for event in self.events]
        return dict(confirmed_steps_by_foot=[sides.count('left'), sides.count('right')],
                    alternating_transitions=sum(a != b for a, b in zip(sides, sides[1:])),
                    footstep_events=[dict(event) for event in self.events],
                    footstep_sampling_gaps=self.gaps,
                    footstep_definition='Measured lift >=3mm, forward sole travel >=5mm, flight >=30ms, loaded landing >=30ms; left/right. Independent of planner count.')
