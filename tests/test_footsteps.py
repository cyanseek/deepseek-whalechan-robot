"""Explicit measurement fixtures exercise false-step rejection, not robot physics."""
import numpy as np
import pytest
from opendafeiyu.footsteps import FootstepTracker


def fixture():
    tracker = FootstepTracker()
    p = np.array([[0., .03, 0.], [0., -.03, 0.]])
    for tick in range(5):tracker.sample(tick*.01, p, [0., 0.], [True, True], [1., 0.], False)
    return tracker, p


def swing(tracker, p, side, start, *, height=.006, advance=.02, chatter=False):
    for tick in range(1, 9):
        loaded = np.ones(2, dtype=bool);loaded[side] = chatter and tick == 4
        clearance = np.zeros(2);clearance[side] = height
        tracker.sample(start+tick*.01, p, clearance, loaded, [1., 0.], True)
    p[side, 0] += advance
    for tick in range(9, 14):tracker.sample(start+tick*.01, p, [0., 0.], [True, True], [1., 0.], False)


def test_landing_chatter_counts_one_step_and_both_sides_alternate():
    tracker,p=fixture()
    swing(tracker,p,0,.04,chatter=True)
    swing(tracker,p,1,.17)
    r=tracker.report()
    assert r['confirmed_steps_by_foot']==[1,1]
    assert r['alternating_transitions']==1
    assert all(x['confirmed_s']-x['landing_s']>=.03-1e-8 for x in r['footstep_events'])


@pytest.mark.parametrize('height,advance',[(.001,.03),(.006,0),(.006,-.02)])
def test_drag_stationary_lift_and_backward_motion_are_not_forward_steps(height,advance):
    tracker,p=fixture();swing(tracker,p,0,.04,height=height,advance=advance)
    assert tracker.report()['confirmed_steps_by_foot']==[0,0]


def test_sample_gap_discards_pending_flight_instead_of_imputing_landing():
    tracker,p=fixture()
    tracker.sample(.05,p,[.006,0.],[False,True],[1.,0.],True)
    p[0,0]+=.02
    for tick in range(10):tracker.sample(.2+tick*.01,p,[0.,0.],[True,True],[1.,0.],False)
    assert tracker.report()['confirmed_steps_by_foot']==[0,0]
    assert tracker.report()['footstep_sampling_gaps']==1


def test_invalid_or_repeated_sample_rejected_without_mutating_state():
    tracker,p=fixture();before=tracker.report()
    with pytest.raises(ValueError):tracker.sample(.05,p,[float('nan'),0.],[True,True],[1.,0.],True)
    with pytest.raises(ValueError):tracker.sample(.04,p,[0.,0.],[True,True],[1.,0.],True)
    assert tracker.last_time==.04 and tracker.report()==before
