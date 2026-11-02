"""Intent lifecycle checks without character assets or provider requests."""
import threading
from types import SimpleNamespace

import pytest

from opendafeiyu.behaviors import BehaviorController
from opendafeiyu.dialogue import command_task


@pytest.mark.parametrize('followup', ['direct', 'queued_then_direct', 'interpreted'])
def test_repeated_pause_keeps_interrupted_intentions(followup):
    release = threading.Event()
    started = threading.Event()

    def translator(text):
        if text == '稍等，先别动了':
            started.set()
            release.wait(5)
            return {'task': command_task('摆尾'), 'mode': 'delayed_test_fixture'}
        return {'task': command_task('停一下'), 'mode': 'test_fixture'}

    low = SimpleNamespace(phase='ready', contacts=lambda: (True, True), cancel=lambda: None,
                          interaction_gestures=('nod',))
    control = BehaviorController(low, translator)
    try:
        control.submit_sequence(['wave', 'nod'], 0)
        control.submit('稍等，先别动了', .1)
        control.poll(.1)
        assert started.wait(2)
        stale = control.pending[0][1]
        if followup != 'direct':
            control.submit('再等我一会', .2)
        if followup == 'interpreted':
            control.poll(.2)
            control.pending[-1][1].result(timeout=2)
            control.poll(.3)
        else:
            control.submit('停一下', .3)
            control.submit('停一下', .4)
        assert control.saved == ['wave', 'nod']
        assert control.behavior_state()['can_resume'] and control.queued is None
        release.set()
        stale.result(timeout=2)
        control.poll(.5)
        assert control.saved == ['wave', 'nod'] and control.action is None
        assert any(e['event'] == 'stale_response_rejected' for e in control.events)
        control.submit('继续', .6)
        assert control.plan == ['wave', 'nod'] and control.action == 'wave'
        assert not control.saved and not control._pending_resume
    finally:
        release.set()
        control.close()
