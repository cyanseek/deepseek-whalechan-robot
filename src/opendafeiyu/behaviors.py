"""Feedback-driven behavior composition; only InteractionController owns motion."""
from .dialogue import InteractionController, normalized, STOP_WORDS, HEAD_ACTIONS

LABELS = {'wave': '挥手', 'tail': '摆尾', 'nod': '点头', 'shake': '摇头',
          'look_left': '向左看', 'look_right': '向右看', 'approach:red': '过来',
          'approach:green': '去绿色标记', 'jump': '跳一下'}
ROUTINES = {'打个招呼': ['wave', 'tail', 'nod'], '看看左右': ['look_left', 'look_right', 'nod']}


class BehaviorController(InteractionController):
    """Opt-in sequences with pause/resume, replacement and simulated-time deadlines.

    Resume restarts an interrupted gesture from its beginning after settling;
    approach reacquires the RGB target. No joint/state trajectory is saved.
    """
    def __init__(self, low, translator=None):
        super().__init__(low, translator)
        self.plan = []
        self.cursor = 0
        self.saved = []
        self._pending_resume = []
        self.active_token = None
        self.deadline = None
        self.behavior_status = 'idle'

    def _validate(self, actions):
        if not isinstance(actions, list) or not 1 <= len(actions) <= 6:
            raise ValueError('Expected 1–6 named actions')
        for token in actions:
            if not isinstance(token, str) or token not in LABELS:
                raise ValueError('Unsupported named action')
            if token in HEAD_ACTIONS and token not in getattr(self.low, 'interaction_gestures', ()):
                raise ValueError('Head gesture unavailable in this profile')
            if token == 'jump' and not getattr(self.low, 'interaction_jump', False):
                raise ValueError('Jump unavailable in this profile')
        return list(actions)

    def _launch(self, now):
        token = self.plan[self.cursor]
        self.generation += 1
        self._halt(now)
        action, _, target = token.partition(':')
        self.text = '组合动作：' + LABELS[token]
        self.accept(self.generation, {'action': action, 'target': target or 'none',
                    'reply': '收到，先站稳，再' + LABELS[token] + '。'}, now, 'explicit_behavior_sequence')
        self.behavior_status = 'running'
        self.event(now, 'sequence_step_started', index=self.cursor, token=token)

    def submit_sequence(self, actions, now, source='structured_sequence'):
        plan = self._validate(actions)  # Reject atomically before touching current motion.
        self._pending_resume = []
        self.plan, self.cursor, self.saved = plan, 0, []
        self.event(now, 'sequence_received', actions=plan, source=source)
        self._launch(now)
        return self.generation

    def accept(self, generation, task, now, source):
        accepted = super().accept(generation, task, now, source)
        if accepted:
            self.active_token = (task['action'] + (':' + task['target'] if task['action'] == 'approach' else '')) if self.action else None
            self.deadline = now + (60 if self.action == 'approach' else 20) if self.action else None
            if task['action'] == 'stop':
                self.saved = self._pending_resume.copy()
                self.behavior_status = 'paused' if self.saved else 'idle'
                self.event(now, 'sequence_paused', remaining=self.saved.copy())
            self._pending_resume = []
        return accepted

    def submit(self, text, now, source='text_entry'):
        if not isinstance(text, str) or not text.strip() or len(text) > 500:
            raise ValueError('Input must contain 1–500 characters')
        word = normalized(text)
        if word in ('继续', '继续任务', 'resume'):
            if self.saved:
                saved = self.saved.copy()
                generation = self.submit_sequence(saved, now, source='resume_saved_intent')
                self.event(now, 'sequence_resumed', actions=saved)
                return generation
            self.event(now, 'resume_unavailable')
            self.reply = '当前没有暂停的任务。'
            return self.generation
        if word in ROUTINES:
            return self.submit_sequence(ROUTINES[word], now, source='named_routine')
        if word in ('取消', '取消任务', 'cancel'):
            self._pending_resume = []
            self.plan, self.saved, self.active_token, self.deadline = [], [], None, None
            generation = super().submit('停一下', now, source)
            self.behavior_status = 'cancelled'
            self.event(now, 'sequence_cancelled')
            return generation
        if word in STOP_WORDS:
            self._pending_resume = []
            remaining = self.plan[self.cursor:] if self.plan else ([self.active_token] if self.active_token else self.saved)
            self.saved = list(remaining)
            self.plan, self.active_token, self.deadline = [], None, None
            generation = super().submit(text, now, source)
            self.behavior_status = 'paused' if self.saved else 'idle'
            self.event(now, 'sequence_paused', remaining=self.saved)
            return generation
        # Keep interrupted intentions until the model distinguishes pause from replacement.
        self._pending_resume = list(self.plan[self.cursor:] if self.plan else
                                    ([self.active_token] if self.active_token else self.saved))
        self.plan, self.saved, self.active_token, self.deadline = [], [], None, None
        self.behavior_status = 'single'
        return super().submit(text, now, source)

    def tick(self, detections, now):
        if self.deadline is not None and now >= self.deadline:
            self.submit('取消任务', now, source='behavior_timeout')
            self.behavior_status = 'timed_out'
            self.reply = '这个动作暂时没有完成，我先停稳。'
            self.event(now, 'behavior_timeout')
        command = super().tick(detections, now)
        if self.low.phase == 'failed' or self.status in ('lost', 'model_error'):
            self._pending_resume = []
            self.plan, self.saved, self.active_token, self.deadline = [], [], None, None
            self.behavior_status = 'failed'
        elif self.plan and self.status in ('complete', 'arrived'):
            self.event(now, 'sequence_step_completed', index=self.cursor, token=self.plan[self.cursor])
            self.cursor += 1
            if self.cursor < len(self.plan):
                self._launch(now)
            else:
                self.plan, self.active_token, self.deadline = [], None, None
                self.behavior_status = 'complete'
                self.event(now, 'sequence_completed')
        elif not self.plan and self.status in ('complete', 'arrived', 'reply'):
            self.active_token, self.deadline = None, None
        return command

    def behavior_state(self):
        return {'status': self.behavior_status, 'remaining': self.plan[self.cursor:],
                'paused_actions': self.saved.copy(), 'can_resume': bool(self.saved),
                'deadline_sim_s': self.deadline}
