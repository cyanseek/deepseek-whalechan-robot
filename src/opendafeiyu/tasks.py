"""Camera-only target tracking and cancelable high-level tasks."""
from dataclasses import dataclass
import math

import cv2
import numpy as np


def manual_task(text):
    """Small explicit command vocabulary when no language model is enabled."""
    text = text.strip().lower()
    if text in ('stop', '停下', '停止', 'cancel'):
        return {'action': 'stop', 'target': 'none'}
    if text in ('red', 'green'):
        return {'action': 'approach', 'target': text}
    raise ValueError('Use red, green or stop; enable -Language for natural language.')


@dataclass(frozen=True)
class Detection:
    target: str
    bearing: float
    distance: float
    pixels: int


def detect_targets(rgb, fovy=60.):
    """Known 9cm color markers, observed only through rendered RGB pixels."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = rgb.shape[:2]
    focal = h / (2 * math.tan(math.radians(fovy) / 2))
    detections = {}
    for target in ('red', 'green'):
        hue = hsv[:, :, 0]
        mask = ((hue < 12) | (hue > 170)) if target == 'red' else ((hue > 40) & (hue < 90))
        mask &= (hsv[:, :, 1] > 140) & (hsv[:, :, 2] > 80)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 12 or min(bw, bh) < 4 or x <= 0 or x + bw >= w or y <= 0 or y + bh >= h:
            continue
        center = x + bw / 2
        bearing = math.atan((w / 2 - center) / focal)
        # Sphere apparent diameter; slant range correction is small at the gated bearing.
        distance = .09 * focal / bw
        detections[target] = Detection(target, bearing, distance, int(area))
    return detections


class TaskController:
    def __init__(self):
        self.generation = 0
        self.target = None
        self.status = 'idle'
        self.events = []
        self.last_seen = None
        self.near_since = None
        self.arrival_distance = .32
        self.arrival_bearing = .15
        self.started_at = 0.
        self.memory = {}

    def request(self, now):
        self.cancel(now, reason='new_request')
        self.status = 'thinking'
        return self.generation

    def accept(self, generation, task, now, source):
        if generation != self.generation:
            self.events.append({'time': now, 'event': 'stale_response_rejected', 'generation': generation})
            return False
        if set(task) != {'action', 'target'} or task['action'] not in ('approach', 'stop', 'look'):
            raise ValueError('Unsupported task')
        if task['action'] == 'stop':
            self.cancel(now)
            return True
        if task['target'] not in ('red', 'green'):
            raise ValueError('Unknown target')
        self.target, self.status = task['target'], task['action']
        self.last_seen = None
        self.near_since = None
        self.started_at = now
        self.events.append({'time': now, 'event': 'task_accepted', 'generation': generation,
                            'task': task, 'source': source})
        return True

    def cancel(self, now, reason='user_cancel'):
        self.generation += 1
        self.target = None
        self.status = 'stopped'
        self.near_since = None
        self.events.append({'time': now, 'event': reason, 'generation': self.generation})

    def command(self, detections, now):
        command = np.zeros(7)
        self.memory.update({name: (det, now) for name, det in detections.items()})
        if self.target is None or self.status in ('arrived', 'lost', 'stopped', 'thinking'):
            return command
        det = detections.get(self.target)
        if det is None:
            self.near_since = None
            # No target in the current camera frame means no blind forward motion.
            if self.last_seen is None and now - self.started_at < 4.:
                remembered = self.memory.get(self.target)
                direction = np.sign(remembered[0].bearing) if remembered else 1.
                command[2] = .45 * (direction or 1.)
                return command
            if self.last_seen is None or now - self.last_seen > .6:
                self.status = 'lost'
                self.events.append({'time': now, 'event': 'target_lost', 'target': self.target})
            return command
        self.last_seen = now
        command[2] = np.clip(1.5 * det.bearing, -.45, .45)
        if self.status == 'look':
            return command
        # A small release margin prevents gait-induced pixel-size oscillation
        # from repeatedly restarting the arrival confirmation after first entry.
        threshold = self.arrival_distance + (.03 if self.near_since is not None else 0.)
        if det.distance <= threshold and abs(det.bearing) < self.arrival_bearing:
            if self.near_since is None:
                self.near_since = now
            if now - self.near_since >= .3:
                self.status = 'arrived'
                self.events.append({'time': now, 'event': 'arrived', 'target': self.target,
                                    'observed_distance': det.distance})
            return np.zeros(7)
        self.near_since = None
        # This policy has a measured low-speed dead zone; approach at a supported
        # command, then switch to zero based on the observed arrival condition.
        command[0] = min(.15, max(.10, .9 * (det.distance - .27))) * max(0, 1 - abs(det.bearing) / .6)
        return command
