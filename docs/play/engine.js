/* Authored three-lane screen game. Fixed 20 ms steps, no network or robot I/O. */
class OfflineStars {
  constructor(waves) {
    this.waves = structuredClone(waves);
    this.goal = 6;
    this.duration = 35;
    this.round = 0;
    this.reset('ready');
  }

  reset(phase) {
    this.phase = phase;
    this.time = 0;
    this.lane = 1;
    this.score = 0;
    this.lives = 3;
    this.objects = [];
    this.acc = 0;
    this.wave = 0;
    this.paused = false;
  }

  start() {
    this.reset('running');
    this.round++;
  }

  setPaused(paused) {
    if (this.phase === 'running') this.paused = Boolean(paused);
  }

  move(lane) {
    if (!Number.isInteger(lane) || lane < 0 || lane > 2) {
      throw new Error('lane must be 0, 1 or 2');
    }
    if (this.phase === 'running' && !this.paused) this.lane = lane;
  }

  tick(dt) {
    if (!Number.isFinite(dt) || dt < 0 || dt > 0.5) {
      throw new Error('dt must be finite and between 0 and 0.5 seconds');
    }
    if (this.phase !== 'running' || this.paused) return;
    this.acc += dt;
    while (this.acc >= 0.02 - 1e-10 && this.phase === 'running') {
      this.acc -= 0.02;
      this.step();
    }
  }

  step() {
    this.time += 0.02;
    const wave = this.waves[this.wave];
    if (wave && this.time + 1e-9 >= wave.time) {
      for (const object of wave.objects) this.objects.push({...object, y: 39});
      this.wave++;
    }
    const remaining = [];
    for (const object of this.objects) {
      object.y += 43 * 0.02;
      if (object.y >= 188) {
        if (object.lane === this.lane) {
          if (object.kind === 'star') this.score++;
          else this.lives--;
        }
      } else remaining.push(object);
    }
    this.objects = remaining;
    if (this.lives <= 0) this.phase = 'lost';
    else if (this.score >= this.goal) this.phase = 'won';
    else if (this.time + 1e-9 >= this.duration) this.phase = 'lost';
  }

  snapshot() {
    return structuredClone({
      phase: this.phase, time: this.time, lane: this.lane,
      score: this.score, lives: this.lives, goal: this.goal,
      duration: this.duration, objects: this.objects, round: this.round,
    });
  }
}

if (typeof module !== 'undefined') module.exports = OfflineStars;
