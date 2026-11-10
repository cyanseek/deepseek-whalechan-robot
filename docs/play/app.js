/* Browser input and clock; the game engine owns score and collisions. */
const engine = new OfflineStars(STAR_LEVEL);
let last = performance.now();

function snapshot() {
  return {
    game: engine.snapshot(), stopped: engine.paused, carried: false,
    state: {paused: engine.paused},
  };
}

function input(action, lane) {
  if (action === 'start') engine.start();
  else if (action === 'pause') engine.setPaused(!engine.paused);
  else engine.move(action === 'lane' ? lane
    : Math.max(0, Math.min(2, engine.lane + (action === 'left' ? -1 : 1))));
  window.starView.update(snapshot());
}

window.starView.connect(input);
document.getElementById('start').onclick = () => {
  input('start');
  document.getElementById('game').focus();
};
document.getElementById('pause').onclick = () => input('pause');
document.querySelectorAll('[data-lane]').forEach(button => {
  button.onclick = () => input('lane', Number(button.dataset.lane));
});

function loop(ms) {
  // A suspended/background browser must not fast-forward the whole round.
  engine.tick(Math.min(0.1, Math.max(0, (ms - last) / 1000)));
  last = ms;
  window.starView.update(snapshot());
  requestAnimationFrame(loop);
}
window.starView.update(snapshot());
requestAnimationFrame(loop);
