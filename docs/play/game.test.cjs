const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const Game = require('./engine.js');
const level = require('./level.json');
const build = require('./build.cjs');

function route(kind) {
  const game = new Game(level);
  game.start();
  for (let i = 0; i < 1800 && game.phase === 'running'; i++) {
    const targets = game.objects.filter(o => o.kind === kind).sort((a, b) => b.y - a.y);
    if (targets.length) game.move(targets[0].lane);
    game.tick(0.02);
  }
  return game;
}

test('waiting does not start a round', () => {
  const game = new Game(level), before = game.snapshot();
  game.move(0); game.setPaused(true); game.tick(0.5);
  assert.deepEqual(game.snapshot(), before);
  assert.equal(game.paused, false);
});

test('collecting six stars wins a full round', () => {
  const game = route('star');
  assert.equal(game.phase, 'won');
  assert.equal(game.score, 6);
  assert.equal(game.lives, 3);
  assert.ok(game.time > 10 && game.time < 20);
});

test('three jellyfish collisions lose the round', () => {
  const game = route('jelly');
  assert.equal(game.phase, 'lost');
  assert.equal(game.lives, 0);
  assert.ok(game.time < game.duration);
});

test('missing every star ends at the time limit', () => {
  const game = new Game([]);
  game.start();
  for (let i = 0; i < 1800; i++) game.tick(0.02);
  assert.equal(game.phase, 'lost');
  assert.equal(game.score, 0);
  assert.ok(Math.abs(game.time - 35) < 1e-8);
});

test('pause freezes timer, movement, collisions and fractional step', () => {
  const game = new Game(level);
  game.start(); game.tick(0.5); game.tick(0.111);
  game.setPaused(true);
  const before = game.snapshot(), fraction = game.acc;
  for (let i = 0; i < 100; i++) { game.tick(0.5); game.move(2); }
  assert.deepEqual(game.snapshot(), before);
  assert.equal(game.acc, fraction);
  game.setPaused(false); game.tick(0.02);
  assert.ok(game.time > before.time);
});

test('restart resets a paused or completed round', () => {
  for (const game of [route('star'), route('jelly'), new Game(level)]) {
    if (game.phase === 'ready') { game.start(); game.tick(0.5); game.setPaused(true); }
    const round = game.round;
    game.start();
    assert.deepEqual(game.snapshot(), {
      phase: 'running', time: 0, lane: 1, score: 0, lives: 3,
      goal: 6, duration: 35, objects: [], round: round + 1,
    });
    assert.equal(game.paused, false);
    assert.equal(game.acc, 0);
    assert.equal(game.wave, 0);
  }
});

test('terminal outcomes cannot be overwritten by movement, time or pause', () => {
  for (const game of [route('star'), route('jelly')]) {
    const before = game.snapshot();
    game.move(2); game.tick(0.5); game.setPaused(true);
    assert.deepEqual(game.snapshot(), before);
    assert.equal(game.paused, false);
  }
});

test('20ms and 100ms clocks produce the same fixed-step game', () => {
  const a = new Game(level), b = new Game(level);
  a.start(); b.start();
  for (let i = 0; i < 1500; i++) a.tick(0.02);
  for (let i = 0; i < 300; i++) b.tick(0.1);
  assert.deepEqual(a.snapshot(), b.snapshot());
});

test('caller snapshots and original level cannot change game state', () => {
  const input = structuredClone(level), game = new Game(input);
  input[0].objects[0].lane = 99;
  game.start(); game.tick(0.5); game.tick(0.2);
  const state = game.snapshot();
  state.objects[0].lane = 99;
  assert.notEqual(game.objects[0].lane, 99);
});

test('reject invalid time and lane inputs without advancing', () => {
  const game = new Game(level);
  game.start();
  const before = game.snapshot();
  for (const dt of [-1, NaN, Infinity, 0.501, '0.1']) assert.throws(() => game.tick(dt));
  for (const lane of [-1, 3, 0.5, NaN, '1']) assert.throws(() => game.move(lane));
  assert.deepEqual(game.snapshot(), before);
});

test('checked-in HTML is reproducible and has no external runtime dependencies', () => {
  const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
  assert.equal(html, build());
  assert.equal([...html.matchAll(/<script>/g)].length, 3);
  assert.doesNotMatch(html, /<script[^>]+src=|<link[^>]+href=|<img|fetch\(|WebSocket|XMLHttpRequest|<!-- (ENGINE|VIEW|APP) -->/);
});

function browserHarness() {
  const listeners = {}, raf = [], elements = {};
  const graphics = new Proxy({}, {get: (_, key) => key === 'createLinearGradient'
    ? () => ({addColorStop() {}}) : () => {}});
  for (const id of ['game', 'start', 'pause', 'gameStatus']) elements[id] = {
    hidden: false, dataset: {}, listeners: {}, textContent: '',
    closest() { return null; }, focus() {}, getContext() { return graphics; },
    getBoundingClientRect() { return {left: 0, width: 320}; },
    addEventListener(name, handler) { this.listeners[name] = handler; },
  };
  const lanes = [0, 1, 2].map(lane => ({dataset: {lane: String(lane)}}));
  const document = {
    hidden: false, getElementById: id => elements[id], querySelectorAll: () => lanes,
    addEventListener(name, handler) { listeners[name] = handler; },
  };
  const context = vm.createContext({document, window: {}, structuredClone,
    performance: {now: () => 0}, requestAnimationFrame: callback => raf.push(callback)});
  const html = build();
  for (const [, script] of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) vm.runInContext(script, context);
  const state = () => JSON.parse(vm.runInContext('JSON.stringify(snapshot())', context));
  const key = (value, target = elements.game) => listeners.keydown({
    key: value, target, repeat: false, preventDefault() {},
  });
  const frame = ms => { const callbacks = raf.splice(0); callbacks.forEach(callback => callback(ms)); };
  return {elements, lanes, document, listeners, state, key, frame};
}

test('bundled page connects play, keys, lane buttons and pause to actual state', () => {
  const ui = browserHarness();
  assert.equal(ui.state().game.phase, 'ready');
  ui.elements.start.onclick(); ui.key('ArrowLeft'); ui.frame(100);
  assert.equal(ui.state().game.lane, 0);
  assert.ok(ui.state().game.time > 0);
  ui.lanes[2].onclick();
  assert.equal(ui.state().game.lane, 2);
  ui.key(' ');
  const before = ui.state();
  ui.frame(200); ui.key('ArrowLeft');
  assert.deepEqual(ui.state(), before);
  ui.elements.pause.onclick(); ui.frame(300);
  assert.ok(ui.state().game.time > before.game.time);
  ui.elements.game.listeners.pointerdown({clientX: 1});
  assert.equal(ui.state().game.lane, 0);
  assert.match(ui.elements.gameStatus.textContent, /游戏中/);
});

test('hidden tabs pause once and do not resume on focus', () => {
  const ui = browserHarness();
  ui.elements.start.onclick(); ui.frame(100);
  ui.document.hidden = true; ui.listeners.visibilitychange(); ui.listeners.visibilitychange();
  const before = ui.state();
  assert.equal(before.stopped, true);
  ui.document.hidden = false; ui.listeners.visibilitychange(); ui.frame(60000);
  assert.deepEqual(ui.state(), before);
});

test('Space on an HTML button is left to its native activation behavior', () => {
  const ui = browserHarness();
  ui.elements.start.onclick();
  ui.key(' ', {closest: () => ({tagName: 'BUTTON'})});
  assert.equal(ui.state().stopped, false);
  ui.elements.pause.onclick();
  assert.equal(ui.state().stopped, true);
});
