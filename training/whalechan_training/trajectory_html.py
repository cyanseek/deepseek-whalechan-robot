"""Self-contained SVG diagnostics for recorded trajectories; no browser runtime."""

from html import escape
import math


def _number(value, bounds=None):
    try:
        value = float(value)
        if math.isfinite(value) and (bounds is None or bounds[0] <= value <= bounds[1]):
            return value
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _segments(times, values, *, wrapped=False):
    """Do not draw across missing/invalid/reordered samples or heading wraps."""
    segments, current = [], []
    for t, value in zip(times, values):
        if value is None:
            if current:
                segments.append(current)
            current = []
            continue
        if current and (not 0 < t - current[-1][0] <= .0200001
                        or (wrapped and abs(value - current[-1][1]) > 180)):
            segments.append(current)
            current = []
        current.append((t, value))
    if current:
        segments.append(current)
    return segments


def _chart(title, unit, times, series, *, wrapped=False):
    left, top, width, height = 66, 16, 820, 120
    xmin, xmax = min([0., *times]), max([20., *times])
    numbers = [v for _, _, values in series for v in values if v is not None]
    low, high = min([0., *numbers]), max([0., *numbers])
    span = high - low or 1.
    low, high = low - .08 * span, high + .08 * span
    if not all(math.isfinite(v) for v in [xmax - xmin, low, high, high - low]):
        raise ValueError("Trajectory values exceed the plot's numeric range")
    def x(t): return left + (t - xmin) / (xmax - xmin) * width
    def y(v): return top + (high - v) / (high - low) * height
    svg = [f'<svg viewBox="0 0 920 172" role="img" aria-label="{escape(title)}">',
           f'<title>{escape(title)} ({escape(unit)})</title>']
    for start, end in [(8., 11.), (17., 20.)]:
        svg.append(f'<rect x="{x(start):.3f}" y="{top}" width="{x(end)-x(start):.3f}" '
                   f'height="{height}" fill="#dff1eb"/>')
    for k in range(4):
        value = low + (high - low) * k / 3
        svg.append(f'<path d="M {left} {y(value):.3f} H {left+width}" stroke="#dfe6ee"/>')
        svg.append(f'<text x="{left-9}" y="{y(value)+4:.3f}" text-anchor="end">{value:.3g}</text>')
    for k in range(6):
        t = xmin + (xmax - xmin) * k / 5
        svg.append(f'<text x="{x(t):.3f}" y="158" text-anchor="middle">{t:.3g}s</text>')
    legends = []
    for label, color, values in series:
        legends.append(f'<span><i style="background:{color}"></i>{escape(label)}</span>')
        for segment in _segments(times, values, wrapped=wrapped):
            if len(segment) == 1:
                t, value = segment[0]
                svg.append(f'<circle cx="{x(t):.3f}" cy="{y(value):.3f}" r="2" fill="{color}"/>')
            else:
                points = ' '.join(f'{x(t):.3f},{y(value):.3f}' for t, value in segment)
                svg.append(f'<polyline points="{points}" stroke="{color}" stroke-width="1.6" fill="none"/>')
    svg.append('</svg>')
    missing = sum(v is None for _, _, values in series for v in values)
    note = f' · 缺失/无效值 Missing/invalid: {missing}' if missing else ''
    if not numbers:
        note += ' · 无有效采样 No valid samples'
    return (f'<section><h2>{escape(title)} <small>{escape(unit)}</small></h2>'
            f'<div class="legend">{"".join(legends)}{note}</div>{"".join(svg)}</section>')


def trajectory_html(report, rows, *, source_name):
    """Render the same parsed samples and report used by the JSON analyzer."""
    times = [r['t'] for r in rows]
    command = [r['command'][0] for r in rows]
    speed = [_number(r['speed_m_s']) for r in rows]
    tilt = [_number(r.get('tilt_rad'), (0., math.pi)) for r in rows]
    heading = [_number(r.get('heading_error_rad'), (-math.pi, math.pi)) for r in rows]
    tilt = [math.degrees(v) if v is not None else None for v in tilt]
    heading = [math.degrees(v) if v is not None else None for v in heading]
    charts = _chart('前向速度 / Forward speed', 'm/s', times,
                    [('指令 Command', '#8395aa', command), ('实测 Measured', '#126eaa', speed)])
    charts += _chart('机身倾斜 / Body tilt', 'deg', times, [('倾斜 Tilt', '#9261b2', tilt)])
    charts += _chart('相对初始方向的偏航 / Initial-heading error', 'deg', times,
                     [('有符号偏航 Signed heading', '#d37b2d', heading)], wrapped=True)
    def fmt(value): return '—' if value is None else f'{value:.3f}'
    table = []
    for w in report['posture_windows']:
        state = '完整 Complete' if w['complete'] else '不完整 Partial'
        phase = '行走 Walk' if w['phase'] == 'walk' else '停止 Stop'
        table.append(f'<tr><td>{phase} {w["command_s"]:g}–{w["window_end_s"]:g}s</td>'
                     f'<td>{w["samples"]} · {state}</td><td>{w["invalid_samples"]}</td>'
                     f'<td>{fmt(w["sampled_max_tilt_deg"])}</td>'
                     f'<td>{fmt(w["final_heading_error_deg"])}</td></tr>')
    return f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>轨迹诊断 · Trajectory diagnostics</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef3f8;color:#20334b;font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1020px;margin:auto;padding:28px 24px}}h1{{font-size:26px;margin:8px 0}}h2{{font-size:16px;margin:0}}
p{{margin:6px 0}}.tag{{color:#126eaa;font-weight:700;letter-spacing:1px}}.muted,small{{color:#63758b;font-weight:400}}
section{{background:white;border:1px solid #dae3ed;border-radius:12px;padding:14px 18px;margin:14px 0}}
svg{{width:100%;display:block}}svg text{{font:12px system-ui;fill:#63758b}}.legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin:4px 0}}
i{{display:inline-block;width:16px;height:3px;vertical-align:middle;margin-right:6px}}code{{font-size:11px;overflow-wrap:anywhere}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #e7edf3}}
th{{font-size:12px;color:#63758b}}@media(max-width:600px){{main{{padding:16px 10px}}section{{padding:12px 6px}}h1{{font-size:22px}}}}
</style><main><div class="tag">WHALE-CHAN · OFFLINE DIAGNOSTICS</div>
<h1>看清每一步，也看清停下之后。</h1>
<p>{escape(source_name)} · {report['recorded_samples']} 帧 / recorded samples · 初始等待 / initial stand: {report['initial_stand_s']}s</p>
<p class="muted">离线记录分析，不是新仿真或步态验收。Offline analysis, not a new rollout or qualification.</p>
<p class="muted">绿色背景为预定停止时段；缺帧、无效值和偏航角度跨界处断线，不补帧。Green: scheduled stops; gaps, invalid values and heading wraps break the curves.</p>
{charts}<section><h2>分段姿态 / Posture by period</h2><div class="scroll"><table>
<thead><tr><th>时段 Period</th><th>采样 Samples</th><th>无效 Invalid</th><th>采样最大倾斜 Peak tilt °</th><th>末帧偏航 Final heading °</th></tr></thead>
<tbody>{''.join(table)}</tbody></table></div>
<p class="muted">完整表示采样齐全，不代表平衡达标。50 Hz采样峰值可能漏掉帧间峰值。Complete means recording coverage, not balance; 50 Hz peaks may miss between-frame peaks.</p></section>
<p class="muted">Source SHA-256 <code>{escape(report['source_sha256'])}</code></p>
</main></html>'''
