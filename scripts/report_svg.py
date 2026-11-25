from __future__ import annotations

import html
import math
import textwrap

PALETTE = {
    "data": ("#dbeafe", "#1d4ed8"),
    "gate": ("#fef3c7", "#b45309"),
    "ml": ("#ccfbf1", "#0f766e"),
    "ops": ("#ede9fe", "#6d28d9"),
    "api": ("#dcfce7", "#15803d"),
    "ext": ("#f3f4f6", "#6b7280"),
    "bad": ("#fee2e2", "#b91c1c"),
    "ok": ("#dcfce7", "#15803d"),
    "feat": ("#fce7f3", "#be185d"),
}


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    ticks, v = [], math.floor(lo / step) * step
    while v < hi - 1e-9 or len(ticks) < 2:
        ticks.append(round(v, 10))
        v += step
    ticks.append(round(v, 10)) if ticks[-1] < hi - 1e-9 else None
    return ticks


def lines(text: str, width: int) -> list[str]:
    return textwrap.wrap(str(text), width=width) or [""]


def box(x, y, w, h, text, kind="ml", size=12, dashed=False, bold_first=True) -> str:
    fill, stroke = PALETTE[kind]
    wrap_w = max(6, int(w / (size * 0.56)))
    rows = lines(text, wrap_w)
    total = len(rows) * (size + 3)
    y0 = y + (h - total) / 2 + size
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    tspans = ""
    for i, row in enumerate(rows):
        weight = ' font-weight="700"' if (i == 0 and bold_first and len(rows) > 1) else ""
        tspans += f'<tspan x="{x + w / 2}" y="{y0 + i * (size + 3)}"{weight}>{esc(row)}</tspan>'
    return (
        f'<g class="nd"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dash}/>'
        f'<text text-anchor="middle" font-size="{size}" fill="#111827">{tspans}</text></g>'
    )


def arrow(x1, y1, x2, y2, label="", dashed=False, color="#475569") -> str:
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    out = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"{dash} marker-end="url(#ah)"/>'
    if label:
        out += f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 5}" text-anchor="middle" font-size="10.5" class="lbl">{esc(label)}</text>'
    return out


def poly_arrow(points, label="", dashed=False, color="#475569") -> str:
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    pts = " ".join(f"{x},{y}" for x, y in points)
    out = f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"{dash} marker-end="url(#ah)"/>'
    if label:
        mx, my = points[len(points) // 2]
        out += f'<text x="{mx}" y="{my - 5}" text-anchor="middle" font-size="10.5" class="lbl">{esc(label)}</text>'
    return out


def svg(width, height, body, title="") -> str:
    defs = '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>'
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}" class="fig">{defs}{body}</svg>'


def label(x, y, text, size=12, anchor="start", cls="txt", weight="400") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" class="{cls}" font-weight="{weight}">{esc(text)}</text>'


def hbars(rows, width=900, bar_h=20, gap=8, left=230, maxv=None, fmt=lambda v: f"{v:,}", colors=None, title="") -> str:
    maxv = maxv or max((v for _, v in rows), default=1) or 1
    h = len(rows) * (bar_h + gap) + 16
    plot_w = width - left - 90
    body = ""
    for i, (name, v) in enumerate(rows):
        y = 8 + i * (bar_h + gap)
        w = max(1.5, plot_w * v / maxv)
        col = (colors[i] if colors else "#0f766e")
        body += label(left - 8, y + bar_h * 0.72, name, 12, "end")
        body += f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{col}"/>'
        body += label(left + w + 6, y + bar_h * 0.72, fmt(v), 12)
    return svg(width, h, body, title)


def vbars_grouped(groups, series, width=900, height=300, ymax=None, fmt=lambda v: f"{v:g}", colors=("#0f766e", "#b45309", "#6d28d9"), title="", ymin=0.0) -> str:
    left, right, top, bottom = 56, 16, 22, 62
    data_max = max(v for g in groups for v in g[1])
    data_min = min(v for g in groups for v in g[1])
    ticks = nice_ticks(min(ymin, data_min), max(ymax or 0, data_max * 1.08))
    lo, hi = ticks[0], ticks[-1]
    plot_w, plot_h = width - left - right, height - top - bottom
    ypos = lambda v: top + plot_h - plot_h * (v - lo) / (hi - lo)
    n, k = len(groups), len(series)
    gw = plot_w / n
    bw = min(46, (gw * 0.78) / k)
    base = ypos(0.0 if lo <= 0 <= hi else lo)
    body = ""
    for t in ticks:
        body += f'<line x1="{left}" y1="{ypos(t):.1f}" x2="{width - right}" y2="{ypos(t):.1f}" class="grid"/>' + label(left - 6, ypos(t) + 4, fmt(t), 10.5, "end", "lbl")
    for gi, (name, vals) in enumerate(groups):
        x0 = left + gi * gw + (gw - bw * k) / 2
        for si, v in enumerate(vals):
            y = ypos(v)
            top_y, hh = min(y, base), abs(y - base)
            body += f'<rect x="{x0 + si * bw:.1f}" y="{top_y:.1f}" width="{bw - 2:.1f}" height="{max(hh, 1):.1f}" rx="3" fill="{colors[si % len(colors)]}"/>'
            ly = y - 4 if v >= 0 else y + 12
            body += label(x0 + si * bw + (bw - 2) / 2, ly, fmt(v), 10, "middle", "lbl")
        for li, row in enumerate(lines(name, max(8, int(gw / 6.2)))[:3]):
            body += label(left + gi * gw + gw / 2, top + plot_h + 16 + li * 13, row, 11, "middle")
    lx = left
    for si, sname in enumerate(series):
        body += f'<rect x="{lx}" y="{height - 14}" width="11" height="11" rx="2" fill="{colors[si % len(colors)]}"/>' + label(lx + 16, height - 4, sname, 11)
        lx += 24 + len(sname) * 6.6
    return svg(width, height, body, title)


def line_chart(series, xs, width=900, height=300, ymax=None, fmt=lambda v: f"{v:.1f}", colors=None, title="") -> str:
    colors = colors or ["#0f766e", "#b45309", "#6d28d9", "#be185d", "#1d4ed8", "#15803d"]
    left, right, top, bottom = 52, 210, 16, 34
    ticks = nice_ticks(0.0, max(ymax or 0, max(max(v) for _, v in series) * 1.08))
    hi = ticks[-1]
    plot_w, plot_h = width - left - right, height - top - bottom
    ypos = lambda v: top + plot_h - plot_h * v / hi
    body = ""
    for t in ticks:
        body += f'<line x1="{left}" y1="{ypos(t):.1f}" x2="{left + plot_w}" y2="{ypos(t):.1f}" class="grid"/>' + label(left - 6, ypos(t) + 4, fmt(t), 10.5, "end", "lbl")
    for i, x in enumerate(xs):
        body += label(left + plot_w * i / (len(xs) - 1), height - 12, x, 10.5, "middle", "lbl")
    ends = []
    for si, (name, vals) in enumerate(series):
        pts = " ".join(f"{left + plot_w * i / (len(xs) - 1):.1f},{ypos(v):.1f}" for i, v in enumerate(vals))
        c = colors[si % len(colors)]
        body += f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.2"/>'
        ends.append([ypos(vals[-1]), name, c])
    ends.sort()
    for j in range(1, len(ends)):
        if ends[j][0] - ends[j - 1][0] < 14:
            ends[j][0] = ends[j - 1][0] + 14
    ex = left + plot_w
    for y, name, c in ends:
        body += f'<circle cx="{ex:.1f}" cy="{y:.1f}" r="3.4" fill="{c}"/>' + label(ex + 8, y + 4, name, 11, "start", "txt")
    return svg(width, height, body, title)
