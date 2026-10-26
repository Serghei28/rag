"""Графики экспериментов в SVG на чистом Python.

Почему не matplotlib: на машине разработки включён Windows Smart App Control, и он
блокирует DLL matplotlib (ft2font) и из PyPI, и из conda-forge. SVG — текстовый формат:
не требует бинарных библиотек, отображается в GitHub, VS Code и браузере, масштабируется
без потери качества.
"""

from html import escape
from pathlib import Path

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
W, H = 760, 420
LEFT, RIGHT, TOP, BOTTOM = 60, 180, 40, 90  # поля: справа легенда, снизу подписи


def _frame(title: str, y_max: float = 1.0) -> list[str]:
    plot_h = H - TOP - BOTTOM
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="Segoe UI, Arial, sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W / 2}" y="22" text-anchor="middle" font-size="15" font-weight="bold">'
        f"{escape(title)}</text>",
    ]
    for i in range(6):  # горизонтальная сетка 0, 0.2 … 1.0
        value = y_max * i / 5
        y = TOP + plot_h * (1 - i / 5)
        parts.append(
            f'<line x1="{LEFT}" y1="{y:.1f}" x2="{W - RIGHT}" y2="{y:.1f}" stroke="#ddd"/>'
        )
        parts.append(f'<text x="{LEFT - 6}" y="{y + 4:.1f}" text-anchor="end">{value:.1f}</text>')
    return parts


def _y(value: float, y_max: float = 1.0) -> float:
    return TOP + (H - TOP - BOTTOM) * (1 - min(max(value, 0.0), y_max) / y_max)


def _legend(names: list[str]) -> list[str]:
    parts = []
    for i, name in enumerate(names):
        y = TOP + 10 + i * 20
        x = W - RIGHT + 15
        parts.append(
            f'<rect x="{x}" y="{y - 9}" width="12" height="12" fill="{PALETTE[i % len(PALETTE)]}"/>'
        )
        parts.append(f'<text x="{x + 18}" y="{y + 1}">{escape(name)}</text>')
    return parts


def bar_chart(path: Path, title: str, labels: list[str], series: dict[str, list[float | None]]):
    """Сгруппированные столбцы: по группе на каждую метку, по столбцу на каждую серию."""
    parts = _frame(title)
    plot_w = W - LEFT - RIGHT
    group_w = plot_w / max(len(labels), 1)
    bar_w = group_w * 0.8 / max(len(series), 1)
    for s, values in enumerate(series.values()):
        color = PALETTE[s % len(PALETTE)]
        for g, value in enumerate(values):
            if value is None:
                continue
            x = LEFT + g * group_w + group_w * 0.1 + s * bar_w
            y = _y(value)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" '
                f'height="{H - BOTTOM - y:.1f}" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2 - 1:.1f}" y="{y - 3:.1f}" '
                f'text-anchor="middle" font-size="9">{value:.2f}</text>'
            )
    for g, label in enumerate(labels):
        x = LEFT + (g + 0.5) * group_w
        parts.append(
            f'<text x="{x:.1f}" y="{H - BOTTOM + 14}" text-anchor="end" '
            f'transform="rotate(-30 {x:.1f} {H - BOTTOM + 14})">{escape(label)}</text>'
        )
    parts += _legend(list(series)) + ["</svg>"]
    path.write_text("\n".join(parts), encoding="utf-8")


def line_chart(
    path: Path, title: str, x_label: str, series: dict[str, list[tuple[float, float | None]]]
):
    """Линии по числовой оси X (размер chunk, порог, K)."""
    parts = _frame(title)
    xs = sorted({x for points in series.values() for x, _ in points})
    lo, hi = xs[0], xs[-1]
    span = (hi - lo) or 1.0
    plot_w = W - LEFT - RIGHT
    # Если точки слишком тесно (пороги 0, 0.01, 0.05 …), подписи налезают —
    # тогда ставим точки равномерно, как категории
    min_gap = min((b - a for a, b in zip(xs, xs[1:], strict=False)), default=span)
    even = len(xs) > 2 and min_gap / span < 0.04

    def px(x: float) -> float:
        if even:
            return LEFT + plot_w * xs.index(x) / (len(xs) - 1)
        return LEFT + plot_w * (x - lo) / span

    for x in xs:
        parts.append(
            f'<text x="{px(x):.1f}" y="{H - BOTTOM + 16}" text-anchor="middle">{x:g}</text>'
        )
    parts.append(
        f'<text x="{LEFT + plot_w / 2}" y="{H - BOTTOM + 40}" text-anchor="middle">'
        f"{escape(x_label)}</text>"
    )
    for s, points in enumerate(series.values()):
        color = PALETTE[s % len(PALETTE)]
        pts = [(px(x), _y(v)) for x, v in sorted(points) if v is not None]
        if len(pts) > 1:
            path_d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(
                f'<polyline points="{path_d}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
    parts += _legend(list(series)) + ["</svg>"]
    path.write_text("\n".join(parts), encoding="utf-8")
