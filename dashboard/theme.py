"""
Shared Plotly styling, ported from MarketPulse's dashboard/theme.py - the
chrome, categorical palette, and build_template/apply_theme are
domain-agnostic and reused as-is. Only the domain-specific color maps below
(sector/risk-tier in MarketPulse) are swapped for this project's own
(bandit arm / auction outcome).
"""
import plotly.graph_objects as go
import plotly.io as pio

# --- Chrome & ink ---
SURFACE = "#fcfcfb"
PAGE_PLANE = "#f9f9f7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# --- Categorical (fixed order - never cycle the assignment within one chart) ---
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]

# --- Sequential (single hue, light -> dark; magnitude only) ---
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

# --- Status (state, never reused as a categorical series color) ---
STATUS = {
    "good": "#0ca30c",     # win / clicked
    "warning": "#fab219",  # running
    "critical": "#d03b3b", # loss / budget exhausted
}

# Fixed bid-multiplier -> categorical slot mapping so a given arm is always
# the same color across charts, regardless of which arms happen to have
# been played in a given run.
ARM_COLOR = {
    0.5: CATEGORICAL[0],
    0.75: CATEGORICAL[1],
    1.0: CATEGORICAL[2],
    1.25: CATEGORICAL[3],
    1.5: CATEGORICAL[4],
    2.0: CATEGORICAL[5],
}

OUTCOME_COLOR = {
    "won": STATUS["good"],
    "lost": STATUS["critical"],
}


def build_template() -> go.layout.Template:
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=TEXT_PRIMARY, size=13),
        colorway=CATEGORICAL,
        xaxis=dict(gridcolor=GRIDLINE, linecolor=BASELINE, zerolinecolor=BASELINE, tickfont=dict(color=TEXT_MUTED)),
        yaxis=dict(gridcolor=GRIDLINE, linecolor=BASELINE, zerolinecolor=BASELINE, tickfont=dict(color=TEXT_MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SECONDARY)),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return template


def apply_theme() -> None:
    pio.templates["ad_auction_optimizer"] = build_template()
    pio.templates.default = "ad_auction_optimizer"
