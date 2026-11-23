from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

STUDIES = [
    "baseline27_240",
    "partition9_27",
    "partition36_27",
    "partition144_27",
]

CLOCK = 120
CANDIDATE = "strong"
ROOT = Path("benchmark/data/route_search")


def load_case(study):
    folder = ROOT / study / "w8401_fleet27_rollout"

    def read(file):
        if not file.exists():
            raise FileNotFoundError(f"Missing: {file}")
        return json.loads(file.read_text())

    layout = read(folder / "map.json")["layout"]
    road = np.array([
        [cell not in "@T" for cell in line]
        for line in layout
    ])

    scenario = read(folder / "scenario.json")
    response = read(
        folder / "bridge" / f"response_{CLOCK}.json"
    )

    candidate = next(
        c for c in response["route_candidates"]
        if c["name"] == CANDIDATE
    )

    costs = np.array(
        candidate["cell_penalties"], dtype=float
    ).reshape(road.shape)

    factor = {
        "baseline27_240": 0,
        "partition9_27": 1,
        "partition36_27": 2,
        "partition144_27": 4,
    }[study]

    return road, scenario, costs, factor


def draw_boundaries(ax, scenario, factor):
    if factor == 0:
        return

    for zone in scenario["zones"]:
        r0, r1 = zone["bounds"]["rows"]
        c0, c1 = zone["bounds"]["cols"]

        rows = np.array_split(
            np.arange(r0, r1 + 1), factor
        )
        cols = np.array_split(
            np.arange(c0, c1 + 1), factor
        )

        for rr in rows:
            for cc in cols:
                ax.add_patch(
                    Rectangle(
                        (cc[0] - 0.5, rr[0] - 0.5),
                        len(cc),
                        len(rr),
                        fill=False,
                        edgecolor="#38bdf8",
                        linewidth=0.7,
                        alpha=0.8,
                    )
                )


cases = [load_case(study) for study in STUDIES]

if any(
    not np.array_equal(case[0], cases[0][0])
    for case in cases
):
    raise ValueError("Warehouse maps differ between studies.")

vmax = max(float(case[2].max()) for case in cases)

fig, axes = plt.subplots(
    2, 4, figsize=(18, 9), constrained_layout=True
)

heat_cmap = plt.get_cmap("magma").copy()
heat_cmap.set_bad("#303743")

for col, (study, case) in enumerate(zip(STUDIES, cases)):
    road, scenario, costs, factor = case

    top = axes[0, col]
    bottom = axes[1, col]

    # Top row: roads and region boundaries
    top.imshow(
        road,
        cmap=ListedColormap(["#303743", "#f8fafc"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    draw_boundaries(top, scenario, factor)
    top.set_title(f"{study}\nRegion boundaries")

    # Bottom row: additional PBS movement costs
    masked = np.ma.masked_where(~road, costs)
    image = bottom.imshow(
        masked,
        cmap=heat_cmap,
        vmin=0,
        vmax=max(vmax, 1e-6),
        interpolation="nearest",
    )
    draw_boundaries(bottom, scenario, factor)
    bottom.set_title(
        f"Movement penalties (max {costs.max():.2f})"
    )

    for ax in (top, bottom):
        ax.set_xticks([])
        ax.set_yticks([])

    print(
        f"{study}: mean road penalty = "
        f"{costs[road].mean():.4f}, "
        f"maximum = {costs.max():.4f}"
    )

fig.colorbar(
    image,
    ax=axes[1, :].tolist(),
    shrink=0.75,
    label="Additional movement cost entering a cell",
)

output = ROOT / "visualizations"
output.mkdir(parents=True, exist_ok=True)

path = output / f"compare_costs_tick{CLOCK}_{CANDIDATE}.png"
fig.savefig(path, dpi=180)
plt.close(fig)

print(f"\nSaved image: {path.resolve()}")
