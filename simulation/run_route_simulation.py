from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle

from simulation.route_dynamics import (
    SimulationConfig,
    default_domains,
    default_routes,
    default_scenarios,
    rank_scenarios,
    route_domain_applicability_table,
    route_utility_timeseries,
    run_simulation,
    simulation_config_payload,
    summarize_routes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "simulation" / "outputs"
FIGURE_DIR = PROJECT_ROOT / "figures"


ROUTE_COLORS = {
    "specialized_foundation_model": "#2F6F9F",
    "llm_research_assistant": "#6B8E23",
    "multi_agent_coscientist": "#B26A00",
    "ai_scientist_pipeline": "#8E44AD",
    "algorithmic_math_agent": "#2A9D8F",
    "self_driving_laboratory": "#C0392B",
    "hybrid_calibrated_loop": "#D1495B",
}

ROUTE_SHORT_LABELS = {
    "specialized_foundation_model": "Spec-FM",
    "llm_research_assistant": "LLM assist",
    "multi_agent_coscientist": "Co-scientist",
    "ai_scientist_pipeline": "AI scientist",
    "algorithmic_math_agent": "Math agent",
    "self_driving_laboratory": "SDL",
    "hybrid_calibrated_loop": "Hybrid-Cal",
}

SCENARIO_SHORT_LABELS = {
    "base_model_scaling": "Base model",
    "base_model_plus_evaluator_scaling": "Base + evaluator",
    "cheap_lab_future": "Cheap lab",
    "calibration_governance": "Calibration gov.",
    "no_calibration": "No calibration",
    "perfect_calibration": "Perfect calibration",
    "goodhart_benchmark_environment": "Goodhart",
    "evaluator_bottleneck": "Evaluator bottleneck",
}


def _write_tables(
    frame: pd.DataFrame,
    route_summary: pd.DataFrame,
    rankings: pd.DataFrame,
    applicability_table: pd.DataFrame,
    config: SimulationConfig,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / "yearly_route_metrics.csv", index=False)
    route_summary.to_csv(OUTPUT_DIR / "route_summary.csv", index=False)
    rankings.to_csv(OUTPUT_DIR / "scenario_rankings.csv", index=False)
    applicability_table.to_csv(
        OUTPUT_DIR / "route_domain_applicability.csv",
        index=False,
    )
    with (OUTPUT_DIR / "simulation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(simulation_config_payload(config), handle, indent=2, ensure_ascii=False)


def _add_simulation_step(frame: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
    with_step = frame.copy()
    with_step.insert(1, "simulation_step", with_step["year"] - config.start_year + 1)
    return with_step


def _augment_route_summary(
    frame: pd.DataFrame,
    route_summary: pd.DataFrame,
) -> pd.DataFrame:
    scenario_labels = frame[
        ["scenario", "calibration_mode", "scenario_label"]
    ].drop_duplicates()
    claim_metrics = (
        frame.groupby(["scenario", "calibration_mode", "route"], as_index=False)
        .agg(
            mean_claim_gap=("claim_gap", "mean"),
            mean_overclaim_gap=("overclaim_gap", "mean"),
            mean_claim_calibration_accuracy=("claim_calibration_accuracy", "mean"),
            mean_final_claim_level=("final_claim_level", "mean"),
            mean_licensed_claim_level=("licensed_claim_level", "mean"),
        )
    )
    terminal_step = int(frame["simulation_step"].max())
    terminal_metrics = (
        frame[frame["simulation_step"] == terminal_step]
        .groupby(["scenario", "calibration_mode", "route"], as_index=False)
        .agg(
            terminal_mean_overclaim_gap=("overclaim_gap", "mean"),
            terminal_mean_overclaim_rate=("overclaim_rate", "mean"),
        )
    )
    return route_summary.merge(
        scenario_labels,
        on=["scenario", "calibration_mode"],
        how="left",
    ).merge(
        claim_metrics,
        on=["scenario", "calibration_mode", "route"],
        how="left",
    ).merge(
        terminal_metrics,
        on=["scenario", "calibration_mode", "route"],
        how="left",
    )


def _augment_rankings(frame: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    scenario_labels = frame[
        ["scenario", "calibration_mode", "scenario_label"]
    ].drop_duplicates()
    claim_metrics = (
        frame.groupby(
            ["scenario", "calibration_mode", "domain", "route"],
            as_index=False,
        )
        .agg(
            mean_claim_gap=("claim_gap", "mean"),
            mean_overclaim_gap=("overclaim_gap", "mean"),
            mean_claim_calibration_accuracy=("claim_calibration_accuracy", "mean"),
        )
    )
    return rankings.merge(
        scenario_labels,
        on=["scenario", "calibration_mode"],
        how="left",
    ).merge(
        claim_metrics,
        on=["scenario", "calibration_mode", "domain", "route"],
        how="left",
    )


def _route_order() -> list[str]:
    return [route.name for route in default_routes()]


def _scenario_mode_label(row: pd.Series) -> str:
    return f"{row['scenario_label']}\n({row['calibration_mode']} calibration)"


def _plot_route_utility(frame: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    by_step = route_utility_timeseries(frame)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for route in _route_order():
        group = by_step[by_step["route"] == route]
        if group.empty:
            continue
        x = group["simulation_step"].to_numpy(dtype=float)
        y = group["mean_licensed_discovery_utility"].to_numpy(dtype=float)
        se = group["sem_licensed_discovery_utility"].to_numpy(dtype=float)
        color = ROUTE_COLORS.get(route, "#555555")
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=3.8,
            color=color,
            label=group["route_label"].iloc[0],
        )
        ax.fill_between(
            x,
            y - se,
            y + se,
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    ax.set_title("Synthetic licensed-utility diagnostic under base model scaling")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Mean synthetic licensed utility")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "route_licensed_utility.png", dpi=220)
    plt.close(fig)


def _plot_overclaim_vs_evaluator(frame: pd.DataFrame) -> None:
    terminal_step = int(frame["simulation_step"].max())
    latest = frame[frame["simulation_step"] == terminal_step]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for route in _route_order():
        group = latest[latest["route"] == route]
        if group.empty:
            continue
        ax.scatter(
            group["evaluator_strength"],
            group["overclaim_gap"],
            s=56,
            alpha=0.72,
            color=ROUTE_COLORS.get(route, "#555555"),
            label=group["route_label"].iloc[0],
        )

    ax.set_title("Overclaim gap versus evaluator strength at terminal simulation step")
    ax.set_xlabel("Evaluator strength")
    ax.set_ylabel("Mean final-over-licensed claim-level gap")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "overclaim_vs_evaluator_strength.png", dpi=220)
    plt.close(fig)


def _plot_domain_score_landscape(rankings: pd.DataFrame) -> None:
    full = rankings.copy()
    full["scenario_mode_label"] = full.apply(_scenario_mode_label, axis=1)
    scenario_order = [scenario.name for scenario in default_scenarios()]
    available_scenarios = set(full["scenario"])
    scenario_order = [scenario for scenario in scenario_order if scenario in available_scenarios]
    domain_order = [domain.name for domain in default_domains()]
    domain_order = [domain for domain in domain_order if domain in set(full["domain"])]
    domain_labels = {
        domain.name: domain.label for domain in default_domains() if domain.name in domain_order
    }
    domain_labels["open_ended_problem_formulation"] = "Open-ended"
    route_order = _route_order()

    score_min = float(full["mean_licensed_discovery_utility"].min())
    score_max = float(full["mean_licensed_discovery_utility"].max())
    if score_min < 0.0 < score_max:
        norm: Normalize | TwoSlopeNorm = TwoSlopeNorm(
            vmin=score_min,
            vcenter=0.0,
            vmax=score_max,
        )
    else:
        norm = Normalize(vmin=score_min, vmax=score_max)
    cmap = LinearSegmentedColormap.from_list(
        "licensed_utility",
        ["#9E3D55", "#F7F3EA", "#166B72"],
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15.2, 10.6),
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()
    image = None

    for axis_index, scenario in enumerate(scenario_order):
        ax = axes_flat[axis_index]
        scenario_frame = full[full["scenario"] == scenario]
        matrix = (
            scenario_frame.pivot_table(
                index="domain",
                columns="route",
                values="mean_licensed_discovery_utility",
            )
            .reindex(index=domain_order, columns=route_order)
            .to_numpy()
        )
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

        ax.set_title(
            f"{SCENARIO_SHORT_LABELS.get(scenario, scenario.replace('_', ' '))}\n"
            f"({scenario_frame['calibration_mode'].iloc[0]} calibration)",
            fontsize=8,
            pad=5,
        )
        ax.set_xticks(range(len(route_order)))
        ax.set_xticklabels(
            [ROUTE_SHORT_LABELS.get(route, route) for route in route_order],
            rotation=45,
            ha="right",
            fontsize=6,
        )
        ax.set_yticks(range(len(domain_order)))
        ax.set_yticklabels(
            [
                (
                    f"{domain_labels[domain]}\n(no ordering)"
                    if int(
                        scenario_frame.loc[
                            scenario_frame["domain"] == domain,
                            "licensed_ordering_available",
                        ].min()
                    )
                    == 0
                    else domain_labels[domain]
                )
                for domain in domain_order
            ],
            fontsize=6.4,
        )
        ax.tick_params(axis="both", length=0)
        ax.set_xticks([index - 0.5 for index in range(len(route_order) + 1)], minor=True)
        ax.set_yticks([index - 0.5 for index in range(len(domain_order) + 1)], minor=True)
        ax.grid(which="minor", color="#FFFFFF", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for row_index, domain in enumerate(domain_order):
            domain_frame = scenario_frame[scenario_frame["domain"] == domain]
            unordered = int(domain_frame["licensed_ordering_available"].min()) == 0
            if unordered:
                for col_index in range(len(route_order)):
                    ax.add_patch(
                        Rectangle(
                            (col_index - 0.5, row_index - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            hatch="///",
                            edgecolor="#8C8C8C",
                            linewidth=0.0,
                            alpha=0.55,
                        )
                    )

            for col_index, route in enumerate(route_order):
                value = matrix[row_index, col_index]
                if pd.isna(value):
                    continue
                text_color = "#111111"
                if abs(float(value)) > 0.18:
                    text_color = "#FFFFFF"
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.1,
                    color=text_color,
                )

    for unused_axis in axes_flat[len(scenario_order) :]:
        unused_axis.axis("off")

    fig.suptitle(
        "AISim-Cal score landscape: synthetic licensed-utility diagnostic",
        fontsize=11,
        y=0.972,
    )
    fig.text(
        0.5,
        0.045,
        "Columns within each panel are routes; rows are domains. Cell values are synthetic mean licensed-utility scores. Hatched rows show scores but do not license route ordering because goal clarity is below threshold.",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.895,
        top=0.89,
        bottom=0.155,
        wspace=0.11,
        hspace=0.36,
    )
    if image is not None:
        colorbar_axis = fig.add_axes([0.92, 0.22, 0.014, 0.58])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Mean synthetic licensed utility", fontsize=7)
        colorbar.ax.tick_params(labelsize=6)

    fig.savefig(FIGURE_DIR / "domain_score_landscape.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(FIGURE_DIR / "domain_score_landscape.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_calibration_ablation(route_summary: pd.DataFrame) -> None:
    ablation = (
        route_summary[
            route_summary["scenario"].isin(
                ["no_calibration", "base_model_scaling", "perfect_calibration"]
            )
        ]
        .groupby(["scenario", "scenario_label", "calibration_mode"], as_index=False)
        .agg(
            mean_overclaim_gap=("mean_overclaim_gap", "mean"),
            mean_overclaim_rate=("mean_overclaim_rate", "mean"),
        )
    )
    order = ["no_calibration", "base_model_scaling", "perfect_calibration"]
    ablation["order"] = ablation["scenario"].map({name: index for index, name in enumerate(order)})
    ablation = ablation.sort_values("order")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    labels = [
        f"{row.scenario_label}\n({row.calibration_mode})"
        for row in ablation.itertuples(index=False)
    ]
    bars = ax.bar(
        labels,
        ablation["mean_overclaim_gap"],
        color=["#A23E48", "#5B7C99", "#3A8C6E"],
        width=0.62,
    )
    ax.set_title("Calibration ablation on claim-level overstatement")
    ax.set_xlabel("Synthetic calibration condition")
    ax.set_ylabel("Mean final-over-licensed claim-level gap")
    ax.grid(True, axis="y", alpha=0.24)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "calibration_ablation.png", dpi=220)
    plt.close(fig)


def _write_notes(
    frame: pd.DataFrame,
    route_summary: pd.DataFrame,
    rankings: pd.DataFrame,
) -> None:
    base_top = route_summary[
        (route_summary["scenario"] == "base_model_scaling")
        & (route_summary["rank"] == 1)
    ].iloc[0]
    governance = route_summary[route_summary["scenario"] == "calibration_governance"]
    governance_top = governance[governance["rank"] == 1].iloc[0]
    goodhart = route_summary[route_summary["scenario"] == "goodhart_benchmark_environment"]
    goodhart_top = goodhart[goodhart["rank"] == 1].iloc[0]
    ablation = (
        route_summary[
            route_summary["scenario"].isin(
                ["no_calibration", "base_model_scaling", "perfect_calibration"]
            )
        ]
        .groupby(["scenario", "scenario_label", "calibration_mode"], as_index=False)
        .agg(mean_overclaim_gap=("mean_overclaim_gap", "mean"))
    )
    no_calibration_gap = ablation.loc[
        ablation["scenario"] == "no_calibration", "mean_overclaim_gap"
    ].iloc[0]
    base_gap = ablation.loc[
        ablation["scenario"] == "base_model_scaling", "mean_overclaim_gap"
    ].iloc[0]
    perfect_gap = ablation.loc[
        ablation["scenario"] == "perfect_calibration", "mean_overclaim_gap"
    ].iloc[0]
    math_top_score = rankings[
        (rankings["scenario"] == "base_model_scaling")
        & (rankings["domain"] == "mathematics_algorithms")
        & (rankings["rank"] == 1)
    ].iloc[0]
    terminal_step = int(frame["simulation_step"].max())

    note = f"""# Route Simulation Result Notes

This is a synthetic sensitivity model for the manuscript's calibration semantics framework. It is not an empirical forecast, and the horizontal axis in the figures should be read as stylized simulation steps rather than calendar time.

## First-order readout

- Under this illustrative parameterization, the highest mean synthetic licensed-utility diagnostic in base model scaling is assigned to `{base_top["route_label"]}`.
- Under this illustrative parameterization, the highest mean synthetic licensed-utility diagnostic in calibration governance is assigned to `{governance_top["route_label"]}`.
- Under this illustrative parameterization, the highest mean synthetic licensed-utility diagnostic in the Goodhart benchmark condition is assigned to `{goodhart_top["route_label"]}`.
- In the evaluator-rich mathematics and algorithms domain under base model scaling, the highest conditional diagnostic score is assigned to `{math_top_score["route_label"]}`.
- In the calibration ablation, mean final-over-licensed claim-level gap is `{no_calibration_gap:.3f}` with no calibration, `{base_gap:.3f}` under the imperfect/base condition, and `{perfect_gap:.3f}` with perfect calibration.
- The route-utility curves report means with propagated Monte Carlo standard-error bands over 128 draws; the bands are diagnostic uncertainty summaries, not empirical confidence intervals.
- The overclaim scatter uses the terminal simulation step `{terminal_step}` and compares claim-level overstatement against evaluator strength; it is a conditional diagnostic, not a time forecast.
- The score landscape reports all route-domain-scenario scores, not a single highlighted route. Low goal-clarity rows still show synthetic utility values, but they are hatched because the objective or scoring rule is not yet clear enough to license a route ordering.
- Route ordering is also conditional on route-domain applicability. A strict evaluator does not make a route generally applicable outside domains where its operators are compatible with the task.

## Interpretation

The simulation operationalizes a central manuscript claim: as foundation-model capability improves, the decisive bottleneck shifts toward evaluator quality, calibration quality, cost, update loops, and route-domain applicability. Routes with strong external evaluators do well in the synthetic diagnostic when claims can be checked cheaply and when the route is compatible with the target domain. Routes with high throughput but weaker calibration can generate higher apparent productivity while carrying larger overclaim burdens. These readouts are diagnostic consequences of the chosen parameterization, not evidence of real-world route superiority.

The score landscape is a full route-domain-scenario matrix. Each panel corresponds to one scenario and calibration mode; within a panel, every route receives a continuous synthetic licensed-utility diagnostic score for every domain. Hatched low-clarity rows do not mean that no numerical score exists. They mean that the score should be read as an exploratory diagnostic rather than as a licensed ordering, because the target is still an open-ended problem-formulation task. In that regime, the scientific work moves upstream to defining the objective, building the evaluator, and specifying what would count as a licensed claim.

## Limits

All parameter values are stylized and synthetic. The model is useful for generating figures, stress-testing the formal vocabulary, and making assumptions explicit. It should not be used as a quantitative projection of scientific output without empirical calibration.
"""
    (OUTPUT_DIR / "result_notes.md").write_text(note, encoding="utf-8")


def run() -> None:
    config = SimulationConfig()
    frame = run_simulation(
        routes=default_routes(),
        domains=default_domains(),
        config=config,
    )
    frame = _add_simulation_step(frame, config)
    route_summary = _augment_route_summary(frame, summarize_routes(frame))
    rankings = _augment_rankings(frame, rank_scenarios(frame))
    applicability_table = route_domain_applicability_table(
        routes=default_routes(),
        domains=default_domains(),
    )

    _write_tables(frame, route_summary, rankings, applicability_table, config)
    _plot_route_utility(frame)
    _plot_overclaim_vs_evaluator(frame)
    _plot_domain_score_landscape(rankings)
    _plot_calibration_ablation(route_summary)
    _write_notes(frame, route_summary, rankings)

    print(f"Wrote simulation outputs to {OUTPUT_DIR}")
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    run()
