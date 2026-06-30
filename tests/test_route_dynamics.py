from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from simulation.route_dynamics import (
    CLAIM_LEVEL_THRESHOLDS,
    LICENSED_ORDERING_GOAL_CLARITY_THRESHOLD,
    ScenarioParameters,
    SimulationConfig,
    claim_level_from_score,
    default_domains,
    default_routes,
    evaluator_strength,
    foundation_level,
    rank_scenarios,
    route_domain_applicability,
    route_domain_applicability_table,
    route_utility_timeseries,
    run_simulation,
    summarize_routes,
)


def test_default_routes_are_complete():
    routes = default_routes()
    names = [route.name for route in routes]

    assert len(names) == 7
    assert len(set(names)) == len(names)
    assert {
        "specialized_foundation_model",
        "llm_research_assistant",
        "multi_agent_coscientist",
        "ai_scientist_pipeline",
        "algorithmic_math_agent",
        "self_driving_laboratory",
        "hybrid_calibrated_loop",
    } == set(names)


def test_default_domains_include_low_clarity_problem_formulation_domain():
    domains = default_domains()
    by_name = {domain.name: domain for domain in domains}

    assert "open_ended_problem_formulation" in by_name
    assert (
        by_name["open_ended_problem_formulation"].goal_clarity
        < LICENSED_ORDERING_GOAL_CLARITY_THRESHOLD
    )


def test_default_domains_constrain_route_domain_applicability():
    domains = {domain.name: domain for domain in default_domains()}

    biology = domains["biology"].route_applicability
    mathematics = domains["mathematics_algorithms"].route_applicability

    assert biology["algorithmic_math_agent"] < biology["multi_agent_coscientist"]
    assert biology["algorithmic_math_agent"] < biology["self_driving_laboratory"]
    assert mathematics["algorithmic_math_agent"] > biology["algorithmic_math_agent"]


def test_route_domain_applicability_table_is_complete_and_bounded():
    table = route_domain_applicability_table(
        routes=default_routes(),
        domains=default_domains(),
    )

    assert set(table["route"]) == {route.name for route in default_routes()}
    assert set(table["domain"]) == {domain.name for domain in default_domains()}
    assert len(table) == len(default_routes()) * len(default_domains())
    assert table["route_domain_applicability"].between(0.0, 1.0).all()


def test_route_domain_applicability_requires_explicit_route_key():
    route = replace(default_routes()[0], name="unlisted_route")
    domain = default_domains()[0]

    with pytest.raises(KeyError, match="Missing route-domain applicability"):
        route_domain_applicability(route, domain)


def test_claim_ladder_threshold_boundaries():
    assert claim_level_from_score(0.0) == 0
    assert claim_level_from_score(CLAIM_LEVEL_THRESHOLDS[0] - 1e-9) == 0
    for level, threshold in enumerate(CLAIM_LEVEL_THRESHOLDS):
        assert claim_level_from_score(threshold) == level
        if level > 0:
            assert claim_level_from_score(threshold - 1e-9) == level - 1
    assert claim_level_from_score(1.0) == 6


def test_evaluator_strength_is_monotone_in_components():
    baseline = evaluator_strength(independence=0.45, reliability=0.55, grounding=0.65)

    assert evaluator_strength(0.55, 0.55, 0.65) > baseline
    assert evaluator_strength(0.45, 0.65, 0.65) > baseline
    assert evaluator_strength(0.45, 0.55, 0.75) > baseline
    assert evaluator_strength(0.55, 0.65, 0.75) > baseline
    assert evaluator_strength(0.0, 0.8, 0.8) == 0.0


def test_foundation_level_increases_monotonically():
    config = SimulationConfig(start_year=2026, end_year=2035)
    levels = [foundation_level(year, config) for year in config.years]

    assert all(0.0 <= level <= 1.0 for level in levels)
    assert all(left <= right for left, right in zip(levels, levels[1:]))
    assert levels[-1] > levels[0]


def test_same_seed_reproduces_identical_dataframes():
    config = SimulationConfig(
        start_year=2026,
        end_year=2028,
        monte_carlo_runs=16,
        random_seed=997,
    )

    first = run_simulation(config=config, scenarios=["base_model_scaling"])
    second = run_simulation(config=config, scenarios=["base_model_scaling"])

    pd.testing.assert_frame_equal(first, second)


def test_scenario_order_does_not_change_scenario_rows():
    config = SimulationConfig(
        start_year=2026,
        end_year=2028,
        monte_carlo_runs=16,
        random_seed=997,
    )
    solo = run_simulation(config=config, scenarios=["perfect_calibration"])
    combined = run_simulation(
        config=config,
        scenarios=[
            "base_model_scaling",
            "perfect_calibration",
            "no_calibration",
        ],
    )
    filtered = combined[combined["scenario"] == "perfect_calibration"].reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(solo, filtered)


def test_perfect_calibration_never_exceeds_licensed_claim_level():
    config = SimulationConfig(start_year=2026, end_year=2030, monte_carlo_runs=32)
    frame = run_simulation(config=config, scenarios=["perfect_calibration"])

    assert (frame["final_claim_level"] <= frame["licensed_claim_level"]).all()


def test_custom_scenarios_can_share_seed_key_across_calibration_modes():
    config = SimulationConfig(start_year=2026, end_year=2030, monte_carlo_runs=32)
    shared_parameters = {
        "label": "Paired calibration mode",
        "seed_key": "paired_calibration_mode",
        "foundation_shift": 0.02,
        "evaluator_shift": 0.01,
        "evaluator_multiplier": 1.03,
        "lab_cost_shift": 0.91,
        "calibration_shift": 0.04,
        "overclaim_shift": 0.03,
        "noise_shift": 1.02,
        "evaluator_model_coupling": 0.07,
        "benchmark_goodhart": 0.02,
    }
    no_calibration = ScenarioParameters(
        **shared_parameters,
        name="paired_no_calibration",
        calibration_mode="none",
    )
    perfect_calibration = ScenarioParameters(
        **shared_parameters,
        name="paired_perfect_calibration",
        calibration_mode="perfect",
    )

    no_calibration_frame = run_simulation(config=config, scenarios=[no_calibration])
    perfect_calibration_frame = run_simulation(
        config=config,
        scenarios=[perfect_calibration],
    )

    pd.testing.assert_series_equal(
        no_calibration_frame["raw_claim_strength"],
        perfect_calibration_frame["raw_claim_strength"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        no_calibration_frame["evidence_licensed_claim"],
        perfect_calibration_frame["evidence_licensed_claim"],
        check_names=False,
    )
    assert no_calibration_frame["calibration_mode"].unique().tolist() == ["none"]
    assert perfect_calibration_frame["calibration_mode"].unique().tolist() == [
        "perfect"
    ]

    assert (
        no_calibration_frame["overclaim_gap"].mean()
        >= perfect_calibration_frame["overclaim_gap"].mean()
    )


def test_calibration_governance_reduces_overclaim():
    config = SimulationConfig(start_year=2026, end_year=2035, monte_carlo_runs=64)
    frame = run_simulation(
        routes=default_routes(),
        domains=default_domains(),
        config=config,
        scenarios=["base_model_scaling", "calibration_governance"],
    )

    overclaim = frame.groupby("scenario")["overclaim_rate"].mean()

    assert overclaim["calibration_governance"] < overclaim["base_model_scaling"]


def test_rank_scenarios_assigns_contiguous_route_ranks_per_scenario_domain():
    config = SimulationConfig(start_year=2026, end_year=2035, monte_carlo_runs=64)
    frame = run_simulation(
        routes=default_routes(),
        domains=default_domains(),
        config=config,
        scenarios=["base_model_scaling"],
    )
    ranking = rank_scenarios(frame)
    expected_ranks = list(range(1, len(default_routes()) + 1))

    for _, group in ranking.groupby(["scenario", "calibration_mode", "domain"]):
        if int(group["licensed_ordering_available"].iloc[0]) == 0:
            assert group["rank"].isna().all()
            continue

        ranks = sorted(group["rank"].dropna().astype(int).tolist())
        assert ranks == expected_ranks
        assert ranks.count(1) == 1


def test_invalid_calibration_mode_raises_before_route_iteration():
    class ExplodingRoutes:
        def __iter__(self):
            raise AssertionError("routes should not be iterated")

    invalid_scenario = ScenarioParameters(
        name="invalid_calibration",
        label="Invalid calibration",
        calibration_mode="optimistic",
    )

    with pytest.raises(ValueError, match="Unknown calibration mode: optimistic"):
        run_simulation(routes=ExplodingRoutes(), scenarios=[invalid_scenario])


def test_run_simulation_has_expected_columns():
    config = SimulationConfig(start_year=2026, end_year=2028, monte_carlo_runs=16)
    frame = run_simulation(
        routes=default_routes(),
        domains=default_domains(),
        config=config,
        scenarios=["base_model_scaling"],
    )

    expected_columns = {
        "year",
        "scenario",
        "calibration_mode",
        "domain",
        "goal_clarity",
        "route_domain_applicability",
        "licensed_ordering_available",
        "route",
        "foundation_level",
        "evidential_support",
        "evidence_licensed_claim",
        "true_licensed_claims",
        "overclaim_rate",
        "false_discovery_burden",
        "licensed_discovery_utility",
        "cost_normalized_utility",
        "raw_claim_level",
        "licensed_claim_level",
        "final_claim_level",
        "overclaim_gap",
        "claim_calibration_accuracy",
        "evidence_bottleneck_index",
        "evaluator_independence",
        "evaluator_reliability",
        "evaluator_grounding",
    }

    assert expected_columns.issubset(frame.columns)
    assert not frame[list(expected_columns)].isna().any().any()
    assert np.isfinite(frame["licensed_discovery_utility"]).all()
    assert frame["raw_claim_level"].between(0, 6).all()
    assert frame["licensed_claim_level"].between(0, 6).all()
    assert frame["final_claim_level"].between(0, 6).all()
    assert (frame["overclaim_gap"] >= 0.0).all()
    assert frame["claim_calibration_accuracy"].between(0.0, 1.0).all()
    assert frame["evidence_bottleneck_index"].between(0.0, 1.0).all()
    assert frame["goal_clarity"].between(0.0, 1.0).all()
    assert frame["route_domain_applicability"].between(0.0, 1.0).all()
    assert set(frame["licensed_ordering_available"].unique()).issubset({0.0, 1.0})
    assert frame["evaluator_independence"].between(0.0, 1.0).all()
    assert frame["evaluator_reliability"].between(0.0, 1.0).all()
    assert frame["evaluator_grounding"].between(0.0, 1.0).all()
    assert (
        frame["raw_claim_level"]
        == frame["raw_claim_strength"].map(claim_level_from_score)
    ).all()
    assert (
        frame["licensed_claim_level"]
        == frame["evidence_licensed_claim"].map(claim_level_from_score)
    ).all()
    assert (
        frame["final_claim_level"]
        == frame["final_claim_strength"].map(claim_level_from_score)
    ).all()


def test_summarize_routes_preserves_route_rows():
    config = SimulationConfig(start_year=2026, end_year=2028, monte_carlo_runs=16)
    frame = run_simulation(
        routes=default_routes(),
        domains=default_domains(),
        config=config,
        scenarios=["base_model_scaling"],
    )
    summary = summarize_routes(frame)

    assert set(summary["route"]) == {route.name for route in default_routes()}
    assert "mean_licensed_discovery_utility" in summary.columns
    assert "licensed_ordering_share" in summary.columns


def test_route_utility_timeseries_reports_standard_error():
    config = SimulationConfig(start_year=2026, end_year=2028, monte_carlo_runs=16)
    frame = run_simulation(
        routes=default_routes(),
        domains=default_domains(),
        config=config,
        scenarios=["base_model_scaling"],
    )
    frame.insert(1, "simulation_step", frame["year"] - config.start_year + 1)

    timeseries = route_utility_timeseries(frame)

    assert "mean_licensed_discovery_utility" in timeseries.columns
    assert "sem_licensed_discovery_utility" in timeseries.columns
    assert timeseries["sem_licensed_discovery_utility"].ge(0.0).all()
    assert not timeseries["sem_licensed_discovery_utility"].isna().any()
