from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
from hashlib import blake2b
from typing import Iterable

import numpy as np
import pandas as pd

CLAIM_LEVEL_THRESHOLDS: tuple[float, ...] = (
    0.10,
    0.25,
    0.40,
    0.55,
    0.70,
    0.85,
    0.95,
)
CALIBRATION_MODES: frozenset[str] = frozenset({"imperfect", "none", "perfect"})
LICENSED_ORDERING_GOAL_CLARITY_THRESHOLD = 0.45


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def evaluator_strength(
    independence: float,
    reliability: float,
    grounding: float,
) -> float:
    independence = clamp(independence)
    reliability = clamp(reliability)
    grounding = clamp(grounding)
    return clamp((independence * reliability * grounding) ** (1.0 / 3.0))


def claim_level_from_score(score: float) -> int:
    bounded_score = clamp(score)
    if bounded_score < CLAIM_LEVEL_THRESHOLDS[0]:
        return 0

    level = 0
    for candidate, threshold in enumerate(CLAIM_LEVEL_THRESHOLDS[1:], start=1):
        if bounded_score >= threshold:
            level = candidate
    return level


@dataclass(frozen=True)
class SimulationConfig:
    start_year: int = 2026
    end_year: int = 2035
    foundation_start: float = 0.46
    foundation_ceiling: float = 0.94
    foundation_growth: float = 0.24
    monte_carlo_runs: int = 128
    random_seed: int = 4317
    false_discovery_penalty: float = 0.62
    overclaim_penalty: float = 0.38

    @property
    def years(self) -> range:
        return range(self.start_year, self.end_year + 1)


@dataclass(frozen=True)
class RouteParameters:
    name: str
    label: str
    generation: float
    prediction: float
    evaluator: float
    evaluator_independence: float
    evaluator_reliability: float
    evaluator_grounding: float
    update: float
    calibration: float
    scope: float
    throughput: float
    cost: float
    noise: float
    overclaim_bias: float
    scaling_sensitivity: float
    evaluator_sensitivity: float
    cost_decline: float
    lab_dependency: float

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class DomainParameters:
    name: str
    label: str
    goal_clarity: float
    route_applicability: dict[str, float]
    evaluator_availability: float
    lab_cost_multiplier: float
    noise_multiplier: float
    claim_value: float

    def to_dict(self) -> dict[str, float | str | dict[str, float]]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioParameters:
    name: str
    label: str
    seed_key: str | None = None
    foundation_shift: float = 0.0
    evaluator_shift: float = 0.0
    evaluator_multiplier: float = 1.0
    lab_cost_shift: float = 1.0
    calibration_shift: float = 0.0
    overclaim_shift: float = 0.0
    noise_shift: float = 1.0
    calibration_mode: str = "imperfect"
    evaluator_model_coupling: float = 0.0
    benchmark_goodhart: float = 0.0

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def default_routes() -> list[RouteParameters]:
    return [
        RouteParameters(
            name="specialized_foundation_model",
            label="Specialized foundation model",
            generation=0.52,
            prediction=0.86,
            evaluator=0.62,
            evaluator_independence=0.60,
            evaluator_reliability=0.65,
            evaluator_grounding=0.61,
            update=0.48,
            calibration=0.63,
            scope=0.52,
            throughput=0.76,
            cost=0.56,
            noise=0.28,
            overclaim_bias=0.11,
            scaling_sensitivity=0.52,
            evaluator_sensitivity=0.42,
            cost_decline=0.18,
            lab_dependency=0.42,
        ),
        RouteParameters(
            name="llm_research_assistant",
            label="Human-led LLM assistant",
            generation=0.70,
            prediction=0.50,
            evaluator=0.48,
            evaluator_independence=0.44,
            evaluator_reliability=0.50,
            evaluator_grounding=0.51,
            update=0.45,
            calibration=0.54,
            scope=0.86,
            throughput=0.66,
            cost=0.34,
            noise=0.42,
            overclaim_bias=0.18,
            scaling_sensitivity=0.74,
            evaluator_sensitivity=0.30,
            cost_decline=0.20,
            lab_dependency=0.20,
        ),
        RouteParameters(
            name="multi_agent_coscientist",
            label="Multi-agent co-scientist",
            generation=0.80,
            prediction=0.58,
            evaluator=0.54,
            evaluator_independence=0.48,
            evaluator_reliability=0.56,
            evaluator_grounding=0.58,
            update=0.62,
            calibration=0.57,
            scope=0.82,
            throughput=0.78,
            cost=0.48,
            noise=0.38,
            overclaim_bias=0.20,
            scaling_sensitivity=0.82,
            evaluator_sensitivity=0.44,
            cost_decline=0.26,
            lab_dependency=0.32,
        ),
        RouteParameters(
            name="ai_scientist_pipeline",
            label="End-to-end AI scientist pipeline",
            generation=0.82,
            prediction=0.56,
            evaluator=0.45,
            evaluator_independence=0.38,
            evaluator_reliability=0.48,
            evaluator_grounding=0.50,
            update=0.58,
            calibration=0.43,
            scope=0.76,
            throughput=0.92,
            cost=0.43,
            noise=0.46,
            overclaim_bias=0.28,
            scaling_sensitivity=0.88,
            evaluator_sensitivity=0.36,
            cost_decline=0.34,
            lab_dependency=0.22,
        ),
        RouteParameters(
            name="algorithmic_math_agent",
            label="Algorithmic and mathematical discovery agent",
            generation=0.72,
            prediction=0.62,
            evaluator=0.88,
            evaluator_independence=0.88,
            evaluator_reliability=0.90,
            evaluator_grounding=0.86,
            update=0.78,
            calibration=0.76,
            scope=0.60,
            throughput=0.70,
            cost=0.30,
            noise=0.16,
            overclaim_bias=0.07,
            scaling_sensitivity=0.70,
            evaluator_sensitivity=0.86,
            cost_decline=0.22,
            lab_dependency=0.02,
        ),
        RouteParameters(
            name="self_driving_laboratory",
            label="Self-driving laboratory",
            generation=0.66,
            prediction=0.64,
            evaluator=0.80,
            evaluator_independence=0.76,
            evaluator_reliability=0.82,
            evaluator_grounding=0.83,
            update=0.82,
            calibration=0.69,
            scope=0.42,
            throughput=0.52,
            cost=0.92,
            noise=0.32,
            overclaim_bias=0.10,
            scaling_sensitivity=0.50,
            evaluator_sensitivity=0.72,
            cost_decline=0.42,
            lab_dependency=0.94,
        ),
        RouteParameters(
            name="hybrid_calibrated_loop",
            label="Hybrid-Cal loop",
            generation=0.72,
            prediction=0.68,
            evaluator=0.76,
            evaluator_independence=0.72,
            evaluator_reliability=0.78,
            evaluator_grounding=0.77,
            update=0.74,
            calibration=0.82,
            scope=0.66,
            throughput=0.68,
            cost=0.58,
            noise=0.25,
            overclaim_bias=0.06,
            scaling_sensitivity=0.62,
            evaluator_sensitivity=0.70,
            cost_decline=0.30,
            lab_dependency=0.52,
        ),
    ]


def default_domains() -> list[DomainParameters]:
    return [
        DomainParameters(
            name="biology",
            label="Biology",
            goal_clarity=0.58,
            route_applicability={
                "specialized_foundation_model": 0.82,
                "llm_research_assistant": 0.78,
                "multi_agent_coscientist": 0.90,
                "ai_scientist_pipeline": 0.60,
                "algorithmic_math_agent": 0.22,
                "self_driving_laboratory": 0.84,
                "hybrid_calibrated_loop": 0.86,
            },
            evaluator_availability=0.50,
            lab_cost_multiplier=1.10,
            noise_multiplier=1.10,
            claim_value=1.20,
        ),
        DomainParameters(
            name="materials",
            label="Materials",
            goal_clarity=0.72,
            route_applicability={
                "specialized_foundation_model": 0.88,
                "llm_research_assistant": 0.60,
                "multi_agent_coscientist": 0.72,
                "ai_scientist_pipeline": 0.56,
                "algorithmic_math_agent": 0.34,
                "self_driving_laboratory": 0.92,
                "hybrid_calibrated_loop": 0.86,
            },
            evaluator_availability=0.58,
            lab_cost_multiplier=1.00,
            noise_multiplier=0.96,
            claim_value=1.16,
        ),
        DomainParameters(
            name="machine_learning",
            label="Machine learning",
            goal_clarity=0.82,
            route_applicability={
                "specialized_foundation_model": 0.62,
                "llm_research_assistant": 0.72,
                "multi_agent_coscientist": 0.66,
                "ai_scientist_pipeline": 0.86,
                "algorithmic_math_agent": 0.78,
                "self_driving_laboratory": 0.20,
                "hybrid_calibrated_loop": 0.82,
            },
            evaluator_availability=0.68,
            lab_cost_multiplier=0.58,
            noise_multiplier=0.74,
            claim_value=0.96,
        ),
        DomainParameters(
            name="mathematics_algorithms",
            label="Mathematics and algorithms",
            goal_clarity=0.95,
            route_applicability={
                "specialized_foundation_model": 0.42,
                "llm_research_assistant": 0.60,
                "multi_agent_coscientist": 0.55,
                "ai_scientist_pipeline": 0.70,
                "algorithmic_math_agent": 0.96,
                "self_driving_laboratory": 0.05,
                "hybrid_calibrated_loop": 0.82,
            },
            evaluator_availability=0.92,
            lab_cost_multiplier=0.28,
            noise_multiplier=0.34,
            claim_value=1.06,
        ),
        DomainParameters(
            name="chemistry",
            label="Chemistry",
            goal_clarity=0.66,
            route_applicability={
                "specialized_foundation_model": 0.74,
                "llm_research_assistant": 0.62,
                "multi_agent_coscientist": 0.72,
                "ai_scientist_pipeline": 0.50,
                "algorithmic_math_agent": 0.25,
                "self_driving_laboratory": 0.94,
                "hybrid_calibrated_loop": 0.84,
            },
            evaluator_availability=0.56,
            lab_cost_multiplier=1.18,
            noise_multiplier=1.00,
            claim_value=1.22,
        ),
        DomainParameters(
            name="open_ended_problem_formulation",
            label="Open-ended problem formulation",
            goal_clarity=0.25,
            route_applicability={
                "specialized_foundation_model": 0.45,
                "llm_research_assistant": 0.72,
                "multi_agent_coscientist": 0.66,
                "ai_scientist_pipeline": 0.58,
                "algorithmic_math_agent": 0.35,
                "self_driving_laboratory": 0.30,
                "hybrid_calibrated_loop": 0.60,
            },
            evaluator_availability=0.26,
            lab_cost_multiplier=0.78,
            noise_multiplier=1.12,
            claim_value=0.78,
        ),
    ]


def default_scenarios() -> list[ScenarioParameters]:
    return [
        ScenarioParameters(
            name="base_model_scaling",
            label="Base model-only scaling",
        ),
        ScenarioParameters(
            name="base_model_plus_evaluator_scaling",
            label="Base model plus evaluator scaling",
            foundation_shift=0.06,
            evaluator_shift=0.04,
            evaluator_multiplier=1.08,
            evaluator_model_coupling=0.16,
            calibration_shift=0.03,
            noise_shift=0.96,
        ),
        ScenarioParameters(
            name="cheap_lab_future",
            label="Cheap laboratory future",
            evaluator_shift=0.06,
            evaluator_multiplier=1.08,
            lab_cost_shift=0.68,
            calibration_shift=0.03,
            noise_shift=0.94,
        ),
        ScenarioParameters(
            name="calibration_governance",
            label="Calibration governance",
            evaluator_shift=0.04,
            evaluator_multiplier=1.05,
            calibration_shift=0.14,
            overclaim_shift=-0.11,
            noise_shift=0.92,
            calibration_mode="imperfect",
        ),
        ScenarioParameters(
            name="no_calibration",
            label="No calibration",
            calibration_shift=-0.28,
            overclaim_shift=0.09,
            noise_shift=1.04,
            calibration_mode="none",
        ),
        ScenarioParameters(
            name="perfect_calibration",
            label="Perfect calibration",
            evaluator_shift=0.05,
            evaluator_multiplier=1.04,
            calibration_shift=0.22,
            overclaim_shift=-0.08,
            noise_shift=0.94,
            calibration_mode="perfect",
        ),
        ScenarioParameters(
            name="goodhart_benchmark_environment",
            label="Goodhart benchmark environment",
            foundation_shift=0.05,
            evaluator_shift=-0.05,
            evaluator_multiplier=0.90,
            calibration_shift=-0.06,
            overclaim_shift=0.12,
            noise_shift=1.08,
            benchmark_goodhart=0.18,
        ),
        ScenarioParameters(
            name="evaluator_bottleneck",
            label="Evaluator bottleneck",
            foundation_shift=0.06,
            evaluator_shift=-0.09,
            evaluator_multiplier=0.84,
            overclaim_shift=0.05,
            noise_shift=1.04,
        ),
    ]


def scenario_lookup() -> dict[str, ScenarioParameters]:
    return {scenario.name: scenario for scenario in default_scenarios()}


def foundation_level(year: int, config: SimulationConfig) -> float:
    elapsed = year - config.start_year
    level = config.foundation_ceiling - (
        config.foundation_ceiling - config.foundation_start
    ) * exp(-config.foundation_growth * elapsed)
    return clamp(level)


def _scenario_sequence(
    scenario_names: Iterable[str | ScenarioParameters] | None,
) -> list[ScenarioParameters]:
    scenarios = scenario_lookup()
    if scenario_names is None:
        return list(scenarios.values())

    scenario_list: list[ScenarioParameters] = []
    unknown: list[str] = []
    for scenario in scenario_names:
        if isinstance(scenario, ScenarioParameters):
            scenario_list.append(scenario)
        elif scenario in scenarios:
            scenario_list.append(scenarios[scenario])
        else:
            unknown.append(scenario)

    if unknown:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown)}")
    return scenario_list


def _perturb(base_value: float, rng: np.random.Generator, scale: float) -> float:
    if scale <= 0.0:
        return base_value
    return clamp(float(rng.normal(base_value, scale)))


def _scenario_seed(base_seed: int, scenario_name: str) -> int:
    digest = blake2b(
        f"{base_seed}:{scenario_name}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _validate_calibration_modes(scenarios: Iterable[ScenarioParameters]) -> None:
    unknown = sorted(
        {
            scenario.calibration_mode
            for scenario in scenarios
            if scenario.calibration_mode not in CALIBRATION_MODES
        }
    )
    if unknown:
        raise ValueError(f"Unknown calibration mode: {', '.join(unknown)}")


def route_domain_applicability(
    route: RouteParameters,
    domain: DomainParameters,
) -> float:
    if route.name not in domain.route_applicability:
        raise KeyError(
            "Missing route-domain applicability for "
            f"route {route.name!r} in domain {domain.name!r}"
        )
    return clamp(domain.route_applicability[route.name])


def route_domain_applicability_table(
    routes: Iterable[RouteParameters] | None = None,
    domains: Iterable[DomainParameters] | None = None,
) -> pd.DataFrame:
    route_list = list(routes or default_routes())
    domain_list = list(domains or default_domains())
    rows = []
    for domain in domain_list:
        for route in route_list:
            rows.append(
                {
                    "domain": domain.name,
                    "domain_label": domain.label,
                    "route": route.name,
                    "route_label": route.label,
                    "route_domain_applicability": route_domain_applicability(
                        route,
                        domain,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _calibrated_claim_strength(
    raw_claim_strength: float,
    licensed_claim_strength: float,
    calibration_quality: float,
    calibration_mode: str,
) -> float:
    if calibration_mode == "none":
        return raw_claim_strength
    if calibration_mode == "perfect":
        return min(raw_claim_strength, licensed_claim_strength)
    if calibration_mode not in CALIBRATION_MODES:
        raise ValueError(f"Unknown calibration mode: {calibration_mode}")

    gap = max(0.0, raw_claim_strength - licensed_claim_strength)
    residual_gap = gap * clamp(1.0 - 0.88 * calibration_quality, 0.06, 0.62)
    return clamp(min(raw_claim_strength, licensed_claim_strength) + residual_gap)


def _route_domain_year_metrics(
    route: RouteParameters,
    domain: DomainParameters,
    scenario: ScenarioParameters,
    year: int,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    model = clamp(foundation_level(year, config) + scenario.foundation_shift)
    time_progress = (year - config.start_year) / max(1, config.end_year - config.start_year)
    licensed_ordering_available = int(
        domain.goal_clarity >= LICENSED_ORDERING_GOAL_CLARITY_THRESHOLD
    )
    applicability = route_domain_applicability(route, domain)

    generation = clamp(
        route.generation
        + route.scaling_sensitivity * 0.24 * model
        + _perturb(0.0, rng, 0.018)
    )
    prediction = clamp(
        route.prediction
        + route.scaling_sensitivity * 0.18 * model
        + _perturb(0.0, rng, 0.014)
    )
    evaluator_model_gain = (
        scenario.evaluator_model_coupling * model * route.evaluator_sensitivity
    )
    evaluator_independence = clamp(
        (
            0.68 * route.evaluator_independence
            + 0.24 * domain.evaluator_availability
            + 0.08 * route.evaluator
            + evaluator_model_gain
        )
        * scenario.evaluator_multiplier
        + scenario.evaluator_shift
        - 0.40 * scenario.benchmark_goodhart
        + _perturb(0.0, rng, 0.016)
    )
    evaluator_reliability = clamp(
        (
            0.58 * route.evaluator_reliability
            + 0.30 * domain.evaluator_availability
            + 0.12 * route.evaluator
            + evaluator_model_gain
        )
        * scenario.evaluator_multiplier
        + scenario.evaluator_shift
        - 0.20 * scenario.benchmark_goodhart
        + _perturb(0.0, rng, 0.015)
    )
    evaluator_grounding = clamp(
        (
            0.50 * route.evaluator_grounding
            + 0.38 * domain.evaluator_availability
            + 0.12 * route.evaluator
            + evaluator_model_gain
        )
        * scenario.evaluator_multiplier
        + scenario.evaluator_shift
        - 0.12 * scenario.benchmark_goodhart
        + _perturb(0.0, rng, 0.015)
    )
    evaluator = evaluator_strength(
        evaluator_independence,
        evaluator_reliability,
        evaluator_grounding,
    )
    update = clamp(route.update + 0.12 * evaluator + 0.08 * model)
    calibration = clamp(
        route.calibration
        + 0.10 * evaluator
        + scenario.calibration_shift
        + _perturb(0.0, rng, 0.014)
    )
    noise = clamp(
        route.noise * domain.noise_multiplier * scenario.noise_shift
        + _perturb(0.0, rng, 0.018)
    )

    evidential_support = clamp(
        0.27 * generation
        + 0.27 * prediction
        + 0.13 * evaluator_independence
        + 0.10 * evaluator_reliability
        + 0.09 * evaluator_grounding
        + 0.17 * update
        + 0.06 * domain.goal_clarity
        - 0.16 * noise
    )
    benchmark_pressure = scenario.benchmark_goodhart * route.scaling_sensitivity
    automation_pressure = 0.08 * route.throughput * (1.0 - evaluator_independence)
    raw_claim_strength = clamp(
        evidential_support
        + route.overclaim_bias
        + scenario.overclaim_shift
        + automation_pressure
        + benchmark_pressure
        + _perturb(0.0, rng, 0.012)
    )
    license_ceiling = clamp(
        evidential_support
        + 0.13 * evaluator_independence
        + 0.09 * evaluator_reliability
        + 0.14 * evaluator_grounding
        + 0.05 * domain.goal_clarity
        - 0.13 * noise
    )
    evidence_licensed_claim = min(raw_claim_strength, license_ceiling)
    final_claim_strength = _calibrated_claim_strength(
        raw_claim_strength=raw_claim_strength,
        licensed_claim_strength=evidence_licensed_claim,
        calibration_quality=calibration,
        calibration_mode=scenario.calibration_mode,
    )
    claim_gap = max(0.0, raw_claim_strength - evidence_licensed_claim)
    final_claim_gap = max(0.0, final_claim_strength - evidence_licensed_claim)
    raw_claim_level = claim_level_from_score(raw_claim_strength)
    licensed_claim_level = claim_level_from_score(evidence_licensed_claim)
    final_claim_level = claim_level_from_score(final_claim_strength)
    overclaim_gap = max(0, final_claim_level - licensed_claim_level)
    claim_calibration_accuracy = (
        1.0 if claim_gap == 0.0 else clamp(1.0 - final_claim_gap / claim_gap)
    )
    evidence_bottleneck_index = clamp(1.0 - evaluator)
    overclaim_rate = clamp(
        (final_claim_strength - evidential_support)
        * (1.0 - 0.72 * claim_calibration_accuracy)
    )
    truth_probability = clamp(
        sigmoid(4.0 * (evidential_support - 0.50))
        * (0.56 + 0.44 * evaluator)
        * (1.0 - 0.28 * noise)
    )

    cost = max(
        0.05,
        route.cost
        * (1.0 - route.cost_decline * time_progress)
        * (1.0 + route.lab_dependency * (domain.lab_cost_multiplier - 1.0))
        * (1.0 - route.lab_dependency * (1.0 - scenario.lab_cost_shift)),
    )
    license_rate = clamp(
        (
            0.34
        + 0.33 * evaluator
        + 0.23 * calibration
        - 0.35 * overclaim_rate
        - 0.20 * final_claim_gap
        )
        * (0.55 + 0.45 * domain.goal_clarity)
    )
    effective_throughput = route.throughput * (0.74 + 0.28 * model)
    true_licensed_claims = (
        effective_throughput
        * route.scope
        * truth_probability
        * license_rate
    )
    false_discovery_burden = (
        effective_throughput
        * (0.25 + route.scope)
        * overclaim_rate
        * (1.0 - evaluator)
        * (0.72 + 0.28 * noise)
    )
    raw_licensed_discovery_utility = (
        domain.claim_value * domain.goal_clarity * true_licensed_claims
        - config.false_discovery_penalty * false_discovery_burden
        - config.overclaim_penalty * overclaim_rate
    )
    licensed_discovery_utility = applicability * raw_licensed_discovery_utility
    cost_normalized_utility = licensed_discovery_utility / cost

    return {
        "year": year,
        "scenario": scenario.name,
        "scenario_label": scenario.label,
        "calibration_mode": scenario.calibration_mode,
        "domain": domain.name,
        "domain_label": domain.label,
        "goal_clarity": domain.goal_clarity,
        "route_domain_applicability": applicability,
        "licensed_ordering_available": licensed_ordering_available,
        "route": route.name,
        "route_label": route.label,
        "foundation_level": model,
        "generation_strength": generation,
        "prediction_strength": prediction,
        "evaluator_independence": evaluator_independence,
        "evaluator_reliability": evaluator_reliability,
        "evaluator_grounding": evaluator_grounding,
        "evaluator_strength": evaluator,
        "belief_update_strength": update,
        "calibration_quality": calibration,
        "evidential_support": evidential_support,
        "raw_claim_strength": raw_claim_strength,
        "evidence_licensed_claim": evidence_licensed_claim,
        "final_claim_strength": final_claim_strength,
        "raw_claim_level": raw_claim_level,
        "licensed_claim_level": licensed_claim_level,
        "final_claim_level": final_claim_level,
        "claim_gap": claim_gap,
        "overclaim_gap": overclaim_gap,
        "claim_calibration_accuracy": claim_calibration_accuracy,
        "evidence_bottleneck_index": evidence_bottleneck_index,
        "truth_probability": truth_probability,
        "license_rate": license_rate,
        "true_licensed_claims": true_licensed_claims,
        "overclaim_rate": overclaim_rate,
        "false_discovery_burden": false_discovery_burden,
        "raw_licensed_discovery_utility": raw_licensed_discovery_utility,
        "licensed_discovery_utility": licensed_discovery_utility,
        "cost": cost,
        "cost_normalized_utility": cost_normalized_utility,
    }


def run_simulation(
    routes: Iterable[RouteParameters] | None = None,
    domains: Iterable[DomainParameters] | None = None,
    config: SimulationConfig | None = None,
    scenarios: Iterable[str | ScenarioParameters] | None = None,
) -> pd.DataFrame:
    config = config or SimulationConfig()
    scenario_list = _scenario_sequence(scenarios)
    _validate_calibration_modes(scenario_list)
    route_list = list(routes or default_routes())
    domain_list = list(domains or default_domains())
    rows: list[dict[str, float | int | str]] = []

    for scenario in scenario_list:
        rng = np.random.default_rng(
            _scenario_seed(config.random_seed, scenario.seed_key or scenario.name)
        )
        for year in config.years:
            for domain in domain_list:
                for route in route_list:
                    samples = [
                        _route_domain_year_metrics(
                            route=route,
                            domain=domain,
                            scenario=scenario,
                            year=year,
                            config=config,
                            rng=rng,
                        )
                        for _ in range(config.monte_carlo_runs)
                    ]
                    row = samples[0].copy()
                    numeric_keys = [
                        key
                        for key, value in row.items()
                        if isinstance(value, (float, int)) and key != "year"
                    ]
                    for key in numeric_keys:
                        values = np.array([sample[key] for sample in samples], dtype=float)
                        row[key] = float(np.mean(values))
                        if key == "licensed_discovery_utility":
                            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                            row["mc_std_licensed_discovery_utility"] = std
                            row["mc_sem_licensed_discovery_utility"] = (
                                std / float(np.sqrt(len(values))) if len(values) > 0 else 0.0
                            )
                    row["raw_claim_level"] = claim_level_from_score(
                        float(row["raw_claim_strength"])
                    )
                    row["licensed_claim_level"] = claim_level_from_score(
                        float(row["evidence_licensed_claim"])
                    )
                    row["final_claim_level"] = claim_level_from_score(
                        float(row["final_claim_strength"])
                    )
                    row["overclaim_gap"] = max(
                        0,
                        int(row["final_claim_level"])
                        - int(row["licensed_claim_level"]),
                    )
                    rows.append(row)

    return pd.DataFrame(rows)


def summarize_routes(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(
            ["scenario", "calibration_mode", "route", "route_label"],
            as_index=False,
        )
        .agg(
            mean_licensed_discovery_utility=(
                "licensed_discovery_utility",
                "mean",
            ),
            mean_cost_normalized_utility=("cost_normalized_utility", "mean"),
            mean_true_licensed_claims=("true_licensed_claims", "mean"),
            mean_overclaim_rate=("overclaim_rate", "mean"),
            mean_evaluator_strength=("evaluator_strength", "mean"),
            mean_calibration_quality=("calibration_quality", "mean"),
            mean_goal_clarity=("goal_clarity", "mean"),
            mean_route_domain_applicability=(
                "route_domain_applicability",
                "mean",
            ),
            licensed_ordering_share=("licensed_ordering_available", "mean"),
        )
        .sort_values(
            ["scenario", "calibration_mode", "mean_licensed_discovery_utility"],
            ascending=[True, True, False],
        )
    )
    grouped["rank"] = grouped.groupby(["scenario", "calibration_mode"])[
        "mean_licensed_discovery_utility"
    ].rank(method="first", ascending=False)
    grouped["rank"] = grouped["rank"].astype(int)
    return grouped


def rank_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = (
        frame.groupby(
            [
                "scenario",
                "calibration_mode",
                "domain",
                "domain_label",
                "route",
                "route_label",
            ],
            as_index=False,
        )
        .agg(
            mean_licensed_discovery_utility=(
                "licensed_discovery_utility",
                "mean",
            ),
            mean_cost_normalized_utility=("cost_normalized_utility", "mean"),
            mean_overclaim_rate=("overclaim_rate", "mean"),
            mean_evaluator_strength=("evaluator_strength", "mean"),
            goal_clarity=("goal_clarity", "mean"),
            route_domain_applicability=("route_domain_applicability", "mean"),
            licensed_ordering_available=("licensed_ordering_available", "min"),
        )
        .sort_values(
            [
                "scenario",
                "calibration_mode",
                "domain",
                "mean_licensed_discovery_utility",
            ],
            ascending=[True, True, True, False],
        )
    )
    ranked["rank"] = ranked.groupby(["scenario", "calibration_mode", "domain"])[
        "mean_licensed_discovery_utility"
    ].rank(method="first", ascending=False)
    ranked["rank"] = ranked["rank"].astype("Int64")
    ranked.loc[ranked["licensed_ordering_available"] == 0, "rank"] = pd.NA
    return ranked


def route_utility_timeseries(
    frame: pd.DataFrame,
    scenario: str = "base_model_scaling",
    calibration_mode: str = "imperfect",
) -> pd.DataFrame:
    filtered = frame[
        (frame["scenario"] == scenario)
        & (frame["calibration_mode"] == calibration_mode)
    ]
    rows = []
    for (step, route, route_label), group in filtered.groupby(
        ["simulation_step", "route", "route_label"],
        sort=False,
    ):
        mean_utility = float(group["licensed_discovery_utility"].mean())
        if "mc_sem_licensed_discovery_utility" in group:
            sem_values = group["mc_sem_licensed_discovery_utility"].to_numpy(dtype=float)
            sem_utility = float(np.sqrt(np.sum(np.square(sem_values))) / len(sem_values))
        else:
            sem_utility = (
                float(group["licensed_discovery_utility"].sem())
                if len(group) > 1
                else 0.0
            )
        rows.append(
            {
                "simulation_step": step,
                "route": route,
                "route_label": route_label,
                "mean_licensed_discovery_utility": mean_utility,
                "sem_licensed_discovery_utility": sem_utility,
            }
        )
    return pd.DataFrame(rows).sort_values(["route", "simulation_step"])


def simulation_config_payload(config: SimulationConfig) -> dict[str, object]:
    return {
        "config": asdict(config),
        "routes": [route.to_dict() for route in default_routes()],
        "domains": [domain.to_dict() for domain in default_domains()],
        "scenarios": [scenario.to_dict() for scenario in default_scenarios()],
    }
