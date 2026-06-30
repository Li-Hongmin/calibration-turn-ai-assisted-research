# Route Simulation Result Notes

This is a synthetic sensitivity model for the manuscript's calibration semantics framework. It is not an empirical forecast, and the horizontal axis in the figures should be read as stylized simulation steps rather than calendar time.

## First-order readout

- Under this illustrative parameterization, the highest mean synthetic licensed-utility diagnostic in base model scaling is assigned to `Hybrid-Cal loop`.
- Under this illustrative parameterization, the highest mean synthetic licensed-utility diagnostic in calibration governance is assigned to `Hybrid-Cal loop`.
- Under this illustrative parameterization, the highest mean synthetic licensed-utility diagnostic in the Goodhart benchmark condition is assigned to `Hybrid-Cal loop`.
- In the evaluator-rich mathematics and algorithms domain under base model scaling, the highest conditional diagnostic score is assigned to `Algorithmic and mathematical discovery agent`.
- In the calibration ablation, mean final-over-licensed claim-level gap is `0.514` with no calibration, `0.133` under the imperfect/base condition, and `0.000` with perfect calibration.
- The route-utility curves report means with propagated Monte Carlo standard-error bands over 128 draws; the bands are diagnostic uncertainty summaries, not empirical confidence intervals.
- The overclaim scatter uses the terminal simulation step `10` and compares claim-level overstatement against evaluator strength; it is a conditional diagnostic, not a time forecast.
- The score landscape reports all route-domain-scenario scores, not a single highlighted route. Low goal-clarity rows still show synthetic utility values, but they are hatched because the objective or scoring rule is not yet clear enough to license a route ordering.
- Route ordering is also conditional on route-domain applicability. A strict evaluator does not make a route generally applicable outside domains where its operators are compatible with the task.

## Interpretation

The simulation operationalizes a central manuscript claim: as foundation-model capability improves, the decisive bottleneck shifts toward evaluator quality, calibration quality, cost, update loops, and route-domain applicability. Routes with strong external evaluators do well in the synthetic diagnostic when claims can be checked cheaply and when the route is compatible with the target domain. Routes with high throughput but weaker calibration can generate higher apparent productivity while carrying larger overclaim burdens. These readouts are diagnostic consequences of the chosen parameterization, not evidence of real-world route superiority.

The score landscape is a full route-domain-scenario matrix. Each panel corresponds to one scenario and calibration mode; within a panel, every route receives a continuous synthetic licensed-utility diagnostic score for every domain. Hatched low-clarity rows do not mean that no numerical score exists. They mean that the score should be read as an exploratory diagnostic rather than as a licensed ordering, because the target is still an open-ended problem-formulation task. In that regime, the scientific work moves upstream to defining the objective, building the evaluator, and specifying what would count as a licensed claim.

## Limits

All parameter values are stylized and synthetic. The model is useful for generating figures, stress-testing the formal vocabulary, and making assumptions explicit. It should not be used as a quantitative projection of scientific output without empirical calibration.
