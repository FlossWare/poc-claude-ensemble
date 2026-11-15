# Learning scoring and confidence semantics

Issue #109 is the specification gate for learning score formulas. This document defines
the meaning of each quantity before any formula or optimizer is restored.

## Scope

The score is a **model/task capability score**. It describes the relative capability
estimate for a model on a task under the evidence available to the learning system.

The score is not:

- a probability that a future answer will be correct;
- a sample-size measure;
- a model-agreement measure;
- an empirical success rate;
- a recommendation-confidence value;
- a cost score unless a later specification explicitly adds cost as a component.

The score is normalized to [0, 1] when a concrete implementation is introduced.

## Score components

The initial contract has three components:

| Component | Meaning | Positive adjustment | Negative adjustment |
|---|---|---|---|
| base_score | Baseline capability estimate for a model/task pair | Not an adjustment | Not an adjustment |
| complexity_adjustment | Evidence-based effect of task/file complexity on this model | Evidence that the model performs better as complexity increases | Evidence that the model performs worse as complexity increases |
| domain_adjustment | Evidence-based effect of relevant domain knowledge | Evidence that the model performs better for the domain | Evidence that the model performs worse for the domain |

The intended composition is conceptually:

score = clamp(base_score + complexity_adjustment + domain_adjustment, 0, 1)

This is a semantic contract, not an implementation requirement. A later implementation
must preserve these meanings even if the representation changes.

### Sign rules

The sign belongs to the **effect on the model**, not to the feature itself.

- More complexity is not inherently a bonus or penalty.
- Domain markers are not inherently a bonus or penalty.
- A model that handles increased complexity better receives a positive complexity adjustment.
- A model that handles increased complexity worse receives a negative complexity adjustment.
- A model with demonstrated domain-specific advantage receives a positive domain adjustment.
- A model with demonstrated domain-specific disadvantage receives a negative domain adjustment.
- A neutral effect is zero.
- Reversing either sign is a correctness defect.

The abandoned learning/scoring_function.py used negative values for Opus/Gemini
complexity adjustments while naming them bonuses. That implementation is not a source of truth and must not be restored wholesale.

## Evidence rules

A score component must describe evidence that the system actually has.

Hard-coded model rankings or invented benchmark values are not empirical evidence.
A future implementation may use configured priors, but those values must be identified as
priors/configuration rather than presented as observed performance.

The following quantities are evidence or metadata and must remain separate from the score:

### Sample count

sample_count is the number of eligible observations contributing to an empirical
measurement.

It answers: **How much data do we have?**

It does not answer how correct the model is.

### Model agreement

agreement_rate is the proportion of eligible comparable model judgments that agree
under an explicitly defined agreement rule.

It answers: **How consistently did the compared judgments agree?**

Agreement does not establish correctness. Several models can agree on the same wrong
answer.

### Empirical success rate

empirical_success_rate is the proportion of eligible observed outcomes classified as
successful by the learning system's explicit outcome rule.

It answers: **How often did observed outcomes meet the defined success criterion?**

It is an observed rate, not a probability claim about an unobserved future outcome.

### Recommendation confidence

recommendation_confidence describes how strongly the available evidence supports
using the recommendation produced by the decision process.

It answers: **How strong is the evidence supporting this recommendation under the
system's decision rule?**

It is a decision-support measure, not model correctness, sample size, agreement, or
empirical success rate. Its future formula must be specified separately before use.

## Confidence naming contract

No generic confidence field should be introduced when one of the specific meanings
above is intended. APIs should use the explicit field name:

- sample_count
- agreement_rate
- empirical_success_rate
- recommendation_confidence

If an API exposes a field literally named confidence, its documentation must define
which single semantic quantity it represents. It must not silently combine the four.

## No unsupported probabilistic terminology

The recovery implementation must not call a value a Bayesian posterior, credible
interval, posterior probability, or similar probabilistic quantity unless the
implementation actually defines and computes the required probabilistic model.

A normal approximation, heuristic range, or sample-size adjustment is not automatically
a Bayesian interval.

Until such a model is deliberately specified, use descriptive terms such as
"observed rate", "sample count", "agreement rate", and "recommendation confidence".

## Required regression cases for formula implementation

When the concrete scoring implementation is added, its regression suite must include
at least these semantic cases:

1. Increasing complexity for a model with a positive complexity effect increases its
   score relative to the same baseline.
2. Increasing complexity for a model with a negative complexity effect decreases its
   score relative to the same baseline.
3. A positive domain effect increases the score.
4. A negative domain effect decreases the score.
5. Zero adjustments leave the baseline unchanged.
6. Clamping cannot produce values outside [0, 1].
7. Changing sample_count alone must not silently change the model/task score.
8. Changing agreement_rate alone must not silently change the model/task score.
9. Changing empirical_success_rate alone must not silently change the model/task score.
10. Recommendation confidence must remain a separately named decision-support value.

These are contract tests for the eventual implementation, not permission to restore the
abandoned scoring function.

## Deferred work

This issue deliberately does not choose:

- numerical weights for the components;
- a particular statistical estimator;
- a Bayesian model;
- a confidence interval formula;
- a recommendation policy;
- an optimizer or genetic algorithm;
- a cost/latency component.

Those choices require evidence and their own implementation/test review.
