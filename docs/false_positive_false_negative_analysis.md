# False Positive / False Negative Analysis — Project 4 ML Pipeline & Anomaly Detection

## Purpose

This document explains how Project 4 handles false positives, false negatives, anomaly thresholds, and proxy metrics.

An anomaly detection platform is not production-ready just because it can return `is_anomaly=true`.

A production anomaly system must answer harder questions:

* What happens when the model is wrong?
* What counts as a false positive?
* What counts as a false negative?
* What threshold is being used?
* Why was that threshold chosen?
* What happens if the threshold is too aggressive?
* What happens if the threshold is too weak?
* Which metrics are real?
* Which metrics are proxy metrics?
* What evidence would be needed to calculate real precision, recall, false-positive rate, and false-negative rate?

Project 4 currently uses an unsupervised Isolation Forest model. The model is trained from real-source Project 1 and Project 2/3 feature extracts, but it does not currently have human-reviewed anomaly labels or delayed ground-truth labels.

That means Project 4 can discuss false-positive and false-negative risk honestly, but it cannot truthfully claim real false-positive rate, false-negative rate, precision, recall, or F1.

This document keeps that line clean.

---

## Current model context

Current production model state:

| Field                 | Value                            |
| --------------------- | -------------------------------- |
| Model name            | `isolation_forest`               |
| Active version        | `v002`                           |
| Snapshot type         | `real_source_extract`            |
| Training row count    | `16,750`                         |
| Feature count         | `51`                             |
| Baseline anomaly rate | `0.05002985074626866`            |
| Label availability    | `unlabeled_proxy_metrics`        |
| Metric type           | `unsupervised_training_baseline` |

The active model was trained on a real-source snapshot, but the anomaly labels are not real labels.

The current baseline tells us that the model flagged approximately 5.0% of the training snapshot as anomalous under the active scoring behavior.

It does not tell us that 5.0% of records were truly bad.

That distinction matters.

---

## Definitions

### False positive

A false positive is a normal record incorrectly flagged as anomalous.

In Project 4, a false positive could be:

* a legitimate high-value customer flagged because their cart value is unusually large
* a planned campaign spike flagged as suspicious
* a normal seasonal conversion-rate change flagged as abnormal
* a temporary latency increase that does not represent a real incident
* a real business promotion misread as behavioral drift
* an engaged user session incorrectly treated as fraud-like behavior
* a warehouse metric shift caused by valid upstream data refresh timing

False positives are loud errors.

They create work.

They create noise.

They can make operators stop trusting the system.

### False negative

A false negative is a real anomaly that the model fails to flag.

In Project 4, a false negative could be:

* a fraud-like behavior pattern hidden inside otherwise normal-looking aggregate features
* a campaign performance collapse that does not cross the anomaly threshold
* a source-data problem that passes schema validation but changes business meaning
* a latency degradation that hurts users but stays below the model boundary
* an abnormal cart abandonment pattern that blends into historical noise
* a model drift issue that appears slowly and does not trigger a sharp anomaly score
* a bad upstream feature distribution that produces stable-looking predictions

False negatives are quiet errors.

They do not create noise.

They create silence.

And silence can be expensive.

---

## Why false positives and false negatives matter

Anomaly detection is not a pure accuracy problem.

It is an operational tradeoff.

A model can be mathematically clean and still operationally useless if it creates too many false positives. A model can be calm and stable and still dangerous if it misses real anomalies.

The right balance depends on the use case.

For fraud, abuse, security, and critical reliability monitoring, the system may tolerate more false positives to reduce missed anomalies.

For executive dashboards, campaign monitoring, and business reporting, too many false positives can make the system useless because teams stop trusting the alerts.

Project 4 sits between both worlds:

* it monitors behavioral and business anomalies
* it monitors model and data drift
* it emits alert events
* it supports rollback
* it exposes Prometheus and Grafana observability
* it tracks latency and prediction evidence

So false-positive and false-negative behavior affects more than a single prediction. It affects the whole operating loop.


---

## Current label position

Project 4 currently has no real anomaly labels.

Current status:

| Metric / evidence type        | Current status                                |
| ----------------------------- | --------------------------------------------- |
| Real labels                   | Not implemented                               |
| Human-reviewed anomaly labels | Not available                                 |
| Delayed truth labels          | Not available                                 |
| Injected replay labels        | Not implemented                               |
| Precision                     | Not claimed                                   |
| Recall                        | Not claimed                                   |
| F1 score                      | Not claimed                                   |
| False-positive rate           | Not claimed                                   |
| False-negative rate           | Not claimed                                   |
| Baseline anomaly rate         | Implemented                                   |
| Feature baseline statistics   | Implemented                                   |
| Current anomaly rate          | Available through prediction logs and metrics |
| Drift checks                  | Implemented                                   |
| Alert events                  | Implemented                                   |
| Rollback evidence             | Implemented                                   |
| Latency budget validation     | Implemented                                   |

The current label mode is:

`unlabeled_proxy_metrics`

This means the system can track operational symptoms, but it cannot yet prove classification correctness.

That is not a weakness if documented honestly.

It becomes a weakness only if the project pretends proxy metrics are real labels.

---

## Real metrics versus proxy metrics

### Real metrics

Real false-positive and false-negative metrics require trusted labels.

Valid real-label sources would include:

* human-reviewed anomaly investigations
* confirmed fraud or non-fraud outcomes
* confirmed incident or non-incident outcomes
* delayed business truth data
* manually labeled replay datasets
* known injected anomalies in controlled replay data
* confirmed historical incidents mapped back to feature rows

With real labels, Project 4 could calculate:

| Metric              | Meaning                                                          |
| ------------------- | ---------------------------------------------------------------- |
| Precision           | Of records flagged anomalous, how many were truly anomalous      |
| Recall              | Of truly anomalous records, how many the model caught            |
| F1                  | Balance between precision and recall                             |
| False-positive rate | Normal records incorrectly flagged                               |
| False-negative rate | Real anomalies missed                                            |
| Confusion matrix    | True positives, false positives, true negatives, false negatives |

### Proxy metrics

Proxy metrics are signals that help judge model behavior without labels.

Current proxy metrics include:

* baseline anomaly rate
* current anomaly rate
* anomaly-rate delta
* anomaly-rate ratio
* feature mean drift
* feature variance drift
* drift status
* alert event volume
* prediction error count
* prediction latency p50 and p95
* rollback events
* investigation notes, when manually added later
* delayed truth labels, when added later

Proxy metrics do not prove model correctness.

They indicate whether live behavior is materially different from approved baseline behavior.

That is still valuable.

It answers:

> Is the system behaving differently from the model we approved?

It does not answer:

> Was every prediction correct?


---

## Threshold logic

An anomaly threshold converts a continuous anomaly score into a binary decision.

The threshold controls how aggressive the model is.

In Project 4 helper logic, lower anomaly scores are treated as more anomalous when estimating a cutoff from score distribution.

Conceptually:

| Threshold behavior        | Effect                |
| ------------------------- | --------------------- |
| Less aggressive threshold | Fewer records flagged |
| More aggressive threshold | More records flagged  |

A threshold that is too aggressive increases false-positive risk.

A threshold that is too weak increases false-negative risk.

There is no free lunch here. You choose which pain you are more willing to carry.

---

## Current threshold implementation

Project 4 now includes threshold and error-tradeoff helper logic in:

`src/anomaly_detection/evaluation.py`

Implemented helper functions:

* `estimate_threshold_from_scores`
* `summarize_threshold_tradeoff`

These helpers support:

* threshold estimation from anomaly scores
* target anomaly-rate based threshold selection
* baseline anomaly rate comparison
* current anomaly rate comparison
* anomaly-rate delta calculation
* anomaly-rate ratio calculation
* explicit proxy metric reporting
* explicit real false-positive / false-negative availability flags
* threshold behavior explanation

The helper intentionally does not calculate fake precision, fake recall, fake F1, fake false-positive rate, or fake false-negative rate.

That is the correct choice.

---

## Why the current threshold position is reasonable

The current active model baseline anomaly rate is:

`0.05002985074626866`

That is approximately 5.0%.

For an unsupervised anomaly detector, anchoring initial threshold behavior to the training anomaly-rate baseline is reasonable because it gives the system a controlled starting point.

It prevents two bad extremes:

* flagging almost everything
* flagging almost nothing

A 5% baseline does not mean 5% of records are truly anomalous.

It means the model identified roughly 5% of the training records as outside the learned normal pattern under the active scoring setup.

The threshold is therefore an operational starting point, not a proven truth boundary.

---

## Threshold tuning strategy

Threshold tuning should be driven by evidence.

Recommended tuning loop:

1. Start from the approved training baseline anomaly rate.
2. Monitor current anomaly rate from batch and online prediction logs.
3. Compare current anomaly rate against baseline anomaly rate.
4. Track anomaly-rate delta and anomaly-rate ratio.
5. Review drift events by feature.
6. Review alert events by severity and type.
7. Inspect examples near the threshold boundary.
8. Collect operator feedback from investigations.
9. Add delayed labels when possible.
10. Tune threshold only when evidence shows the current boundary is too noisy or too quiet.
11. Record threshold changes with model version, timestamp, reason, and observed impact.

Threshold tuning should not be done casually.

Changing the threshold changes production behavior.

It can change alert volume, rollback pressure, dashboard interpretation, and team trust.

---

## If false positives are too high

Symptoms:

* anomaly rate is materially higher than baseline
* alerts spike without confirmed incidents
* operators repeatedly mark alerts as normal
* dashboards look noisy
* rollback is considered too often
* high-value normal behavior is flagged
* planned campaigns or seasonal behavior trigger alerts
* alert volume grows faster than investigation capacity

Business impact:

* wasted investigation time
* alert fatigue
* reduced trust in the model
* ignored dashboards
* slower response to real incidents
* unnecessary rollback discussions
* poor stakeholder confidence

Recommended response:

* make the threshold less aggressive
* add suppression rules for known benign patterns
* segment thresholds by entity type or business domain
* separate campaign anomalies from user-behavior anomalies
* add features that distinguish planned spikes from unexpected spikes
* require drift confirmation before raising critical alerts
* review records near the threshold boundary
* collect human review outcomes

False positives are smoke without fire.

Too much smoke and nobody runs when the building burns.

---

## If false negatives are too high

Symptoms:

* incidents occur without anomaly flags
* business metrics degrade while anomaly rate stays flat
* drift appears after impact is already visible
* manual review finds missed abnormal records
* delayed labels show missed cases near the threshold boundary
* source-data quality issues are discovered outside the ML monitoring path
* latency or reliability incidents are caught manually before the anomaly system reacts

Business impact:

* delayed incident response
* missed fraud or abuse signals
* missed campaign performance collapse
* missed customer behavior shifts
* missed data quality degradation
* delayed rollback
* silent model degradation
* business loss before detection

Recommended response:

* make the threshold more aggressive
* add domain-specific high-risk rules
* add richer behavioral and temporal features
* retrain with a more recent snapshot
* segment the model by entity type
* add replay tests with known bad patterns
* add delayed-label backtesting
* review false-negative candidates near the model boundary

False negatives are fire without smoke.

The room burns quietly.


---

## Alert fatigue

Alert fatigue happens when the system produces too many alerts that do not lead to useful action.

This is a production failure mode.

It does not matter if the model is technically working. If operators stop trusting alerts, the monitoring layer has failed.

In Project 4, alert fatigue can come from:

* threshold too aggressive
* drift threshold too sensitive
* duplicate alerts
* repeated benign campaign spikes
* latency noise treated as critical incidents
* anomaly-rate spikes without business context
* lack of alert grouping
* no suppression window
* no investigation feedback loop

Impact:

* operators ignore alerts
* real incidents are missed
* dashboards lose credibility
* rollback signals become noisy
* executives lose confidence
* engineering time gets burned on false alarms

Current mitigation in Project 4:

* alert events are explicit and inspectable
* drift severity is separated into normal, warning, and critical
* Prometheus and Grafana make alert volume visible
* rollback is not blind automation
* rollback requires evidence rather than a single noisy signal

Planned hardening:

* alert deduplication
* suppression windows
* severity-based routing
* manual investigation outcome tracking
* delayed truth labels
* alert quality review

---

## Missed anomaly risk

Missed anomaly risk is the cost of false negatives.

In an anomaly system, missed anomalies are dangerous because they do not create obvious noise. They hide inside normal operations until another system, user, operator, or business metric reveals the issue.

In Project 4, missed anomalies could affect:

* customer behavior monitoring
* campaign performance monitoring
* fraud and risk proxy monitoring
* latency and service reliability monitoring
* source-data quality monitoring
* model drift monitoring
* rollback timing

Current mitigation in Project 4:

* baseline-vs-current anomaly-rate comparison
* mean and variance drift checks
* prediction evidence logging
* Prometheus metrics
* Grafana dashboards
* alert events
* rollback controls
* latency budget validation

Planned hardening:

* delayed labels
* replay datasets with known incidents
* human review queue
* threshold backtesting
* business-rule overlays for critical conditions
* per-domain thresholds
* incident-linked model evaluation

---

## Interaction with drift monitoring

False positives and false negatives should not be judged only from prediction scores.

They should be interpreted alongside drift.

Example:

> If current anomaly rate rises, feature drift is critical, and alert events increase, there is stronger evidence of real degradation.

Another example:

> If current anomaly rate rises, there is no feature drift, no incident evidence, and alerts are repeatedly marked benign, the threshold may be false-positive heavy.

Another example:

> If current anomaly rate stays flat while feature drift is critical and business metrics degrade, the system may have false-negative risk.

Drift does not prove labels.

But drift gives context.

Without context, anomaly scores are just numbers wearing a warning sign.

---

## Interaction with alert events

Alert events are downstream of model behavior.

False positives can inflate alert volume.

False negatives can prevent alert events from firing.

Project 4 alert types include:

* drift-critical alerts
* anomaly-rate spike alerts
* latency budget breach alerts
* prediction error rate alerts
* manual alert events

When alert volume rises, the system should ask:

> Is this real degradation, or are we over-alerting?

When alert volume stays quiet during known business or system degradation, the system should ask:

> Are we missing anomalies?

This is why prediction evidence, drift events, alert events, and rollback events must be analyzed together.

---

## Interaction with rollback

Rollback should not be triggered by one noisy prediction.

Rollback should consider multiple evidence streams:

* active model version
* previous stable model version
* current anomaly rate
* baseline anomaly rate
* anomaly-rate ratio
* drift status
* alert event severity
* prediction error rate
* latency budget status
* manual operator reason
* registry status

False positives can cause unnecessary rollback if the alert layer is too sensitive.

False negatives can delay rollback if the anomaly system misses degradation.

That is why Project 4 treats rollback as a controlled operation, not blind automation.

Good ML systems do not just deploy.

They recover carefully.

---

## Interaction with latency budget

Threshold logic also interacts with latency.

If the system adds heavy investigation logic directly inside the online request path, latency can degrade.

Project 4 should keep online prediction lightweight:

* validate payload
* score with in-memory model
* return prediction
* persist evidence
* emit metrics

Heavier analysis should happen outside the hot path:

* batch review
* threshold tuning
* delayed-label analysis
* human investigation
* drift review
* rollback review

The online target remains:

`p95 < 200 ms`

The model should not become slow just because the evaluation process becomes mature.


---

## Batch versus online threshold behavior

Project 4 supports both batch and online inference.

The same model version can serve both paths, but threshold review may differ by path.

### Batch inference

Batch inference is better for:

* historical review
* backfills
* threshold sensitivity analysis
* replay tests
* delayed-label comparison
* aggregate anomaly-rate review
* business reporting

Batch threshold changes can be tested safely before changing online behavior.

### Online inference

Online inference is better for:

* immediate scoring
* low-latency anomaly response
* live service monitoring
* high-value event detection

Online threshold changes should be conservative because they immediately affect alerts, dashboards, and operational decisions.

Recommended approach:

1. Test threshold changes in batch first.
2. Review alert and anomaly volume.
3. Inspect examples near the threshold boundary.
4. Promote threshold behavior to online inference only if justified.

---

## Recommended future label strategy

Project 4 should eventually add delayed labels or review labels.

### Human review labels

Add an investigation workflow where sampled anomalies are reviewed and marked as:

* `confirmed_anomaly`
* `benign`
* `needs_more_context`

### Delayed business labels

Join predictions later to business outcomes such as:

* `fraud_confirmed`
* `campaign_failure_confirmed`
* `incident_confirmed`
* `refund_spike_confirmed`
* `conversion_collapse_confirmed`
* `latency_incident_confirmed`

### Replay labels

Create controlled replay datasets with known injected anomalies:

* `known_latency_spike`
* `known_fraud_pattern`
* `known_campaign_collapse`
* `known_schema_shift`
* `known_cart_abandonment_spike`

### Incident-linked labels

When real incidents happen, link them back to prediction windows:

* `incident_id`
* `incident_start_time`
* `incident_end_time`
* `affected_entities`
* `expected_detection_window`
* `was_detected`
* `detection_delay_minutes`

Once one of these exists, Project 4 can add real confusion-matrix metrics.

Until then, proxy metrics remain the honest boundary.

---

## What Project 4 can claim today

Project 4 can honestly claim:

* The platform trains an unsupervised Isolation Forest anomaly model.
* The active model is versioned as `v002`.
* The active model was trained from a real-source extraction.
* The model has baseline anomaly-rate statistics.
* The model has feature-level baseline statistics.
* Batch and online inference paths persist prediction evidence.
* Prediction evidence includes model version, threshold used, anomaly score, entity ID, and payload hash.
* Drift checks compare current behavior against baseline feature statistics.
* Prometheus metrics expose inference, anomaly, drift, rollback, and latency signals.
* Grafana dashboards visualize operational model behavior.
* Alert events capture drift and anomaly-related operational signals.
* Rollback controls can revert to a previous stable model.
* The platform documents false-positive and false-negative risk honestly.
* Real precision, recall, false-positive rate, and false-negative rate are not claimed without labels.

---

## What Project 4 must not claim today

Project 4 must not claim:

* real production precision
* real production recall
* real F1 score
* real false-positive rate
* real false-negative rate
* human-reviewed anomaly accuracy
* fraud detection accuracy
* incident detection accuracy
* fully automated rollback based on proven model correctness
* cloud deployment if only local execution exists
* real labels if only proxy metrics exist

This is the line between credible engineering and project fiction.

Do not cross it.

---

## Operational failure modes

### Failure mode: threshold too aggressive

Result:

* too many anomalies
* too many alerts
* operator fatigue
* possible unnecessary rollback

Mitigation:

* reduce sensitivity
* review false-positive candidates
* add domain-specific suppression
* segment thresholds
* collect labels

### Failure mode: threshold too weak

Result:

* missed anomalies
* silent degradation
* late incident detection
* delayed rollback

Mitigation:

* increase sensitivity
* add risk rules
* review missed incidents
* add delayed labels
* retrain on newer data

### Failure mode: proxy metrics treated as real metrics

Result:

* misleading documentation
* bad operational confidence
* wrong threshold decisions
* weak model governance

Mitigation:

* label proxy metrics clearly
* avoid fake precision or recall claims
* document label limitations
* add delayed-truth evaluation when available

### Failure mode: one global threshold is too blunt

Result:

* some domains over-alert
* some domains under-alert
* global anomaly rate hides local failure modes

Mitigation:

* domain-specific thresholds
* entity-type segmentation
* separate business and system anomaly paths
* threshold review by source domain

### Failure mode: alert volume grows without investigation feedback

Result:

* alert fatigue
* no learning loop
* stale threshold
* no path from operations back into evaluation

Mitigation:

* add review outcomes
* track alert usefulness
* connect labels back to model evaluation
* add delayed-truth or human-review fields later

---

## Practical threshold review checklist

Before changing threshold behavior, review:

| Question                                           | Why it matters                                  |
| -------------------------------------------------- | ----------------------------------------------- |
| Has current anomaly rate moved away from baseline? | Detects behavior shift                          |
| Is drift warning or critical?                      | Adds feature-distribution context               |
| Are alerts increasing?                             | Shows operator impact                           |
| Are alerts useful?                                 | Separates signal from noise                     |
| Are there known business events?                   | Avoids flagging planned activity                |
| Are latency or error rates changing?               | Separates model issues from service issues      |
| Are missed incidents known?                        | Indicates false-negative risk                   |
| Are labels available?                              | Determines whether real metrics can be computed |
| Was the model recently changed?                    | Connects threshold behavior to model lifecycle  |
| Is rollback being considered?                      | Raises operational risk level                   |

Threshold tuning should leave an audit trail.

Recommended future threshold-change metadata:

* `threshold_change_id`
* `model_name`
* `model_version`
* `old_threshold`
* `new_threshold`
* `reason`
* `approved_by`
* `changed_at`
* `baseline_anomaly_rate`
* `current_anomaly_rate_before`
* `current_anomaly_rate_after`
* `alert_volume_before`
* `alert_volume_after`
* `rollback_related`
* `notes`

---

## Current completion status

Implemented:

* false-positive definition
* false-negative definition
* threshold tradeoff explanation
* proxy-versus-real metric boundary
* baseline anomaly-rate interpretation
* alert fatigue impact
* missed anomaly impact
* relationship to drift monitoring
* relationship to alert events
* relationship to rollback
* relationship to latency budget
* batch versus online threshold review guidance
* future label strategy
* operational failure modes
* threshold review checklist

Implemented in code:

* `src/anomaly_detection/evaluation.py`

Implemented helper functions:

* `estimate_threshold_from_scores`
* `summarize_threshold_tradeoff`

Documented in:

* `docs/false_positive_false_negative_analysis.md`

---

## Bottom line

Project 4 currently has a production-style anomaly detection platform with versioning, inference, prediction evidence, drift checks, metrics, dashboards, alerts, rollback, and latency validation.

The current model does not have real labels.

So the honest position is:

> Project 4 tracks anomaly behavior and operational degradation using proxy metrics today. Real false-positive and false-negative rates require labels and are planned for a future hardening pass.

That is credible.

That is production-minded.

And that is much stronger than pretending an unsupervised anomaly detector has ground truth it does not have.
