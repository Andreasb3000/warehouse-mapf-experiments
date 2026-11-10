# Modeling and evaluation

## State and forecasts

Native motion is recorded at 0.1-second intervals. A forecast origin uses observed poses, headings, controller queues, accepted actions and dependencies, tasks, map geometry and the safety configuration. It can be an intermediate state in a run.

The common prediction target has two map layers: robot occupancy and slow occupancy. Slow occupancy counts robots whose average speed over the preceding two seconds is at most 0.3 m/s, whose heading departure is at most 0.1 rad, and whose position is more than 0.25 m from assigned service locations. It describes slow traffic, not its cause.

Targets are time averages over 0–5, 5–10 and 10–20 seconds into the future. Each cell contains an average robot count. Multiplying by the window duration gives robot-seconds. Histories use six observations spanning the previous ten seconds. Entire workload seeds, including their paired safety conditions, belong to one train/validation/test split.

## Predictor methods

| Method | Construction |
|---|---|
| Zero | Predict zero occupancy in both output layers |
| Persistence | Repeat the latest observed occupancy and slow occupancy |
| Kinematic / plan projection | Advance accepted actions without modeled fleet interactions |
| Joint rollout | Advance robots together with safety fields, dependencies and controller state |
| Local ridge | Regress future cell values on local history, plan and map features |
| ConvLSTM | Encode spatial history and plan features with a convolutional recurrent model |
| Residual ConvLSTM | Correct plan-projected occupancy and learn its slow fraction |

The separate plan-feature tensor sets its slow channel to zero; it is an input feature and should not be confused with the evaluated kinematic baseline. The neural models are lightweight adaptations for this benchmark, not paper replications. Exact feature and loss definitions live in `benchmark/code/features.py` and `learned.py`.

Cellwise MAE/RMSE are reported on roads, junction/approach regions and positive target cells. Positive-cell filtering uses future truth only during scoring. Empty cells can make an uninformative zero predictor look competitive, so inspect positive-cell errors, event precision/recall, aggregate robot-seconds and the workload distribution together. Cellwise error does not give distance credit for a prediction that misses by one neighboring cell.

## Route-cost search

At each two-second review, a 20-second rollout of the current accepted plan proposes a slow-traffic field. Window weights are 0.5, 0.3 and 0.2; a 3×3 maximum expands the field. Destination-cell penalties use multipliers 0, 2 and 8, with an upper cap. A MOVE costs 1 plus its destination penalty; a WAIT costs 1. Costs influence preferences, while PBS still handles space-time conflicts and robot priorities.

Candidates share the current planning starts, original service targets and native commitment cut. Each receives the same per-solve CPU budget. A successful path is converted to an action-graph preview before scoring. The chosen preview is installed exactly once.

For candidate $k$, let $\widehat T_i^{(k)}$ be robot $i$'s predicted remaining time to its last currently planned service. A service may be a pickup if the delivery has not yet been planned. All remaining service IDs and locations must match across alternatives.

$$J_k = \frac{1}{N}\sum_{i=1}^{N}\widehat T_i^{(k)}.$$

- **Ordinary planning:** only the zero-penalty plan.
- **Nominal selection:** remaining travel at nominal speed plus turns and service; no additional interaction delay.
- **Rollout selection:** jointly simulate the candidate for 30 seconds. Use the completion offset if its terminal service finishes; otherwise use 30 seconds plus nominal remaining work. Confirmed completed robots contribute zero and remain in the fleet denominator.

Choose the best eligible alternative only if it improves $J_0$ by more than 0.5 seconds. Otherwise keep ordinary planning. If the ordinary solve fails, use the native fallback and skip alternatives. A review is an opportunity to intervene, not a requirement.

Both selectors use a forecast to generate candidates. Their difference is the candidate-scoring model. Rollout scoring needs a model conditioned on alternative plans; a slowdown-only forecast under the current plan is insufficient by itself.

The selectors can also spend three PBS solves where ordinary planning spends one. A comparison with ordinary planning therefore combines the cost-menu search and its prediction model; it does not isolate the value of prediction at matched total planning compute.

With no switching threshold, perfect candidate scores would select the best available candidate, with no degradation relative to an included baseline under the same fully evaluated execution policy. Our 0.5-second threshold permits retaining the baseline for smaller improvements. Neither rule guarantees global optimality: varying a scalar multiplier can miss feasible plans altogether. For an idealized uncapped objective $L(P)+\lambda C(P)$, let $L$ be nominal planner cost, including waits, and $C$ the accumulated fixed-map penalties. A sufficient special case is an exact planner and a true objective $T(P)=aL(P)+bC(P)+c$ for every feasible joint plan, with $a>0$ and $b\ge0$; then $\lambda=b/a$ recovers that objective. The matching multiplier must be available, and penalty caps must not distort it. Neither the current finite menu, an accurate slow-occupancy map nor the current rollout tail establishes this special case.

## Execution outcomes

Primary controller outcomes are completed original deliveries and mean capped completion time over **all** assigned tasks. At a 240-second horizon, unfinished tasks contribute 240 seconds. Do not average only the successful deliveries.

Report planning failures, fallback use, eligible alternatives and selected interventions separately. A failed planner call does not establish mathematical infeasibility. No eligible alternative does not mean no useful detour exists.

Computation pauses the native simulator in these studies. Predictor latency, candidate planning time and whole-review latency are measured separately and are not added to simulated completion time. Hardware, emulation and the planner's CPU cutoff affect measured runtime and can affect selected plans.

## Limits

The rollout approximates native execution; it is not a complete engine checkpoint. Future replanning and native contact physics are absent. Its nominal tail omits subsequent interaction delays. The floor and finite workloads are deliberately small; limited communication and thousand-robot coordination are research extensions. Current corrected runs should be compared with matched current controls, rather than pooled with historical runs from before the parking fix.
