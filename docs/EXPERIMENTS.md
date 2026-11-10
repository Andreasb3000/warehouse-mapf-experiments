# Running the experiments

All commands below run from the repository root after installing the Python dependencies and running `make native`. `PYTHON=/path/to/python` and `IMAGE=your-image-tag` override Makefile defaults. The default image is `warehouse-congestion:local`.

## Fast checks

```bash
make smoke          # Tiny generated cases, including CPU neural-model checks
make test           # All source-only Python regression suites
make route-smoke    # Three native controller runs, 27 robots, 40 s each
```

The native build runs C++ safety, finite-task, parking, path-export and planner tests. The source-only suite has one optional regression against an archived recording; that fixture is deliberately not distributed and its test skips explicitly. Smoke success establishes working code paths, not a useful prediction model or a significant control improvement.

## 1. Predictors on the nine-zone floor

```bash
make record
make dataset JOBS=2
make train EPOCHS=30 JOBS=2 DEVICE=cpu
make evaluate
make figures
make timing JOBS=1 DEVICE=cpu
```

`make predictors` executes the first four steps sequentially. The full recording protocol uses 27 robots, 180 simulated seconds, 11 workload seeds and two safety rules (22 recordings). Seeds 7001–7006 train, 7007–7008 validate and 7009–7011 test. Both safety conditions for a workload stay in the same split.

Outputs:

| Location | Contents |
|---|---|
| `benchmark/data/manifest.json` | Recording configuration, split and provenance |
| `benchmark/data/recordings/` | Native states, accepted plans, tasks and logs |
| `benchmark/data/dataset.npz` | History, plan and map features, future labels |
| `benchmark/data/baseline_predictions.npz` | Zero, persistence, kinematic and joint-rollout predictions |
| `benchmark/results/local/` | Ridge model and predictions |
| `benchmark/results/learned/` | ConvLSTM and residual-ConvLSTM checkpoints and predictions |
| `benchmark/results/evaluation.json` | Common cellwise and aggregate errors |
| `benchmark/results/event_evaluation.json` | Slow-traffic event metrics |

For an end-to-end installation check, `make predictor-quick` uses one workload per split, 40-second recordings and two training epochs. **Do not report these results as the full benchmark.** It uses the same default data/model directories as the full pipeline; use a separate checkout or move the generated `benchmark/data` and `benchmark/results` directories aside before switching protocols. Existing incompatible recordings are not silently overwritten.

For custom recording subsets:

```bash
.venv/bin/python benchmark/code/record_dataset.py \
  --seeds 7001 7007 7009 --triggers outer_overlap --duration-s 60
```

Keep all three splits for training/evaluation. GPU acceleration is not required; the current training CLI supports CPU and macOS MPS. Choose hardware and threads explicitly when reporting timing.

## 2. Fleet scaling

After training the predictors:

```bash
make fleet FLEET_STUDY=fleet_run_1
make fleet-evaluate FLEET_STUDY=fleet_run_1 JOBS=2
make fleet-figures FLEET_STUDY=fleet_run_1
```

This records 27/54/81 robots, seeds 8301–8303, both safety rules and 240 simulated seconds. The larger fleets add robots and tasks to the same floor. They include lane starts; this differs slightly from the bay-based 27-robot training initialization. Frozen predictor evaluation measures transfer without retraining on fleet test data.

Recordings and their manifest are under `benchmark/data/fleet_pilots/fleet_run_1/`. Scores and latency summaries are under `benchmark/results/fleet_prediction/`. Shorter custom recordings need matching forecast-origin bounds; the full evaluator expects enough recorded future for every requested 20-second horizon.

## 3. Route-cost search

```bash
make route-search STUDY=route_run_1
# More independent workloads:
make route-search STUDY=route_run_2 SEEDS='8402 8403'
```

Each study matches tasks, starting states and Safety rule 2 across ordinary planning (`none`), nominal selection (`nominal`) and rollout selection (`rollout`). The default horizon is 240 seconds. The two selectors share the forecast-derived cost-menu mechanism; they score candidates differently. Forecasts are conditional on each candidate plan, not only on the initially accepted plan.

Results are under `benchmark/data/route_search/STUDY/`, including a manifest, source snapshot, native requests, exact candidate previews, selections and per-run metrics. The summary builder writes tables and plots under `benchmark/results/` and `benchmark/figures/`. Computation is reported separately from native task time.

## 4. Safety and capacity comparison

```bash
make safety STUDY=safety_run_1
make safety-figures STUDY=safety_run_1
```

This is 27 matched runs: 3 fleet sizes × 3 seeds × 3 safety conditions, using ordinary planning. Raw configuration keys map to presentation labels as follows:

| Raw key | Label |
|---|---|
| `disabled` | No safety rule |
| `body_in_outer` | Safety rule 1 |
| `outer_overlap` | Safety rule 2 |

All conditions retain the robot bodies, task system and planning dependencies. Results are under `benchmark/data/safety_capacity/STUDY/`. Comparisons isolate the outer slowdown response within this model; they do not alone identify physical aisle capacity or prove that safety-aware planning would eliminate the delays.

`make safety-figures` writes the matched plots, table and source values to that study's `analysis/` directory.

Verify those outputs with:

```bash
.venv/bin/python benchmark/model_audit/verify.py --study safety_run_1
```

Optional isolated controls execute the shortest, median and longest original 81-robot tasks individually under all three rules:

```bash
.venv/bin/python benchmark/model_audit/run_solo.py --study solo_run_1
.venv/bin/python benchmark/model_audit/verify.py \
  --study safety_run_1 --solo-study solo_run_1
```

The combined verification expects the default 240-second main design, including fleet 81 and seed 8401. These three selected trips are not an estimate of the entire fleet's free-flow average.

## 5. Other control baselines

```bash
make gates STUDY=gate_run_1
make search-gates STUDY=hold_run_1
```

Entry gates compare no intervention, reactive admission and forecast admission. Search gates evaluate alternative holds with rollouts. These exploratory baselines are retained even when they fail to improve the control. Their policies, thresholds and objectives are in `entry_gate_policy.py`, `search_admission_policy.py` and `long_hold_search_policy.py`.

The earlier direct-weighting study uses `make decisions` after training ridge. It compares zero, persistence, kinematic, ridge and joint-rollout forecasts. Its protocol is in `benchmark/configs/decision_protocol.json`; it is a separate experiment from the three-candidate route search. `run_congestion.py` and `congestion_cases.py` retain the earlier demand-placement pilots. Use their `--help` options for explicit study names and cases.

## 6. Junction scenarios and inspection

The simulator includes a six-robot T-junction, a deliberately crowded stress preset, a matched T/four-way junction benchmark and the nine-zone floor generator. Scenario JSON contains starts, goals, task stages and safety dimensions. Inspect or regenerate these using the scripts under `simulator/experiments/junction` and `simulator/experiments/zones`.

The browser viewer is optional. Its runtime comes from the pinned base image; unrelated web source and frontend build dependencies are not included in this research export. The main experiments and their figures run headlessly.

## Reproducibility checklist

For each reported result, retain the study manifest, source snapshot/hashes, image ID, parameters, workload seeds, model checkpoints and environment metadata. Compare methods with the same task population and safety rule. State whether unfinished tasks are capped and whether compute is charged. Report enough independent workloads to distinguish a repeatable effect from one workload or a planner CPU-cutoff fluctuation.

The current export uses corrected terminal parking. Historical runs and publication-only figure scripts are not prerequisites. Do not combine new measurements with old pre-fix measurements without treating the simulator revision as a changed experimental condition.
