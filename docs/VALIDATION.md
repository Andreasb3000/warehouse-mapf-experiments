# Export validation

Validated on 2026-11-09 using a newly created Python 3.12.12 environment on macOS ARM64 and a Docker engine running the `linux/amd64` native image through emulation. Dependencies were installed from the documented requirements; exact resolved versions are in `constraints-tested.txt`.

## Completed checks

| Check | Result |
|---|---|
| Python regression suites | 365 tests: 364 passed, one optional archived-recording regression skipped |
| Python smoke with learning | 75 tests, including generated 27/54/81-robot cases and tiny CPU model training |
| Public ARGoS download | Fresh download passed the pinned SHA-256 check |
| Native build | Client, server and PBS rebuilt; ten native regression groups passed |
| Junction run | Six robots completed their crossings and parked by 38 s in a valid 60 s recording |
| Route selection | Ordinary, nominal and rollout methods: three 27-robot, 40 s runs; both selectors exercised nonzero candidate selection |
| Predictor pipeline | Six new 40 s recordings; 24 forecast origins; ridge and both neural variants trained; all seven predictors evaluated |
| Fleet transfer | 27/54/81 robots under both safety rules: six 40 s recordings; frozen predictors evaluated at 12 origins |
| Fleet prediction timing | All seven predictors, CPU-only, all three fleet sizes |
| Entry gates | Three 27-robot, 40 s method runs and result generation |
| Admission search | No-intervention and hold-search 27-robot, 40 s runs and result generation |
| Safety ablation | Nine 20 s runs: three fleet sizes × three safety conditions, one workload seed |
| Safety analysis and verification | Study-local figures and tables; 895 source/recording checks passed |
| Direct prediction weighting | Five 27-robot, 40 s pilot runs: zero, persistence, kinematic, ridge and joint rollout |
| Result builders | Fresh predictor, fleet, route, gate, hold-search and safety outputs generated successfully |
| Source packaging | Documentation links, retained licenses, excluded outputs and absence of machine-specific paths checked |
| Extracted source ZIP | All 75 smoke checks and the full 365-test suite passed again in a separate directory, with the same optional skip |

The native checks cover safety geometry, finite tasks, entry gates, admission bridges, pure candidate previews, fallback parking, movement penalties, route selection, service-marker export and terminal parking.

## Scope

These runs verify installation, execution and result generation. They are **not** new evidence of predictor quality or controller effectiveness. The small training run uses two epochs and one workload per split. Short controller runs often end before any delivery; completing the configured recording and its integrity checks is the integration criterion.

The full 180/240-second workload sweeps, 30-epoch training and nine isolated native trip controls were not rerun for this export. All nine isolated-control inputs did pass scenario validation. The direct-weighting acceptance evaluator correctly refuses pilot recordings; their per-run diagnostics remain available.

No historical recordings or trained weights are required by the distributed source. Generated validation data, figures, models, build proofs and caches are excluded from the source ZIP and Git index. Rebuild and rerun the documented commands to create your own artifacts.

GitHub Actions is configured for the source-only tests on Ubuntu/Python 3.12; that remote workflow has not been run as part of this local export. Native build and end-to-end checks were performed locally as described above.
