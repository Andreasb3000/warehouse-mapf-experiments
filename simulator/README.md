# Simulator source

This directory contains the modified LSMART runtime used by the experiments:
the ARGoS robot controller, action-dependency execution server, PBS planner,
scenario generators, and local prediction utilities. See the repository root
for the experiment workflow and Python dependencies.

Build the current native image from the repository root:

```sh
python benchmark/native/build.py
```

The build uses a public, digest-pinned LSMART image and a checksum-pinned ARGoS
development package. It compiles the controller, server and PBS, runs their
native regressions, and writes fresh image/source provenance. A working Docker
runtime is required. `DOCKER_HOST`, the active Docker context, or an explicit
`--docker-host` selects the daemon. Linux/amd64 is used on all hosts; ARM machines
need Docker's amd64 emulation.

For a small interactive simulation:

```sh
python simulator/experiments/zones/run.py --view --robot-trigger outer_overlap --inner-stop off
```

Open the printed localhost URL and select **Start Simulation**. Stop it with:

```sh
python simulator/experiments/zones/run.py --stop-view
```

For a recorded run, replace `--view` with
`--output-dir simulator/.local/my-zone-run`. Output directories must be fresh.
The benchmark workflow generates its own 27/54/81-robot workloads.

Safety labels used in this repository:

| Display name | Configuration | Response |
|---|---|---|
| No safety rule | `safety.enabled=false` | No safety-field-induced slowdown |
| Safety rule 1 | `robot_trigger=body_in_outer` | Slow when another robot body enters the outer field |
| Safety rule 2 | `robot_trigger=outer_overlap` | Slow when outer fields overlap |

Inner stopping is a separate option and is disabled in the main experiments.
Physical bodies and action dependencies remain under all three conditions.
Terminal parking is corrected in this code: robots remain at their final bays.
This does not make PBS model the execution-time safety slowdown.

Only PBS is rebuilt and supported by the packaged experiment commands. The
generic upstream launcher retains names of other planners, but their sources
are outside this focused export. The public runtime image supplies the browser
visualizer; its source is not duplicated here.

Upstream simulator licensing is in [LICENSE.txt](LICENSE.txt). PBS has separate
[USC license terms](planner/PBS/license.md), and vendored RPC dependencies retain
their own notices. See the repository's third-party notices before reuse.
