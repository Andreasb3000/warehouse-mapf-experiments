#!/usr/bin/env python3
"""Run or preview the connected nine-zone transport experiment."""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.junction.scenario import load_scenario
from experiments.zones.analyze import analyze

ROBOT_TRIGGERS = ("body_in_outer", "outer_overlap")


def optional_git(*args):
    """Source archives may be run before they are initialized as a Git repo."""
    try:
        result = subprocess.run(["git", "-C", str(ROOT), *args], text=True,
                                capture_output=True)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def flags(parameters):
    return [f"--{key}={str(value).lower() if isinstance(value, bool) else value}" for key, value in parameters.items()]


def docker_prefix(host=None):
    """Use an explicit daemon or Docker's normal environment/context selection."""
    return ["docker"] + (["--host", host] if host else [])


def capture(command):
    return subprocess.check_output(command, text=True).strip()


def native_paths():
    return sorted([*(ROOT / "server/src").rglob("*.cpp"), *(ROOT / "server/inc").rglob("*.h"),
                   *(p for p in (ROOT / "client").rglob("*")
                     if p.suffix in (".cpp", ".h") and "externalDependencies" not in p.parts)])


def pbs_paths():
    return sorted([*(ROOT / "planner/PBS/src").rglob("*.cpp"),
                   *(ROOT / "planner/PBS/inc").rglob("*.h")])


def check_pbs_sources(docker, image_id):
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in pbs_paths()}
    try:
        compiled = json.loads(capture(docker + ["run", "--rm", "--platform", "linux/amd64", image_id,
                                               "cat", "/usr/project/pbs-source-sha256.json"]))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise SystemExit("PBS source metadata is missing or invalid. Run with --build.") from error
    if hashes != compiled:
        raise SystemExit("PBS sources differ from this image. Run with --build.")
    return compiled


def build(docker, image):
    command = [sys.executable, str(ROOT.parent / "benchmark/native/build.py"), "--tag", image]
    if "--host" in docker:
        command += ["--docker-host", docker[docker.index("--host") + 1]]
    subprocess.run(command, check=True)


def mounts(inputs):
    return ["-v", f"{inputs}:/inputs:ro",
            "-v", f"{ROOT / 'run_lifelong.py'}:/usr/project/run_lifelong.py:ro",
            "-v", f"{ROOT / 'experiments/junction/scenario.py'}:/usr/project/experiments/junction/scenario.py:ro"]


def environment(config):
    result = ["-e", f"LSMART_SCENARIO_FILE=/inputs/{config['scenario']}"]
    if config.get("record_plans", False):
        result += ["-e", "LSMART_RECORD_PLANS=1"]
    return result


def scenario_with_policy(scenario, *, safety=None, robot_trigger=None, inner_stop=None):
    """Apply explicit overrides to every robot without changing field geometry.

    Missing new fields preserve the native legacy defaults: body-entry triggers
    and an active inner stop. Per-agent overrides cannot defeat an explicit
    command-line choice or a matched experimental condition.
    """
    copied = deepcopy(scenario)
    policy = copied.setdefault("safety", {})
    policy.setdefault("robot_trigger", "body_in_outer")
    policy.setdefault("inner_stop_enabled", True)
    overrides = {key: value for key, value in (
        ("enabled", safety), ("robot_trigger", robot_trigger),
        ("inner_stop_enabled", inner_stop)) if value is not None}
    policy.update(overrides)
    for agent in copied["agents"]:
        if "safety" in agent:
            agent["safety"].update(overrides)
    return copied


def experiment_cases(scenario, *, safety=True, robot_trigger=None, inner_stop=None,
                     compare_safety=False, compare_triggers=False):
    """Return named scenario copies with matched controls for each comparison."""
    if compare_safety and compare_triggers:
        raise ValueError("Choose only one comparison")
    if compare_triggers:
        return [(trigger, scenario_with_policy(scenario, safety=True,
                                               robot_trigger=trigger, inner_stop=False))
                for trigger in ROBOT_TRIGGERS]
    return [("safety_on" if enabled else "safety_off",
             scenario_with_policy(scenario, safety=enabled, robot_trigger=robot_trigger,
                                  inner_stop=inner_stop))
            for enabled in ((True, False) if compare_safety else (safety,))]


def config_for_run(config, index):
    """Pair each index across conditions while varying repeated-run seeds."""
    if type(index) is not int or index < 1:
        raise ValueError("Run index must be a positive integer")
    result = deepcopy(config)
    base_seed = result["parameters"].get("seed", 42)
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("The base seed must be a nonnegative integer")
    result["parameters"]["seed"] = base_seed + index - 1
    return result


def condition_metadata(case_name, config, scenario, index, run_dir):
    """Describe effective fleet policy, including heterogeneous agent overrides."""
    policies = [{**scenario.get("safety", {}), **agent.get("safety", {})}
                for agent in scenario["agents"]]

    def common(key, default):
        values = {policy.get(key, default) for policy in policies}
        return values.pop() if len(values) == 1 else "mixed"

    return {"condition": case_name, "run_index": index,
            "seed": config["parameters"]["seed"], "run_directory": str(run_dir),
            "safety_enabled": common("enabled", False),
            "robot_trigger": common("robot_trigger", "body_in_outer"),
            "inner_stop_enabled": common("inner_stop_enabled", True),
            "input_sha256": {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
                             for name in ("config.json", config["scenario"],
                                          scenario["map_file"], scenario["tasks_file"])}}


def copy_inputs(source, destination, config, scenario, safety=None, *, robot_trigger=None,
                inner_stop=None):
    destination.mkdir(parents=True, exist_ok=False)
    for name in (scenario["map_file"], scenario["tasks_file"], "zone_allocation.csv", "layout.svg"):
        if (source / name).exists():
            shutil.copyfile(source / name, destination / name)
    copied = scenario_with_policy(scenario, safety=safety, robot_trigger=robot_trigger,
                                  inner_stop=inner_stop)
    write_json(destination / config["scenario"], copied)
    write_json(destination / "config.json", config)
    return copied


def execute(docker, image_id, output, config, scenario, index):
    run_dir = output / f"run-{index:02d}"
    config = config_for_run(config, index)
    scenario = copy_inputs(output, run_dir, config, scenario)
    condition = condition_metadata(output.name, config, scenario, index, run_dir)
    write_json(run_dir / "condition.json", condition)
    provenance = json.loads((output / "provenance.json").read_text())
    write_json(run_dir / "provenance.json", {**provenance, **condition})
    params = dict(config["parameters"], container=True, visualizer="web", save_stats=True,
                  scenario_file=f"/inputs/{config['scenario']}", stats_name="/results/stats.json",
                  argos_config_filepath="/results/simulation.argos", output_log="/results/simulator.log")
    name = "warehouse-zones-" + uuid.uuid4().hex[:10]
    command = docker + ["run", "--rm", "--init", "--name", name, "--platform", "linux/amd64",
                        "--ulimit", "nofile=65536:65536", *mounts(run_dir), *environment(config),
                        "-v", f"{run_dir}:/results", image_id, "python3", "run_lifelong.py",
                        f"/inputs/{scenario['map_file']}", *flags(params)]
    write_json(run_dir / "command.json", command)
    print(f"Running {run_dir}", flush=True)
    started = time.monotonic()
    try:
        with (run_dir / "launcher.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=config.get("wall_timeout_s", 300))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        subprocess.run(docker + ["stop", "--time", "5", name], capture_output=True)
        raise
    metrics = analyze(run_dir)
    metrics.update(condition)
    metrics["wall_runtime_s"] = time.monotonic() - started
    if result.returncode:
        metrics["valid"] = False
        metrics["errors"].append(f"Container exited with code {result.returncode}")
    write_json(run_dir / "zone_metrics.json", metrics)
    if metrics["valid"]:
        from experiments.zones.viewer import write_viewer
        if config.get("record_plans", False):
            from experiments.prediction.corrupt import analyze as score_corruptions
            score_corruptions(run_dir, seed=config["parameters"]["seed"])
        write_viewer(run_dir)
    print(json.dumps({k: metrics.get(k) for k in ("valid", "errors", "pickups_completed", "deliveries_completed",
                                                  "slowed_move_robot_seconds", "stopped_command_robot_seconds", "wall_runtime_s")}), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "nine_zones/config.json")
    parser.add_argument("--docker-host", default=os.environ.get("DOCKER_HOST") or None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--safety", choices=("on", "off"), default="on")
    parser.add_argument("--robot-trigger", choices=ROBOT_TRIGGERS,
                        help="Override the outer slowdown rule for every robot")
    parser.add_argument("--inner-stop", choices=("on", "off"),
                        help="Override the optional inner stopping field for every robot")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare-safety", action="store_true", help="Run matched on/off experiments")
    mode.add_argument("--compare-triggers", action="store_true",
                      help="Compare body entry and outer overlap; enable outer fields only in both cases")
    mode.add_argument("--view", action="store_true"); mode.add_argument("--stop-view", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.compare_triggers and (args.safety == "off" or args.inner_stop == "on" or args.robot_trigger):
        parser.error("--compare-triggers requires safety on, inner stop off, and both robot trigger rules")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    port = args.port or config.get("preview_port", 3004)
    if not 1 <= port <= 65535:
        parser.error("Invalid preview port")
    docker = docker_prefix(args.docker_host)
    preview = f"warehouse-robots-zones-{port}"
    if args.stop_view:
        subprocess.run(docker + ["stop", preview], check=True); return 0
    scenario = json.loads((config_path.parent / config["scenario"]).read_text())
    layout = json.loads((config_path.parent / scenario["map_file"]).read_text())["layout"]
    load_scenario(config_path.parent / config["scenario"], layout, config["parameters"]["num_agents"])
    cases = experiment_cases(scenario, safety=args.safety == "on", robot_trigger=args.robot_trigger,
                             inner_stop=None if args.inner_stop is None else args.inner_stop == "on",
                             compare_safety=args.compare_safety, compare_triggers=args.compare_triggers)
    if args.build:
        build(docker, config["image"])
    image_id = json.loads(capture(docker + ["image", "inspect", config["image"]]))[0]["Id"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in native_paths()}
    compiled = json.loads(capture(docker + ["run", "--rm", "--platform", "linux/amd64", image_id,
                                           "cat", "/usr/project/server-source-sha256.json"]))
    if hashes != compiled:
        raise SystemExit("Client/server sources differ from this image. Run with --build.")
    compiled_pbs = check_pbs_sources(docker, image_id)
    if args.view:
        inputs = ROOT / ".local" / f"zones-preview-{port}-{uuid.uuid4().hex[:6]}"
        copied = copy_inputs(config_path.parent, inputs, config, cases[0][1])
        params = dict(config["parameters"], map_filepath=f"/inputs/{copied['map_file']}")
        subprocess.run(docker + ["run", "-d", "--rm", "--init", "--name", preview, "--platform", "linux/amd64",
                                 "--ulimit", "nofile=65536:65536", "-p", f"127.0.0.1:{port}:3000",
                                 *mounts(inputs), *environment(config), image_id, "lsmart-viz", *flags(params)], check=True)
        print(f"http://localhost:{port} — Start Simulation; safety {args.safety}."); return 0
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    output = args.output_dir.resolve() if args.output_dir else HERE / "results" / f"{stamp}-{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"image_id": image_id, "compiled_native_source_sha256": compiled,
                  "compiled_pbs_source_sha256": compiled_pbs,
                  "compiled_sources_match_checkout": True,
                  "base_revision": optional_git("rev-parse", "HEAD"),
                  "git_diff": optional_git("diff"),
                  "analysis_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for directory in (HERE, ROOT / "experiments/prediction") for p in directory.glob("*.py")}}
    provenance["mounted_source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "run_lifelong.py", ROOT / "experiments/junction/scenario.py", ROOT / "experiments/baseline/analyze.py")}
    metrics = []
    for case_name, case_scenario in cases:
        case = output / case_name
        copied = copy_inputs(config_path.parent, case, config, case_scenario)
        write_json(case / "provenance.json", provenance)
        for index in range(1, args.runs + 1):
            metrics.append(execute(docker, image_id, case, config, copied, index))
    summary = {"valid": all(m["valid"] for m in metrics), "results_directory": str(output),
               "comparison": "robot_trigger" if args.compare_triggers else "safety_enabled" if args.compare_safety else None,
               "runs": [{k: m.get(k) for k in ("scenario", "condition", "run_index", "seed", "run_directory",
                                                "robot_trigger", "inner_stop_enabled", "valid", "errors", "safety_enabled", "pickups_completed",
                                                "deliveries_completed", "slowed_move_robot_seconds", "stopped_command_robot_seconds", "wall_runtime_s")} for m in metrics]}
    if args.compare_triggers:
        from experiments.zones.comparison import summarize
        comparison = summarize(output)
        summary["comparison_report"] = str(output / "comparison.json")
        summary["valid"] = summary["valid"] and comparison["valid"]
    write_json(output / "summary.json", summary); print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
