PYTHON ?= .venv/bin/python
IMAGE ?= warehouse-congestion:local
STUDY ?= run_1
FLEET_STUDY ?= fleet_run_1
FLEETS ?= 27 54 81
SEEDS ?= 8401
FLEET_SEEDS ?= 8301 8302 8303
GATE_SEEDS ?= 8301 8302 8303
SAFETY_SEEDS ?= 8401 8402 8403
JOBS ?= 1
EPOCHS ?= 30
DEVICE ?= cpu
HORIZON ?= 240

.PHONY: help smoke test native route-smoke record dataset train evaluate predictors \
        predictor-quick fleet fleet-evaluate route-search safety gates search-gates \
        decisions timing doctor figures fleet-figures safety-figures

help:
	@echo 'Read README.md and docs/EXPERIMENTS.md.'
	@echo 'Quick checks: smoke test native route-smoke'
	@echo 'Predictors: record dataset train evaluate predictors predictor-quick timing'
	@echo 'Studies: fleet fleet-evaluate route-search safety gates search-gates decisions'
	@echo 'Overrides: PYTHON IMAGE STUDY FLEET_STUDY FLEETS SEEDS JOBS EPOCHS DEVICE HORIZON'

smoke:
	"$(PYTHON)" scripts/smoke.py --learning

doctor:
	"$(PYTHON)" scripts/doctor.py

test:
	"$(PYTHON)" scripts/test.py --quiet

native:
	"$(PYTHON)" benchmark/native/build.py --tag "$(IMAGE)"

route-smoke:
	"$(PYTHON)" benchmark/code/run_route_search.py --image "$(IMAGE)" --study route_smoke --fleets 27 --seeds 8401 --methods none nominal rollout --horizon-s 40

record:
	"$(PYTHON)" benchmark/code/record_dataset.py --image "$(IMAGE)"

dataset:
	"$(PYTHON)" benchmark/code/build_dataset.py --jobs "$(JOBS)" --combine

train:
	"$(PYTHON)" benchmark/code/local_model.py
	"$(PYTHON)" benchmark/code/learned.py --data benchmark/data/dataset.npz --index benchmark/data/dataset_index.json --output benchmark/results/learned --device "$(DEVICE)" --threads "$(JOBS)" --epochs "$(EPOCHS)" --patience 5 --batch-size 8 --hidden 4 --seed 2026 --skip-timing

evaluate:
	"$(PYTHON)" benchmark/code/evaluate.py
	"$(PYTHON)" benchmark/code/event_metrics.py

figures:
	"$(PYTHON)" benchmark/scripts/build_results.py

predictors:
	$(MAKE) record
	$(MAKE) dataset
	$(MAKE) train
	$(MAKE) evaluate

predictor-quick:
	"$(PYTHON)" benchmark/code/record_dataset.py --image "$(IMAGE)" --quick
	$(MAKE) dataset
	$(MAKE) train EPOCHS=2 JOBS=1
	$(MAKE) evaluate

timing:
	"$(PYTHON)" benchmark/code/learned.py --data benchmark/data/dataset.npz --index benchmark/data/dataset_index.json --output benchmark/results/learned --device "$(DEVICE)" --threads "$(JOBS)" --timing-only
	"$(PYTHON)" benchmark/code/benchmark_latency.py

fleet:
	"$(PYTHON)" benchmark/code/run_fleet.py --image "$(IMAGE)" --study "$(FLEET_STUDY)" --fleets $(FLEETS) --seeds $(FLEET_SEEDS) --triggers body_in_outer outer_overlap --horizon-s "$(HORIZON)"

fleet-evaluate:
	"$(PYTHON)" benchmark/code/evaluate_fleet_predictors.py --manifests "benchmark/data/fleet_pilots/$(FLEET_STUDY)/manifest.json" --jobs "$(JOBS)"
	"$(PYTHON)" benchmark/code/benchmark_fleet_latency.py --manifests "benchmark/data/fleet_pilots/$(FLEET_STUDY)/manifest.json"

fleet-figures:
	"$(PYTHON)" benchmark/scripts/build_fleet_results.py --manifests "benchmark/data/fleet_pilots/$(FLEET_STUDY)/manifest.json"

route-search:
	"$(PYTHON)" benchmark/code/run_route_search.py --image "$(IMAGE)" --study "$(STUDY)" --fleets $(FLEETS) --seeds $(SEEDS) --methods none nominal rollout --horizon-s "$(HORIZON)"
	"$(PYTHON)" benchmark/scripts/build_route_search_results.py --studies "benchmark/data/route_search/$(STUDY)"

safety:
	"$(PYTHON)" benchmark/code/run_safety_capacity.py --image "$(IMAGE)" --study "$(STUDY)" --fleets $(FLEETS) --seeds $(SAFETY_SEEDS) --conditions disabled body_in_outer outer_overlap --horizon-s "$(HORIZON)"

safety-figures:
	"$(PYTHON)" benchmark/model_audit/analyze.py --study "$(STUDY)"

gates:
	"$(PYTHON)" benchmark/code/run_entry_gates.py --image "$(IMAGE)" --study "$(STUDY)" --fleets $(FLEETS) --seeds $(GATE_SEEDS) --methods none reactive forecast --horizon-s "$(HORIZON)"
	"$(PYTHON)" benchmark/scripts/build_entry_gate_results.py --study "benchmark/data/entry_gates/$(STUDY)"

search-gates:
	"$(PYTHON)" benchmark/code/run_search_gates.py --image "$(IMAGE)" --study "$(STUDY)" --fleets $(FLEETS) --seeds $(SEEDS) --methods none search_h10_hold6 --horizon-s "$(HORIZON)"
	"$(PYTHON)" benchmark/scripts/build_search_results.py --studies "benchmark/data/admission_search/$(STUDY)" --evaluation-studies "benchmark/data/admission_search/$(STUDY)"

decisions:
	"$(PYTHON)" benchmark/code/run_decisions.py --image "$(IMAGE)" --jobs "$(JOBS)"

