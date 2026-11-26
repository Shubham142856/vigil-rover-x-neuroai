# VIGIL-ROVER X NeuroAI

Drosophila Connectome-Informed NeuroAI for Autonomous Collision Avoidance, Directional Memory and Active Blind-Spot Recovery

The project studies whether biological wiring changes a rover’s ability to anticipate collisions, maintain heading through visual loss and choose a useful recovery action. PyBullet provides the rover benchmark. FlyGym provides a separate reference environment for selected fly behaviors.

## Project files

- [Research and implementation specification](NEURO_RESEARCH_PLAN.md) defines the circuit inputs, learning targets, engineering interfaces and comparison models. The [Word edition](VIGIL_ROVER_X_NeuroAI_Plan.docx) and [PDF edition](VIGIL_ROVER_X_NeuroAI_Plan.pdf) contain the same specification and a linked source index.
- [Milestone tracker](Implementation_Acceptance_Gates.csv) records the 20-week sequence, acceptance evidence and status (G02–G06 and G08–G10 met; G01 historical 2D source unavailable; G07 Kaggle GPU training ready/pending run).
- [Research resources](Research_Resources.csv) indexes the literature, source data, access guidance and software. A catalog entry identifies where to obtain a resource.
- [Graph extractor](extract_graph.py) pools directed synapse counts across rows for reviewed neuron IDs. [Extractor tests](test_extract.py) use synthetic input.
- [Classical stopping baseline](baseline_policy.py) provides a partial M1 comparator for delivered camera and tactile observations. Its [tests](test_policy.py) exercise decision priorities. It has no navigation benchmark result.

## First reproducible steps

From the extracted package directory on Windows, run the standard-library tests:

```powershell
python -B -m unittest discover -s . -p "test_*.py" -v
```

For the first biological graph, use the FAFB v783 products in resource R9a. Record the source URL, release, download date, SHA-256, upstream filter and reuse terms for each table. Verify the annotations and connections belong to the same release. Curate separate cell ID lists for the escape and compass circuits, including laterality and column location where available. Preserve IDs as decimal strings.

Inspect the actual CSV header before running the extractor. The example assumes the specified column names and a local working copy of the connections table.

```powershell
python extract_graph.py --ids ids_escape.txt --connections connections.csv.gz --pre pre_root_id --post post_root_id --count syn_count --minimum 5 --version FAFB-v783 --out escape_graph
```

Inspect the emitted nodes, edges and manifest. Compare thresholds of one and five on an unfiltered input and record both results. Review missing cell types, isolated nodes, edge direction and sign evidence before fitting neural dynamics.

## Workspace and training status

The D-drive workspace keeps the earlier supplied ZIPs under `references/package-originals`, the earlier simulator report under `references`, and the published EMD–LPLC2–GF code under `references/EMD-LPLC2-GF-model` (checkout `50c7c8f3f74052b2562ccba967c9f1032f37405b`). These are separate from the portable starter package.

Kaggle training is scheduled for weeks 13–14. Its inputs are versioned spatial image sequences, validated circuit states and counterfactual labels. A run should record the source and model versions, world-level split, actual CUDA device, frozen thresholds, metrics and checkpoint. The resource index links the official CLI commands. The available evidence at this revision consists of synthetic extractor and policy tests. Biological validation, GPU training and closed-loop performance are pending.
## Running the initial build on D

From D:\rover, use the D-drive Python environment:

```powershell
.\rover-python.ps1 -m unittest discover -s VIGIL_NEURO_FINAL -v
.\rover-python.ps1 VIGIL_NEURO_FINAL/rover_sim.py --scene open --fault freeze --steps 80 --out runs/my-first-drive
.\rover-python.ps1 VIGIL_NEURO_FINAL/vision_assay.py --out runs/vision-assay
.\rover-python.ps1 VIGIL_NEURO_FINAL/build_connectome.py --data data/fafb-v783 --out runs/connectome
```

The simulation uses a velocity-driven rigid chassis and contact-query feelers under planar motion. Replay samples contain delivered spatial RGB, source/delivery times and a separate evaluator state. The initial visual assay measures centered radial image motion on synthetic clips. Circuit response fitting, physical wheel dynamics and trained recovery remain pending.

The first candidate graph audit in runs/connectome/audit.json records release hashes and the thresholded Princeton product used. Cell selection uses exact primary annotations. Validate physiological signs, column correspondence and held-out responses before neural training. These working-tree additions postdate the design ZIP.

## Biological wiring experiments

The supplied November 26 master specification is retained under `references/NeuroAI_master_specification_2026-11-26.txt` in the workspace. Its ten research phases govern progression. The detailed protocol is in NEURO_RESEARCH_PLAN.md. The empirical build status below supersedes the original design-stage inventory.

From D:\rover, export the biological and matched random escape networks, then compare their responses to the same LPLC2 input:

```powershell
.\rover-python.ps1 VIGIL_NEURO_FINAL/graph_controls.py --graph runs/connectome/escape --out runs/graph-controls/escape
.\rover-python.ps1 VIGIL_NEURO_FINAL/circuit_response.py --graph runs/graph-controls/escape --annotations runs/connectome/escape/annotations.csv --source LPLC2 --target DNp01 --out runs/circuit-response/escape
```

The compass run uses the corresponding compass directories, PEG input and EPG readout. Both runs use the same provisional unsigned rate equation and three frozen rewiring seeds. The degree-preserving controls retain each cell's incoming and outgoing edge counts and outgoing weight distribution. Spatial input calibration, synaptic signs and anatomical compass phase assignments remain open. These experiments establish computational signal propagation; behavioral and physiological validation remain separate gates.

## Zero-false-positive vision, blind-spot, and recovery checks

From D:\rover, run the G04 biological vision benchmark, zero-false-positive camera-escape braking evaluation, G06 paired blind-spot dataset generator, and G03/G08 tactile outage recovery checks:

```powershell
.\rover-python.ps1 VIGIL_NEURO_FINAL/vision_benchmark.py --out runs/vision-benchmark
.\rover-python.ps1 VIGIL_NEURO_FINAL/camera_escape.py --graph runs/graph-controls/escape --annotations runs/connectome/escape/annotations.csv --out runs/camera-escape-zero-fp --pairs 8
.\rover-python.ps1 VIGIL_NEURO_FINAL/blindspot_dataset.py --out runs/blindspot-dataset
.\rover-python.ps1 VIGIL_NEURO_FINAL/recovery_policy.py --out runs/recovery-eval
```

Observed results (`39` passing unit tests):
- `runs/vision-benchmark/summary.json`: `96x128` RGB at `10 Hz` and `66°` vertical FOV; multiplicative LPLC2 response peaks at `1.0` on `approach` and `0.0` across `recede`, `translate`, `yaw`, `near_miss`, and `empty`.
- `runs/camera-escape-zero-fp/summary.json`: across `8` collision courses and `8` near misses (`seeds 20000..20015`), bearing-gated LPLC2/LC4 drive and zero-FP threshold calibration yield `0` false warnings, `0` closed-loop collisions, and `8/8` near misses passed (`1.025 s` biological median lead; `1.10 s` rewired median lead).
- `runs/blindspot-dataset/manifest.json`: `80` counterfactual branch records across `8` paired indistinguishable worlds (`max_pre_branch_pixel_diff = 0`, `oracle_keys_in_inputs = 0`, `pair_split_leakage = 0`).
- `runs/recovery-eval/summary.json`: `0` body collisions across bilateral wall stop, one-sided contact with blocked escape during camera freeze (safe abstention), and one-sided contact with verified-clear opposite sector during drop outage.

## Unfiltered connectome audit, directional memory, and scaled G07 pipeline

From D:\rover, run the `G02` unfiltered `min_synapses` `1 vs 5` sensitivity + NT sign audit, `G05` connectome-constrained compass benchmark, and `G07` scaled counterfactual dataset generator:

```powershell
.\rover-python.ps1 VIGIL_NEURO_FINAL/audit_unfiltered_connectome.py
.\rover-python.ps1 VIGIL_NEURO_FINAL/compass_benchmark.py --out runs/compass-benchmark
.\rover-python.ps1 VIGIL_NEURO_FINAL/scale_dataset.py --out runs/scaled-dataset
```

Observed results (`43` passing unit tests):
- `runs/connectome/unfiltered-audit/unfiltered_sensitivity.json`: verified `connections_princeton_no_threshold.csv.gz` (`275,679,780` bytes); escape (`316` nodes, `98.42%` known ACH sign) shifts from `17,448` edges / `40,633` synapses / `0` isolates at `min=1` to `1,513` edges / `12,135` synapses / `5` isolates at `min=5`; compass (`151` nodes, `97.35%` known ACH+GLUT sign) shifts from `8,275` edges / `81,615` synapses at `min=1` to `4,364` edges / `73,686` synapses / `0` isolates at `min=5`.
- `runs/compass-benchmark/summary.json`: FAFB `PEN_a` hemispheres induce opposite-signed `EPG` phase shifts (`+5.32°` left, `-4.78°` right); bump stability `0.7588 ± 0.0117` (`0` missing estimates); under `1–4 s` freeze + `1.4 rad` conflicting distractor at `t = 0.8 s`, `connectome_ring` achieved `16.27°` blackout error, `4.50°` reacquired error, and `5.78°` conflict error (vs. `24.90°` reacquired odometry, `11.42°` Bayesian conflict, and `22.06°` GRU conflict).
- `runs/scaled-dataset/manifest.json` + `VIGIL_NEURO_FINAL/kaggle-training/train_neuro_cover.py`: exported `840` world-split counterfactual samples (`train: 480`, `val: 180`, `test: 180`; `oracle_keys_in_inputs = 0`, `world_split_leakage = 0`) ready for Kaggle GPU training (`M3_GRU`, `M4_BioMasked`, `M5_Rewired`, `M8_NeuroCOVER_L`), plus ready-to-import `train_neuro_cover.ipynb`.

## Closed-loop controller, scientific validation, and research studio (`G08–G10`)

From D:\rover, run the `G08` closed-loop ablation (`recovery_off`, `ordinary_recovery`, `circuit_recovery`), `G09` paired bootstrap validation + BANC v888 transfer audit, and `G10` replay-verified interactive HTML studio:

```powershell
.\rover-python.ps1 VIGIL_NEURO_FINAL/closed_loop_benchmark.py --out runs/closed-loop-eval --seeds 4
.\rover-python.ps1 VIGIL_NEURO_FINAL/scientific_validation.py
.\rover-python.ps1 VIGIL_NEURO_FINAL/research_studio.py
```

Observed results (`46` passing unit tests + Kaggle GNU Octave reference parity `7.11e-15` in `runs/original-octave-v2/parity.json`):
- `runs/closed-loop-eval/summary.json`: `0` body collisions and `0` false emergency stops across `72` episodes (`24` per policy); `circuit_recovery` achieved `+0.0737 m` higher progress (`95% CI [+0.0294, +0.1238]`) over `recovery_off` with safe abstention preserved on blocked outages.
- `runs/scientific-validation/validation_report.json`: paired 95% bootstrap CIs (`2,000` resamples) across topology, recovery, and compass contrasts, with `BANC v888` cell-type homology mapped (`NOT_DOWNLOADED_LOCAL`).
- `runs/research-studio/index.html`: verified deterministic `replay.jsonl`/`frames.npz`/`summary.json` equivalence across `my-first-drive`, `rover-freeze`, and `rover-wall`.
