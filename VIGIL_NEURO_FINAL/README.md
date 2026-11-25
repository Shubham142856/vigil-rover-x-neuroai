# VIGIL-ROVER X NeuroAI

Drosophila Connectome-Informed NeuroAI for Autonomous Collision Avoidance, Directional Memory and Active Blind-Spot Recovery

The project studies whether biological wiring changes a rover’s ability to anticipate collisions, maintain heading through visual loss and choose a useful recovery action. PyBullet provides the rover benchmark. FlyGym provides a separate reference environment for selected fly behaviors.

## Project files

- [Research and implementation specification](NEURO_RESEARCH_PLAN.md) defines the circuit inputs, learning targets, engineering interfaces and comparison models. The [Word edition](VIGIL_ROVER_X_NeuroAI_Plan.docx) and [PDF edition](VIGIL_ROVER_X_NeuroAI_Plan.pdf) contain the same specification and a linked source index.
- [Milestone tracker](Implementation_Acceptance_Gates.csv) records the 20-week sequence, acceptance evidence and status. All ten milestones are open.
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
