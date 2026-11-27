# VIGIL-ROVER X NeuroAI — Master Project Record (Work, Methods, Tools, Skills & Plugins)

**Repository:** [`Shubham142856/vigil-rover-x-neuroai`](https://github.com/Shubham142856/vigil-rover-x-neuroai) (`main`)  
**Workspace:** `D:\rover`  
**Verified Regression Suite:** `118/118` tests passing (`.\rover-python.ps1 -m unittest discover -s VIGIL_NEURO_FINAL -p "test_*.py" -v`, `387.162s`)  
**Current Gate Status:** `P6.9.1 – P6.9.7` completed at commit `a36ebdf`; **`P7` bulk scaling (`5,000` episodes) is held** pending a targeted safe-approach negative calibration pass (`P6.9.8`) to clear `G08` / `Untouched` early-standoff stalls.

---

## 1. Complete Chronological Record of Work Done (`9b29a06` -> `a36ebdf`)

### Stage A — Foundation, Biological Benchmarks & Studio (`G01–G10`, Commits `9b29a06` -> `ae20be5`)
| Gate | Scope | Commit | Key Implemented Files & Verified Results |
|---|---|---|---|
| **G01** | Historical 2D baseline audit | `9b29a06` | `Planned / Unavailable`: earlier 2D source revision could not be recovered (`references/` holds archival artifacts). Documented transparently rather than fabricating baselines. |
| **G02** | FAFB v783 unfiltered connectome audit | `b2d6c0d` | [audit_unfiltered_connectome.py](file:///d:/rover/VIGIL_NEURO_FINAL/audit_unfiltered_connectome.py): audited `connections_princeton_no_threshold.csv.gz`. Escape graph (`316` nodes, `98.42%` known `ACH` sign) shifts `17,448 -> 1,513` edges (`min=1` vs `5`); compass graph (`151` nodes, `97.35%` known `ACH+GLUT` sign) shifts `8,275 -> 4,364` edges. |
| **G03** | PyBullet rover & tactile recovery | `4d8380c` | [rover_sim.py](file:///d:/rover/VIGIL_NEURO_FINAL/rover_sim.py), [recovery_policy.py](file:///d:/rover/VIGIL_NEURO_FINAL/recovery_policy.py): rigid chassis + lateral feelers (`y = ±0.07`), bilateral stop, verified-clear vs blocked outage safe abstention (`0` body collisions). |
| **G04** | Biological looming vision (`LPLC2/LC4`) | `4d8380c`, `ae20be5` | [vision_benchmark.py](file:///d:/rover/VIGIL_NEURO_FINAL/vision_benchmark.py), [camera_escape.py](file:///d:/rover/VIGIL_NEURO_FINAL/camera_escape.py): multiplicative 4-quadrant LPLC2 (`1.0` on approach, `0.0` on recede/yaw/near-miss/empty) matching Kaggle GNU Octave reference within **`7.11e-15`** (`runs/original-octave-v2/parity.json`). |
| **G05** | Connectome compass (`EPG/PEG/PEN/Delta7`) | `b2d6c0d` | [compass_benchmark.py](file:///d:/rover/VIGIL_NEURO_FINAL/compass_benchmark.py): FAFB `PEN_a` asymmetry (`+5.32°` left / `-4.78°` right); `5.78°` conflict error & `4.50°` reacquired error under `1–4s` blackout + `1.4 rad` distractor (beating `11.42°` Bayesian & `22.06°` GRU conflict). |
| **G06** | Blind-spot belief & exact branch rollback | `4d8380c` | [blindspot_dataset.py](file:///d:/rover/VIGIL_NEURO_FINAL/blindspot_dataset.py), [branch_rollout.py](file:///d:/rover/VIGIL_NEURO_FINAL/branch_rollout.py): exact PyBullet/camera state save+restore (`max_pre_branch_pixel_diff = 0`, `0` oracle/world leaks). |
| **G07** | Scaled dataset & initial Kaggle GPU training | `b2d6c0d`, `e1db28f` | [scale_dataset.py](file:///d:/rover/VIGIL_NEURO_FINAL/scale_dataset.py), [train_neuro_cover.py](file:///d:/rover/VIGIL_NEURO_FINAL/kaggle-training/train_neuro_cover.py): `840` world-split samples trained on Kaggle Tesla T4 (`M4_BioMasked`, `M5_Rewired`, `M3_GRU`, `M8_NeuroCOVER_L`). |
| **G08–G10** | Closed-loop, bootstrap validation & Studio | `ae20be5` | [closed_loop_benchmark.py](file:///d:/rover/VIGIL_NEURO_FINAL/closed_loop_benchmark.py), [scientific_validation.py](file:///d:/rover/VIGIL_NEURO_FINAL/scientific_validation.py), [research_studio.py](file:///d:/rover/VIGIL_NEURO_FINAL/research_studio.py): `0` collisions on G08, `BANC v888` homology mapped (`NOT_DOWNLOADED_LOCAL`), replay-verified HTML Studio (`runs/research-studio/index.html`). |

### Stage B — Reality-Grounded Sim-to-Real Environment (`R01–R06`, Commits `9499275` -> `ac7b42c`)
| Gate | Scope | Commit | Key Implemented Files & Verified Results |
|---|---|---|---|
| **R01–R03** | Reality-grounded physics, sensors & `HardwareInterface` | `9499275` | [real_world_env.py](file:///d:/rover/VIGIL_NEURO_FINAL/real_world_env.py) (`±3 mm` heightfield, friction, PWM lag, light flicker), [real_sensor_model.py](file:///d:/rover/VIGIL_NEURO_FINAL/real_sensor_model.py) (OV2640 barrel distortion, rolling shutter, MPU-6050 IMU, 360-CPR encoder, whisker debounce), [hardware_interface.py](file:///d:/rover/VIGIL_NEURO_FINAL/hardware_interface.py) (`SimInterface` live; `ROSInterface` & `SerialInterface` `R07` stubs). |
| **R04–R06** | Realistic scenarios, Kaggle T4 retraining & leaky consensus | `a2edaf9` -> `c8f18d1` | Added 6 realistic scenarios (`corridor_pedestrian`, `warehouse_forklift_cross`, `outdoor_pothole`, `lab_glass_wall`, `ramp_descent`, `night_mode`). Retrained on `1,740` G07+R05 samples (`e65f848`: `M8_NeuroCOVER_L` `test_brier = 0.01657`). Closed-loop `r05` hit `0` collisions yet **`-31.2%` progress vs `recovery_off`** (`0.2477 m` vs `0.3600 m`), triggering the V2 reconstruction audit (`P0`). |
| **Latency** | Real-time CPU budget verification | `0923824`, `ac7b42c` | [test_latency_budget.py](file:///d:/rover/VIGIL_NEURO_FINAL/test_latency_budget.py): full Layer-3 stack runs in **`2.89 ms` mean / `4.09 ms` p95** on CPU (`< 100 ms` `10 Hz` budget). `68/68` unit tests passing. |

### Stage C — First-Principles Scientific & Engineering Reconstruction (`P0 – P6.9`, Commits `e68131b` -> `a36ebdf`)
| Phase | Scope | Commit(s) | Key Fixes, Discoveries & Verified Outcomes |
|---|---|---|---|
| **P0–P2** | Baseline freeze, parity & root-cause audit | *(read-only audit)* | Proved 6 defects causing the R05 prediction-to-control gap: (C1) `y_recover=1` assigned during camera blackouts (`29%` train, `73.4%` val); (C2) red-only pixel thresholds & static size drive; (C3) IMU/encoder missing from `obs` (`compass.step(0.0, cue)`); (C4) `98.6 px` vs `73.914 px` focal length mismatch; (C5) discrete compass shift vs continuous dynamics; (C6) red-only `SectorBelief` & ambiguous `safe_success`. Corrected `M8` vs `M3_GRU` Brier claim from `49%` to exact **`39.12%`**. |
| **P3** | Intrinsics calibration, IMU/encoder wiring & T4/T5 vision | `e68131b` | Fixed `73.914 px` (`66°` V-FOV) parity; attached `imu_gyro_z`, `imu_gyro_y`, `encoder_speed_m_s` to `obs`; implemented [t4_t5_vision.py](file:///d:/rover/VIGIL_NEURO_FINAL/t4_t5_vision.py) (dual-pathway luminance EMD + IMU ego-motion subtraction). **`73/73`** tests passing. |
| **P4** | Factorized belief & signed counterfactual $\Delta U_t(a)$ | `0f51696`, `7a0a9e2` | Implemented [spatial_belief_v2.py](file:///d:/rover/VIGIL_NEURO_FINAL/spatial_belief_v2.py) (360° world-referenced 15-sector Bayes occupancy, entropy $U_{\text{task}}$) and brake-subtracted signed recoverability $\Delta U_t(a) = U_{\text{task}}(b_{t+H}^{\text{brake}}) - U_{\text{task}}(b_{t+H}^{a})$ preserving negative worsening effects. **`77/77`** tests passing. |
| **P5** | 6-layer protection, disaggregated metrics & causal ablation | `253ba77` | Implemented [safety_arbitrator_v2.py](file:///d:/rover/VIGIL_NEURO_FINAL/safety_arbitrator_v2.py), [evaluation_metrics_v2.py](file:///d:/rover/VIGIL_NEURO_FINAL/evaluation_metrics_v2.py), [p5_causal_ablation.py](file:///d:/rover/VIGIL_NEURO_FINAL/p5_causal_ablation.py) (7-way `A–G`, 336 episodes). Visual fusion (`C`) jumped R05 progress **`+172.9%`** (`0.240 -> 0.655 m`) but caused `6` collisions (also logged `1` G08 collision in [ablation_report.json](file:///d:/rover/runs/p5-causal-ablation/ablation_report.json)); Full `P5` (`G`) hit `0` collisions but stalled at `0%` G08 goals because old `2.4 s` weights predicted `r = 0.98` at `0.126 m`. **`96/96`** tests passing. |
| **P6.1–P6.8** | Multi-horizon survival, ground IPM & pilot gate | `8b45247` | Built [ground_range_estimator.py](file:///d:/rover/VIGIL_NEURO_FINAL/ground_range_estimator.py), [pilot_dataset_v3.py](file:///d:/rover/VIGIL_NEURO_FINAL/pilot_dataset_v3.py) (`19` worlds / `285` branches), [pilot_train_v3.py](file:///d:/rover/VIGIL_NEURO_FINAL/pilot_train_v3.py), and [p6_closed_loop_pilot.py](file:///d:/rover/VIGIL_NEURO_FINAL/p6_closed_loop_pilot.py). Fixed `fov=66°` belief window, eliminated permanent stalls (`0.00` in G08 & R05), and proved `Single24s` horizon mismatch (`0.06875` `0.3s` Brier error). Flagged heading-uncertainty `slow` drift & G08/R05 collisions (`4/24`, `2/24`). **`104/104`** tests passing (`P7` held). |
| **P6.9.1–P6.9.7** | Safety, uncertainty & distribution reconstruction | `a991d2e` -> `095ac03` -> `a36ebdf` | **1. Audit & CI Fix:** Proved `pilot-v3-p6` had `1/165` (`0.61%`) protected positives (`0` val/test) and old weights had `68/68` dangerous FNs (`BSS = -0.301`) on realistic hazards; fixed two-sided Clopper-Pearson CIs.<br>**2. Modal Heading & Bounded Exploration:** Added exact von Mises $(\mu_t,\kappa_t)$ fusion with `>35°` cue-conflict mixture contraction, downward-contiguous `[42:86]` IPM, stale-outage invalidation, 3-sector `slow` checks, and `BoundedExplorationMonitor`.<br>**3. 110-World Targeted Dataset & Time-Leak Fix:** Built `runs/pilot-v3-p69` (`1,419` train / `255` val / `255` untouched-family test, `0` leaks). Discovered and masked non-Markovian `obs["time_s"]` in slot `14` (`095ac03`), restoring `4/4` `open_healthy` goals (`1.324 m`).<br>**4. Kaggle GPU v3 & Closed-Loop Gate:** Trained 9 diagnostic + 8 multi-seed architectures on CUDA (`a36ebdf`). **`P69_Full_Bio_Modular`** resolved risk/utility gradient conflict (`cos: -0.203 -> +0.116`), halved `delta_u_mae` (`0.1047 -> 0.0564`), and was the **only learned policy with `0/24` collisions in `R05`** (`50.0%` goal rate, `0` stalls); however `G08` (`16.7%`) and `Untouched` (`0.0%`) still suffer early-standoff stalling. **`118/118`** tests passing. |

---

## 2. Scientific & Engineering Methods Used

| Method Area | Core Mathematical / Algorithmic Formulation | Primary Modules |
|---|---|---|
| **Connectome Constraint & NT Signs** | Directed FAFB v783 synapse graphs (`min=1` sensitivity audited vs `5`), `ACH` (`+1`) / `GLUT` (`-1`) sign masks, continuous leaky-Euler integration $\tau_i \dot{z}_i = -z_i + \phi(W z + U s + b)$ with adaptive learnable $\tau_i$. | [extract_graph.py](file:///d:/rover/VIGIL_NEURO_FINAL/extract_graph.py), [p69_bio_diagnostics.py](file:///d:/rover/VIGIL_NEURO_FINAL/p69_bio_diagnostics.py) |
| **Ego-Compensated T4/T5 Vision** | Calibrated `73.914 px` intrinsics (`66°` V-FOV), dual-pathway luminance EMD, spherical optical-flow subtraction using MPU-6050 `imu_gyro_z` / `imu_gyro_y`, collision-cone radial vs lateral discrimination. | [t4_t5_vision.py](file:///d:/rover/VIGIL_NEURO_FINAL/t4_t5_vision.py), [real_sensor_model.py](file:///d:/rover/VIGIL_NEURO_FINAL/real_sensor_model.py) |
| **Modal Circular & Spatial Belief** | Exact von Mises $(\mu_t, \kappa_t)$ circular posterior via Bessel ratio $A(\kappa)=I_1(\kappa)/I_0(\kappa)$, Gaussian-step dispersion, circular mixture contraction on `>35°` visual/IMU conflict; 360° world-referenced 15-sector Bayes grid with age decay and Shannon entropy $U_{\text{task}}$. | [spatial_belief_v2.py](file:///d:/rover/VIGIL_NEURO_FINAL/spatial_belief_v2.py) |
| **Counterfactual Info Recoverability** | Exact PyBullet + RNG branch save/restore; signed brake-subtracted counterfactual gain $\Delta U_t(a) = U_{\text{task}}(b_{t+H}^{\text{brake}}) - U_{\text{task}}(b_{t+H}^{a})$ distinguishing informative exploration (`>0`), neutral (`0`), and ambiguity-worsening maneuvers (`<0`). | [branch_rollout.py](file:///d:/rover/VIGIL_NEURO_FINAL/branch_rollout.py), [pilot_dataset_v3.py](file:///d:/rover/VIGIL_NEURO_FINAL/pilot_dataset_v3.py) |
| **Monocular IPM & 6-Layer Safety** | Camera-extrinsic downward-contiguous `[42:86]` ground-contact range estimator + empirical stopping-distance curves $d_{\text{stop}}(v, \mu)$; tactile/IPM/dynamic arbitration + `BoundedExplorationMonitor` with uncertainty-penalized gain $\widehat{G} - 0.50\sigma_G$. | [ground_range_estimator.py](file:///d:/rover/VIGIL_NEURO_FINAL/ground_range_estimator.py), [safety_arbitrator_v2.py](file:///d:/rover/VIGIL_NEURO_FINAL/safety_arbitrator_v2.py) |
| **Multi-Horizon Survival & Modularity** | Discrete hazards $h_k$ across $\{0.10, 0.30, 0.60, 1.20, 2.40\}\text{ s}$ enforcing monotonic cumulative survival $S_k = \prod_{j \le k}(1 - h_j)$, separate protected `0.30s+brake` head, and decoupled risk vs utility readout cores to eliminate multi-task gradient conflict. | [p69_bio_diagnostics.py](file:///d:/rover/VIGIL_NEURO_FINAL/p69_bio_diagnostics.py), [train_p69_gpu.py](file:///d:/rover/VIGIL_NEURO_FINAL/kaggle-p69-pilot/train_p69_gpu.py) |
| **Rigorous Statistical Verification** | Zero world/oracle leakage invariants, train-only hard-example mining, isolated untouched test world families, BSS/PR-AUC/ECE/dangerous-FN auditing, per-suite exact two-sided Clopper-Pearson 95% CIs + world-clustered SEs, BPTT finite-difference checks, and `< 1e-6` PyTorch->NumPy parity. | [evaluation_metrics_v2.py](file:///d:/rover/VIGIL_NEURO_FINAL/evaluation_metrics_v2.py), [p69_dataset_audit.py](file:///d:/rover/VIGIL_NEURO_FINAL/p69_dataset_audit.py), [p69_matched_eval.py](file:///d:/rover/VIGIL_NEURO_FINAL/p69_matched_eval.py) |

---

## 3. Software Tools, Platforms & Runtimes Used

| Category | Tool / Runtime | Role in Project |
|---|---|---|
| **Local Python Env** | `D:\rover\.tools\rover-env\python.exe` (via `.\rover-python.ps1`) | PyBullet physics simulation, NumPy & SciPy dataset generation, closed-loop evaluation, unit test runner (`unittest`), and static HTML Research Studio generation. |
| **Cloud GPU Compute** | **Kaggle CLI** (`.\.venv\Scripts\kaggle.exe` v2.2.5) + **Kaggle Tesla T4 CUDA PyTorch** | Offloaded multi-seed, multi-architecture PyTorch GPU training (`vigil-rover-x-neuro-cover-l-training`, `vigil-rover-x-p6-9-pilot-gpu-diagnostics` v1/v3, GNU Octave LPLC2 parity reference). |
| **Version Control** | **Git & GitHub** (`Shubham142856/vigil-rover-x-neuroai`) | Reproducible pinned commits (`repo_commit` logged inside Kaggle JSONs and checkpoints), branch/commit hygiene (`9b29a06` through `a36ebdf`). |
| **Browser & Web Inspection** | **Chrome DevTools Subagents** (`browser`: *Kaggle Job Status Inspector*, *Kaggle Notebook Results Inspector*, *Kaggle Queue & Session Inspector*) & **Tavily MCP** (`tavily_extract`) | Read Kaggle notebook pages/outputs when diagnosing interactive runs (`notebook76b77cb2ff`, `vigil-rover-x-lplc2-stimulus-comparison`, `vigil-rover-x-p6-9-pilot-gpu-diagnostics`). |
| **Agent Workflow Tools** | Antigravity workspace tools (`run_command`, `read_file`, `write_file`, `edit_file`, `schedule`, `manage_task`, `update_steps`, `update_focus`, `search_marketplace`, `invoke_subagent`) | Asynchronous polling/scheduling of long Kaggle GPU & unit test jobs without CPU spin loops, live step checklists, and parallel codebase edits/verifications. |

---

## 4. Skills & Plugins (Requested, Installed, Read & Environment Inventory)

### A. User Requests & Direct Actions Taken
1. **Robotics Plugin Installation (`2026-11-26 09:53Z`):** After a Marketplace lookup (`search_marketplace(queries=["robotics", "simulation", "3d"])`), the **`Robotics`** plugin (`antigravity-plugins-official/plugin/robotics`) was surfaced and **installed by you**.  
   *Honest note:* Its bundled skills target MuJoCo, Blender, and SO-101 / I2RT YAM / FR3 Duo manipulation arms (`sim2real-gauntlet`, `sim2real-iteration`, `motion-planning`, `iteration`, `mujoco-playground`, etc.), whereas VIGIL-ROVER X uses PyBullet + NumPy + PyTorch; hence PyBullet-native harnesses (`p5_causal_ablation.py`, `p6_closed_loop_pilot.py`, `p69_matched_eval.py`) were used instead of MuJoCo skills.
2. **Explicit Instruction to Use Best Skills, Tools & Plugins (`2026-11-26 11:12Z`):** You instructed *"go ahead use best skills and tech tool neded and pugins"*.
3. **Marketplace Query for Version Control (`2026-11-26`):** `search_marketplace(queries=["github"])` was executed while setting up the `Shubham142856/vigil-rover-x-neuroai` remote (handled cleanly via local `git` CLI).

### B. Skills & MCP Tools Actively Loaded / Invoked
| Item | Parent Plugin / Server | Where Loaded / Used | How Applied |
|---|---|---|---|
| **`ml-best-practices`** | `dak` (`C:\Users\HP\.gemini\config\plugins\dak\skills\ml-best-practices\SKILL.md`) | Actively read (`read_file`) after your `11:12Z` instruction | Guided classification/regression calibration rigor, constant-prevalence baselines, BSS/PR-AUC/ECE/FN comparisons, statistical CIs, and non-leaky splits (`R06`, `P5`, `P6`, `P6.9`). |
| **`tavily_extract`** | `tavily` MCP Server | Actively called (`call_mcp_tool`) once (`2026-11-26 12:20Z`) | Attempted content extraction from Kaggle notebook `notebook76b77cb2ff` alongside Chrome DevTools inspection. |
| **`browser` Subagent** | `chrome-devtools-plugin` / Antigravity `browser` | Invoked 3 times (`Kaggle Job Status Inspector`, `Kaggle Notebook Results Inspector`, `Kaggle Queue & Session Inspector`) | Visual verification of Kaggle interactive notebook outputs & queue status (3rd invocation noted local Chrome DevTools port wasn't attached, so CLI polling resumed). |

### C. Full Inventory of Installed Plugins in Local Environment (`C:\Users\HP\.gemini\config\plugins\`)
- **Installed / Relevant to Session:** `robotics` *(installed by user mid-session)*, `dak` *(`ml-best-practices` used)*, `chrome-devtools-plugin` *(used via `browser` subagent)*.
- **Installed / Unrelated Domain Plugins (Not Invoked):** `chemistry`, `genomics`, `scientific-literature`, `structural-biology`, `science-overview`, `firebase`, `google-antigravity-sdk`, `modern-web-guidance-plugin`.

---

## 5. Index of Phase Reports & Evidence Artifacts

- **Phase P6.9 Gate Report (Latest):** [p69_gate_report.md](file:///C:/Users/HP/.gemini/antigravity/brain/a98603b2-8e99-4420-9e02-0a4de6556ed6/p69_gate_report.md)
- **Phase P6 Pilot Gate Report:** [p6_pilot_gate_report.md](file:///C:/Users/HP/.gemini/antigravity/brain/a98603b2-8e99-4420-9e02-0a4de6556ed6/p6_pilot_gate_report.md)
- **Phase P5 Causal Ablation Report:** [p5_causal_ablation_report.md](file:///C:/Users/HP/.gemini/antigravity/brain/a98603b2-8e99-4420-9e02-0a4de6556ed6/p5_causal_ablation_report.md)
- **Phase P0–P2 Root-Cause Audit & V2 Plan:** [p1_root_cause_audit_and_v2_plan.md](file:///C:/Users/HP/.gemini/antigravity/brain/a98603b2-8e99-4420-9e02-0a4de6556ed6/p1_root_cause_audit_and_v2_plan.md)
- **Stage R01–R06 Progress Report & Future Roadmap:** [vigil_rover_progress_report.md](file:///C:/Users/HP/.gemini/antigravity/brain/a98603b2-8e99-4420-9e02-0a4de6556ed6/vigil_rover_progress_report.md)
- **Reality-Grounded Simulation Plan (`R01–R07`):** [reality_sim_plan.md](file:///C:/Users/HP/.gemini/antigravity/brain/a98603b2-8e99-4420-9e02-0a4de6556ed6/reality_sim_plan.md)
- **Latest Machine-Readable Evidence:** [p69_dataset_audit.json](file:///d:/rover/runs/p69-audit/p69_dataset_audit.json), [bio_diagnostics.json](file:///d:/rover/runs/p69-audit/bio_diagnostics.json), [p69_offline_metrics.json](file:///d:/rover/runs/p69-models/p69_offline_metrics.json), [p69_closed_loop_report.json](file:///d:/rover/runs/p69-closed-loop/p69_closed_loop_report.json)
