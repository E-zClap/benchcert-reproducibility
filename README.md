# Deployment-complete benchmarking

Code and data for *"Deployment-complete benchmarking"* by El Mustapha Mansouri and Keigo Arai, Institute of Science Tokyo.

## Overview

This repository provides the scripts, data pipelines and cached outputs to reproduce all experiments, figures and tables in the manuscript. The core idea is **deployment-complete benchmarking**: an audit of whether benchmark evidence determines the deployment action being claimed, which cases remain ambiguous, and what completion evidence would close the gap.

## Requirements

- Python ≥ 3.9
- See `requirements.txt` for dependencies

### Installation

```bash
# Clone the repository
git clone https://github.com/E-zClap/benchcert-reproducibility.git
cd benchcert-reproducibility

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

```
├── scripts/                      # All experiment scripts
│   ├── revision_robustness_experiments.py
│   ├── baseline_comparison_experiment.py
│   ├── decision_sufficiency_generalization.py
│   ├── nonlinear_linearization_ablation.py
│   ├── operational_decision_replay.py
│   ├── operational_consequence_analysis.py
│   ├── tox21_response_certification_audit.py
│   ├── matbench_discovery_certification_audit.py
│   ├── jarvis_response_certification_audit.py
│   ├── jarvis_probe_acquisition.py
│   ├── prospective_probe_acquisition_tox21.py
│   ├── vision_robustness_response_audit.py
│   ├── response_rank_ncs_demo.py
│   └── nmi_article_figures.py
├── outputs/                      # Cached numerical outputs
│   ├── *.csv                     # Summary tables and raw results
│   ├── *.pdf / *.png / *.svg     # Figures
│   └── ...
├── requirements.txt              # Python dependencies
├── LICENCE                       # MIT licence
└── README.md                     # This file
```

## Reproducing the Experiments

All scripts write outputs to the `outputs/` directory. Public datasets are downloaded automatically at runtime.

### Core experiments (main text)

```bash
# Controlled response-channel transfer and leaderboard inversion (Fig. 2)
python scripts/revision_robustness_experiments.py

# Baseline comparisons: conformal, OOD, ensemble, acquisition policies (Fig. 5)
python scripts/baseline_comparison_experiment.py

# Decision-sufficiency generalization with nonlinear constraints
python scripts/decision_sufficiency_generalization.py

# Nonlinear linearization ablation
python scripts/nonlinear_linearization_ablation.py
```

### Public benchmark audits (Fig. 3)

```bash
# Vision robustness audit
python scripts/vision_robustness_response_audit.py

# Tox21 molecular toxicity audit
python scripts/tox21_response_certification_audit.py

# Matbench Discovery stability audit (downloads ~200 MB of public predictions)
python scripts/matbench_discovery_certification_audit.py

# JARVIS cross-property audit
python scripts/jarvis_response_certification_audit.py
```

### Operational replays and acquisition (Figs. 4–5)

```bash
# Held-out operational decision replay (Tox21 + JARVIS)
python scripts/operational_decision_replay.py

# Tox21 probe acquisition (200 seeds, strict support)
python scripts/prospective_probe_acquisition_tox21.py --n-seeds 200 --min-support 50

# JARVIS probe acquisition (200 seeds)
python scripts/jarvis_probe_acquisition.py --n-seeds 200
```

### Generating manuscript figures

```bash
python scripts/nmi_article_figures.py
```

## Data Sources

All public datasets are downloaded or loaded at runtime:

| Dataset | Source | Use |
|---------|--------|-----|
| Handwritten digits | scikit-learn built-in | Vision robustness audit |
| Tox21 | MoleculeNet / DeepChem | Toxicity audit + probe acquisition |
| Matbench Discovery | Figshare + WBM | Materials stability audit |
| JARVIS-Leaderboard | GitHub (NIST) | Cross-property audit + acquisition |
| Spin-defect hosts | Zenodo (Toriyama et al.) | Host/substrate audit |

## Licence

MIT. See [LICENCE](LICENCE).

## Citation

```bibtex
@article{mansouri2026benchmark,
  author  = {Mansouri, El Mustapha and Arai, Keigo},
  title   = {Deployment-complete benchmarking},
  journal = {Submitted},
  year    = {2026}
}
```
