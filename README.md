# Auto-GlaKG

**Beyond Oracle Biomarkers: Auditing and Automating Knowledge Graph Reasoning for Explainable Glaucoma Diagnosis**

Auto-GlaKG extends [GlaKG](https://github.com/CH-YellowOrange/Glaucoma-Biomarker-Based-Knowledge-Graph) (IEEE BIBM 2026). GlaKG reasons over structural biomarkers that are supplied from annotation at inference time. This repository

1. **audits** that setting and shows that, on AI-generated biomarker annotations, knowledge graph (KG) reasoning is circular (the oracle KG reaches AUC 1.000), and
2. **removes** the annotation dependence: a label-blind biomarker network predicts the biomarkers from the fundus image, and only these predictions instantiate a parameter-free KG of clinical-prior rules.

```
fundus image ──► label-blind biomarker network ──► CDR + 6 signs ──► clinical-prior rules ──► signed chain score ──► p_KG + reasoning chain
                 (never sees the diagnosis)                          (soft activation)        (no trainable parameters)
```

## Highlights

| Method | AUC [95% CI] | BalAcc | Sens | Spec | MCC | ECE |
|---|---|---|---|---|---|---|
| ResNet50 (black box) | 0.951 [0.929, 0.970] | 0.907 | 0.920 | 0.893 | 0.810 | 0.058 |
| Oracle KG (annotation ceiling) | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 1.000 | 0.014 |
| Auto-KG (hard) | 0.938 [0.914, 0.960] | 0.888 | 0.937 | 0.840 | 0.786 | 0.075 |
| **Auto-KG (soft)** | **0.951 [0.931, 0.969]** | 0.874 | 0.970 | 0.779 | 0.782 | 0.090 |
| Auto-KG + Image | 0.946 [0.925, 0.967] | 0.901 | 0.944 | 0.859 | 0.811 | 0.044 |

Pooled out-of-fold predictions, nested group-aware 5-fold cross-validation, 689 images; CIs from 1,000 group-level bootstrap resamples.

- Auto-KG matches the black-box ResNet50 (DeLong *p* = 0.99) while explaining every prediction with an explicit rule chain, at 0.2 ms of extra cost per image.
- The oracle-to-automated gap is 0.049 AUC (*p* < 0.001). Replacing only the predicted CDR with its annotation restores the AUC to 0.995, so the **CDR is the main bottleneck**.
- Flipping the CDR-normal or ISNT rule raises the error rate from about 2% to 71% and 60%.
- Reasoning chains of misclassified images agree poorly with the oracle chains (Jaccard 0.13 versus 0.82 for correct predictions).

## Repository structure

```
.
├── data/                       # put the two parquet files here (not included)
├── data_preprocessing.ipynb    # data loading and annotation audit (CPU, about 1 to 2 min)
├── auto_glakg.ipynb            # main experiment, evaluation, figures, KG visualization
├── requirements.txt
└── README.md
```

## Data

The notebooks read the two parquet files of the GlaKG fundus dataset:

```
data/train-00000-of-00001.parquet   # 589 images
data/test-00000-of-00001.parquet    # 100 images
```

The 689 optic-disc-centred images (427 glaucoma, 262 normal) are pooled from ACRIMA, RIM-ONE, and Drishti-GS and come with AI-generated structured descriptions, from which seven biomarkers are parsed: cup-to-disc ratio (CDR), ISNT violation, rim thinning, notching, vessel bayoneting, rim pallor, and the laminar dot sign. The data are not redistributed here; please obtain them from the original sources and respect their licences.

## Installation

```bash
git clone <this-repo>
cd <repo-name>
pip install -r requirements.txt
```

Install PyTorch for your platform first if needed (see pytorch.org). The code selects the device automatically: CUDA, then Apple MPS, then CPU.

## Usage

Run the notebooks in JupyterLab, in this order.

**1. `data_preprocessing.ipynb`** (data loading and annotation audit). Run top to bottom. It quantifies how strongly the annotations encode the label: single-biomarker separability, annotation-only classifiers, biomarker correlations, text leakage, image-size shortcuts, and near-duplicate images across the official split.

**2. `auto_glakg.ipynb`** (main experiment). Run in three passes:

| Pass | Setting in Cell 2 | Purpose |
|---|---|---|
| 1 | `MOCK = True`, skip Cell 9 | pipeline check on synthetic data, no GPU, about 1 to 2 min |
| 2 | `MOCK = False`, `cls_epochs = 1`, `bm_epochs = 1` | smoke test on real data |
| 3 | `MOCK = False`, `cls_epochs = 15`, `bm_epochs = 15` | full experiment |

Use *Kernel → Restart & Run All* for passes 2 and 3. On macOS or Windows keep `num_workers = 0`.

To regenerate figures without retraining, keep the kernel alive and rerun only the cells from "论文用图" onwards.

## Outputs

Everything is written to `outputs_acrima/`. Tables are saved as CSV and LaTeX (booktabs); figures as vector PDF and 300 dpi PNG in `outputs_acrima/figures/`, one panel per file.

**Tables**

| File | Content |
|---|---|
| `table_main_folds` | main results, mean ± std over folds |
| `table_main_ci` | main results with 95% group-bootstrap CIs |
| `table_pooled_full_metrics` | AUC, AUPRC, Acc, BalAcc, Sens, Spec, Prec, F1, MCC, Brier, ECE, Sens@95Spec |
| `table_statistical_tests` | DeLong (AUC) and exact McNemar (accuracy), Holm-corrected |
| `table_biomarker_agreement` | predicted vs. annotated biomarkers |
| `table_oracle_substitution` | KG AUC when one biomarker is replaced by its annotation |
| `table_rule_flips` | rule flip rates and error rates with and without a flip |
| `table_ablation_*`, `table_sensitivity_*` | leave-one-rule-out, biomarker subsets, rule weighting, temperature, score scale |
| `table_explanation_agreement` | Jaccard, exact match, and top-1 agreement of reasoning chains |
| `table_subgroups` | official split and image-size subgroups |
| `table_robustness` | blur, noise, brightness, contrast, JPEG |
| `table_complexity` | parameters and inference time |
| `table_kg_statistics` | node and edge counts of the constructed KGs |
| `kg_nodes_{oracle,auto}.csv`, `kg_edges_{oracle,auto}.csv` | the constructed KGs, Neo4j-ready |
| `misclassified_cases.csv` | misclassified images with their reasoning chains |

**Figures**

| Files | Content |
|---|---|
| `fig01`, `fig11`, `fig12` | ROC, precision-recall, decision curve |
| `fig02*`, `fig03*` | per-fold metrics, fusion weights, confusion matrices |
| `fig04*` | biomarker agreement (CDR scatter, Bland-Altman, kappa, AUC) |
| `fig05*`, `fig06*` | oracle substitution, rule flips |
| `fig07`, `fig08*` | rule activation heatmap, chain score distributions |
| `fig09*` | calibration |
| `fig10*` | reasoning chain examples |
| `fig13*` | robustness per perturbation |
| `fig14*` | ablation and sensitivity |
| `fig15*`, `fig16` | explanation agreement, subgroup forest plot |
| `fig17*`, `fig18*` | Grad-CAM, misclassified cases |
| `fig20a` to `fig20f` | step-by-step construction of a reasoning subgraph |
| `fig21*`, `fig22*` | global KGs, KG statistics, rule co-activation networks |

## Method in brief

**Label-blind biomarker network.** A ResNet50 with seven outputs predicts the CDR (smooth L1 loss) and six signs (class-weighted BCE). The diagnosis label is never used to train it, so the KG sees the image only through the predicted biomarkers.

**Clinical-prior rules.** Ten rules, each with a condition, an evidence direction (glaucoma or normal), and a fixed clinical strength (3, 2, 1):

| Rule | Condition | Evidence | Weight |
|---|---|---|---|
| CDR high | CDR ≥ 0.7 | glaucoma | 3 |
| CDR moderate | 0.6 ≤ CDR < 0.7 | glaucoma | 2 |
| CDR normal | CDR ≤ 0.5 | normal | 3 |
| Rim thinning | present | glaucoma | 3 |
| Notching | present | glaucoma | 3 |
| ISNT violated | present | glaucoma | 2 |
| ISNT followed | absent | normal | 2 |
| Bayoneting | present | glaucoma | 2 |
| Rim pallor | present | glaucoma | 1 |
| Laminar dot sign | present | glaucoma | 1 |

**Soft activation.** CDR thresholds are relaxed as `a = sigmoid((CDR - tau) / T)` with `T = 0.03`; binary signs use the predicted probability. As `T → 0` this recovers the hard rules.

**Signed chain score.** `s = sum_r e_r * w_r * a_r = S_glaucoma - S_normal`, and `p_KG = sigmoid(s / 1.5)`. A glaucoma decision requires glaucoma evidence to outweigh normal evidence. The KG has no trainable parameters, and because the score is additive, each rule's contribution is an exact attribution of the decision.

**Evaluation protocol.** Images are grouped by perceptual hash so that near-duplicates never cross folds. Stratified, group-aware 5-fold cross-validation; within each training fold, 20% is held out to select the fusion weight, and both networks are trained on the remaining 80%. All results are on untouched test folds.

## Differences from the GlaKG (BIBM 2026) code

| GlaKG | Auto-GlaKG |
|---|---|
| annotated biomarkers at inference | biomarkers predicted from the image by a label-blind network |
| rules partly derived from dataset statistics | clinical-prior rules only |
| unsigned chain score with min-max normalization over the dataset | signed score with a fixed sigmoid mapping |
| scaler fitted and fusion weight chosen on the evaluation data | nested cross-validation; everything selected on inner validation splits |
| frozen ImageNet ResNet50 baseline | fine-tuned ResNet50 baseline |
| random node IDs, dense uninformative edges | deterministic node IDs, schema-level and weighted instance-level edges |
| no near-duplicate control | perceptual-hash grouping |

## Limitations

The biomarker targets are AI-generated and strongly tied to the label, so agreement with them is not agreement with clinical measurement. The dataset is small and pooled from several sources. Validation on data with independently measured biomarkers (for example, expert optic disc and cup masks) is ongoing.

## Citation

If you use this code, please cite GlaKG:

```bibtex
@inproceedings{glakg2026,
  title     = {GlaKG: A Biomarker-Centric Fundus Knowledge Graph for Explainable Glaucoma Diagnosis and Risk Assessment},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2026}
}
```

The Auto-GlaKG paper is under review; the citation will be added upon publication.

## License

Please add a license file (for example, MIT) before making the repository public.
