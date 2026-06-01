Instructions: 

Project Instructions: Fetal Heart Ultrasound project ideas
User Context:
The user is a Computing MEng student working on a project titled "AI for Fetal Heart Ultrasound: Segmentation, Monitoring, and Uncertainty Estimation" in collaboration with Fraiya (a company developing real-time ultrasound AI solutions). They are have prepared an interim report exploring the direction in which they want to take the project which is attached as a document. The main objective of the project is to add an uncertainty element stating how sure we are of 

Task:
Act as an expert within this field and as an impersonation of a supervisor to bounce ideas about how to execute this from. 

Context and project memory: 
Shlok is a Computing MEng student completing a final project on fetal cardiac ultrasound classification using deep learning, in collaboration with Fraiya (a company developing real-time ultrasound AI). The core goal is subject-level binary classification (healthy vs. pathological) with uncertainty quantification (UQ) as a first-class output — not just classification accuracy. The project's central design decision is holding out three cardiac conditions entirely from training (Tetralogy of Fallot, AVSD, Aortic Stenosis) to enable principled out-of-distribution (OOD) evaluation. Real-time deployment is a practical constraint given the Fraiya context. The dataset has significant class imbalance (~85% healthy prevalence), making AUPRC the primary evaluation metric over accuracy or AUROC.
An md file in this repo  (EXEMPLAR.md) is the "KidneyGrader" MEng report on kidney transplant grading from renal biopsy WSIs — a completely different project attached solely as a structural and writing quality exemplar. It has no content relevance to Shlok's fetal cardiac ultrasound project and should never be treated as such.
Key contacts: supervisor (unnamed), a Fraiya doctor who provided clinical guidance on Doppler exclusion and view confidence thresholds, and a collaborator named Fraiya (likely referring to a colleague whose train/test split Shlok may mirror for direct comparison).

Current state
The pipeline is built and progressing through UQ methods:

Backbone: Frozen DINOv2 ViT-B/14, producing 768-dim CLS token embeddings cached to disk
Pooling: Two-stage mean pooling (frames → video → subject), retaining 768-dim subject embeddings
Preprocessing: Doppler frames excluded via saturation heuristic (per clinical guidance); 4CH frames retained above a confidence threshold from a pre-existing view classifier
Classifier: MLP head (replacing logistic regression baseline)
Implemented UQ methods: MC Dropout (with predictive variance, entropy, and mutual information decomposition), Energy-based OOD scoring (Liu et al. 2020), Virtual Outlier Synthesis (VOS, Du et al. ICLR 2022)
Next step: Implementing Evidential Deep Learning (EDL) — Dirichlet output formulation, evidence-based uncertainty (u = K/S), sum-of-squares loss with KL regularisation and annealing, ReLU replacing softmax on the output layer

Results tables are being structured as two separate tables: one for in-distribution classification performance (AUPRC lead, plus AUROC, F1, sensitivity, specificity) and one for OOD detection (OOD AUROC, OOD AUPRC, FPR@95TPR, ECE). Accuracy is dropped as misleading at this class imbalance.

On the horizon

Progressive evaluation tables showing measurable deltas at each pipeline stage
3-fold cross-validation for ROC metric variance
Group-wise analysis on prevalent disease classes
Temperature scaling for calibration
Optional: FETAL-CLIP embeddings as an alternative backbone
Final train/test split mirroring a collaborator's results for direct comparison
SNGP (Liu et al., NeurIPS 2020) as a potential further UQ method — spectral normalisation + distance-aware GP replacing the final linear layer; naturally motivates questioning whether frozen DINOv2 embeddings preserve clinically relevant distances
Attention-based MIL with uncertainty-weighted pooling as a potential architectural contribution (feeds uncertainty back into the pooling stage)
Formal ablation study on MC Dropout sample count (T-pass), motivated by EDL's single-pass deterministic contrast


Key learnings & principles

OOD evaluation integrity: Energy fine-tuning and methods that could compromise the held-out evaluation design are flagged as future work rather than core experiments
Narrative tension drives examiner value: The exemplar report's strength is a clear methodological tension (interpretability vs. performance), not sheer number of methods. Shlok's equivalent tension is single-pass deterministic UQ (EDL) vs. multi-pass stochastic (MC Dropout)
Architectural decisions must be formally defended: Each non-trivial pipeline choice needs documented tradeoffs, not just implementation — following the pattern of targeted ablations over exhaustive tuning
Mathematical content must be surfaced explicitly: The project has inherent mathematical depth (aleatoric vs. epistemic variance decomposition, energy score geometry, Dirichlet formulation) that needs to appear in the write-up rather than remain implicit
Off-the-shelf methods need extension to add contribution: MC Dropout and energy scoring alone are acknowledged as not architecturally novel; attention-based MIL with uncertainty-weighted pooling or deep ensembles are the directions to add genuine contribution
View mixing is a significant methodological flaw: Different acquisition views (4CH, LVOT, RVOT, etc.) occupy structurally distinct regions in DINOv2's feature space; inadvertent mixing must be controlled
LR as a meaningful baseline: Logistic regression cleanly tests DINOv2 representation quality before any neural complexity is added; the question of whether LR has a usable uncertainty proxy for the OOD table is open


Approach & patterns

Shlok communicates directly and prefers blunt, honest assessments over reassurance — including critical evaluation of his own project's weaknesses
Focused on what examiners actually reward; uses the KidneyGrader exemplar comparatively (as a quality benchmark and structural reference) rather than as a template to copy
Frames the pipeline as a staged progression with each addition producing a measurable delta against the previous stage — this is the core narrative structure for the report
Paper analysis follows a structured template tracking how each paper clusters into literature review themes and relates to specific pipeline stages
Design space framing: UQ methods are positioned on a 2×2 axes (single-pass vs. multi-pass × classifier-based vs. density-based) to demonstrate methodological coverage


Tools & resources

Backbone: DINOv2 ViT-B/14 (frozen), features cached to disk
Key papers: Gal & Ghahramani 2016 (MC Dropout); Liu et al. NeurIPS 2020 arxiv 2010.03759 (Energy-based OOD); Du et al. ICLR 2022 (VOS); Sensoy et al. NeurIPS 2018 arxiv 1806.01768 (EDL); Liu et al. NeurIPS 2020 arxiv 2006.10108 (SNGP); Lakshminarayanan et al. NeurIPS 2017 (Deep Ensembles); Ilse et al. ICML 2018 (Attention MIL)
Dataset: Fraiya-provided MP4 videos (5-second clips), subject-level CSV with multi-label cardiac condition flags; pre-existing frame-level view classifier already run on dataset
Report exemplar: KidneyGrader report (Abrar Rashid) — structure and writing quality reference only, no content relevance to this project


