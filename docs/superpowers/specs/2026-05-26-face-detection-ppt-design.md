# Face Detection Course PPT Design

## Goal

Create a visually polished course-design PPT for the Viola-Jones face detection project in this repository. The deck should maximize course-scoring potential while staying fully grounded in the existing codebase, report, models, evaluation outputs, and generated figures. It should feel like a strong academic project defense, not a generic engineering report.

## Source Boundaries

The PPT may only use content supported by the following project materials:

- `README.md`
- `报告.md`
- `src/`, `train.py`, `train_v6.py`, `evaluate.py`, `demo.py`, `audit_cascade.py`
- `models/*.json`
- `results/plots/*.png`
- `results/architecture/project_overview.png`
- evaluation outputs already produced in `results/eval_lfw_*`
- the course grading rubric PDF as presentation-orientation guidance

The deck must not invent:

- nonexistent experiments
- fabricated training curves
- unsupported performance claims
- fake screenshots or fake deployment scenes

Re-layout, redraw, summarize, and visually polish existing evidence is allowed.

## Presentation Positioning

- Primary mode: score-maximizing course project defense
- Tone: visual-first and presentation-friendly, but algorithmically clear
- Terminology: use `课程项目`, `系统实现`, `实验设计`, `结果分析`, `创新点`, `局限性`
- Avoid framing the work as a productized engineering system

## Audience And Use Case

- Primary audience: course instructors / defense teachers
- Secondary audience: classmates or teaching assistants in a project presentation setting
- Use case: final course project PPT for in-class report or defense

The deck should help the audience quickly answer:

1. What problem was solved?
2. What parts of Viola-Jones were implemented from scratch?
3. What training and optimization strategies were explored?
4. How good are the final results compared with OpenCV?
5. What are the project's real highlights and limitations?

## Recommended Deck Strategy

Use a results-driven algorithm explanation structure:

`Problem → Goal → Overall pipeline → Core algorithm explanation → Code/module realization → Data/training strategy → Results/comparisons → Highlights/challenges → Limitations → Conclusion`

This structure is chosen because it balances:

- visual clarity during oral presentation
- enough algorithm explanation for academic scoring
- enough result evidence for credibility
- enough reflective analysis for stronger defense impression

## Page Plan

Target page count: about 16 pages.

Recommended slide outline:

1. Cover
2. Background and task definition
3. Project goals and expected outputs
4. Overall system/pipeline overview
5. Viola-Jones core flow
6. Integral image and Haar-like features
7. AdaBoost and cascade classifier
8. Multi-scale sliding window and post-processing
9. Code structure and module mapping
10. Dataset construction and preprocessing
11. Training strategy evolution
12. Model progression and ablation
13. Benchmark results and visual cases
14. Project highlights and problem-solving
15. Limitations and future improvements
16. Conclusion and thanks

## Visual Direction

Overall direction:

- academic-defense style with modern information-design polish
- cleaner and lighter than a traditional template
- strong hierarchy, low clutter, limited but intentional accent colors

Design characteristics:

- 16:9 widescreen layout
- medium-density pages with several breathing moments
- strong use of diagrams, comparison cards, process visuals, and result panels
- large titles, compact supporting text
- few long paragraphs

Suggested palette direction:

- dark ink blue / slate for structure
- warm white background
- restrained teal or orange accents for emphasis

Suggested typography direction:

- readable Chinese-first academic sans/serif mix
- avoid default bland office feel
- prioritize clarity and PPT compatibility

## Content Emphasis

The presentation should emphasize these scoring-relevant strengths:

- complete from-scratch implementation of the classic Viola-Jones pipeline
- clear module decomposition in code
- systematic training strategy evolution
- meaningful parameter tuning and ablation
- strong comparison against OpenCV Haar baseline
- v6.1 reaching LFW F1 = 1.000 under the reported setup
- custom post-processing and smoothing ideas
- honest analysis of failure modes and limitations

## Asset Plan

Use the currently available generated visuals in the first PPT version:

- `results/plots/lfw_benchmark.png`
- `results/plots/model_progression_cards.png`
- `results/plots/lfw_failure_cases.png`
- `results/plots/augmentation_x7.png`
- `results/plots/top_haar_features.png`
- `results/architecture/project_overview.png`

These assets are acceptable as the initial PPT input, even if some need later refinement.

Follow-up visual iteration is expected:

- first build the PPT with current figures
- then selectively improve weak-looking figures after seeing them in context
- avoid blocking the initial deck generation on perfect chart aesthetics

## Accuracy Rules For PPT Writing

Every metric, model name, and claim should map back to existing project evidence.

Examples of supported claims:

- OpenCV Haar baseline LFW F1 around `0.963`
- self-implemented `v3`, `v6`, `v6.1` progression
- `v6.1` with `nb=10` and `CLAHE` reaching `Precision=1.000`, `Recall=1.000`, `F1=1.000` on the reported LFW test set
- training set scaling from `1000` to `3000` positives
- augmentation from `x4` to `x7`
- custom `_cluster_weighted_merge`
- `TrackingBoxSmoother`

Examples of unsupported content that must not appear unless new evidence is created:

- full training loss curves
- additional benchmark datasets beyond what the report actually supports for final claims
- speed numbers for configurations that were not measured
- claims of robustness to side-face or severe occlusion beyond the report's stated limits

## Execution Notes

The first deliverable should be a complete PPT draft generated with `ppt-master`, using existing repository assets and truthful textual summaries.

After the first draft exists, the next iteration should focus on:

1. replacing or retouching weak figures
2. tightening slide density
3. improving cover and section-divider aesthetics
4. checking whether any slide overstates the evidence

## Approval State

Current approved preferences from the user:

- choose the score-maximizing course-defense route
- prioritize visual presentation, while still explaining the algorithm clearly
- avoid "engineering project" wording
- use current generated images first, and refine them later if needed
