# Data

Two releases are separate.

The public archive contains rover RINEX, broadcast navigation, the ground truth read by the experiment, Jev 1.14.0 labels, the teacher question and stored responses, and the trained checkpoints for the student, Soft Jev, Hard Jev, Null prior, and 9-input TDL.

This code package starts from the processed tables below. It does not read the raw RINEX files.

## Routes

Only these four routes are accepted:

- `klt-20210610`
- `whampoa-20210714`
- `science-museum-wongkun-20260630`
- `reclamation-street-20260630`

The processed bundle has 215,421 observations, 14,523 candidate epochs, reference positions, the physical initial solution, the five-level teacher labels, reference relative-risk targets, and the fixed folds. The stable identifiers are `sample_id` and `epoch_id`. The loader rejects duplicate identifiers, a fifth route, and a change in row count.

## Processed files

| File | Role |
|---|---|
| `observations.csv.gz` | Online features, satellite positions, corrected pseudorange, physical variance, and teacher labels |
| `epochs.csv.gz` | Candidate epochs, initial position, and online state |
| `reference_positions.csv.gz` | Reference positions on the common GPST timeline |
| `physical_predictions.csv.gz` | Physical WLS initialization and failure flags |
| `reference_risk_targets.csv.gz` | Reference relative-pseudorange risk proxy |
| `outer_folds.csv` / `inner_folds.csv` | Route-isolated nested splits |

## Sources

KLT comes from KLTDataset and keeps that project's license and citation. Cite Hu, Wen, and Hsu, ITSC 2023. Whampoa comes from UrbanNav and keeps that project's license and citation. Cite Hsu et al., NAVIGATION 2023. Science Museum and Reclamation St were collected by the authors. The code license does not replace the dataset licenses.
