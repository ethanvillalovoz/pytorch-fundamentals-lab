# Figure contract: Iris reference evidence

## Communication job

This figure should allow a skeptical reader to inspect the complete deterministic Iris reference run because it shows the full committed training-loss trace, held-out confusion matrix, and both misclassified examples under the recorded seed and split.

## Supported claim

For seed `42`, the 243-parameter MLP was trained on 120 examples and correctly classified 28 of 30 held-out examples. The two errors are symmetric confusions between versicolor and virginica at prediction rows `19` and `25`.

## Evidence and selection

- `artifacts/iris-reference/metrics.json` supplies every plotted loss value, configuration field, accuracy, and confusion-matrix cell.
- `artifacts/iris-reference/predictions.csv` supplies the complete held-out decision list and exact error rows.
- No example is selected: the audit panel includes every committed error.

## Evidence boundary

- This is one fixed CPU run with one seed and no uncertainty estimate.
- The 30-example held-out set is a reproducibility fixture, not a benchmark.
- Training loss is not validation loss and does not establish generalization by itself.

## Delivery formats

- Generation source: `scripts/render_preview.py`
- README export: `docs/media/iris-reference-evidence.svg`
- Review export: `docs/media/iris-reference-evidence.png`
- Print/preflight export: `docs/media/iris-reference-evidence.pdf`
