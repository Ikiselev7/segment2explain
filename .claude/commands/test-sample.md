Run a headless pipeline test using `run_samples.py`.

## Available samples
Look up `tests/fixtures/sample_cxr_vqa/samples.json` for the full list. Key samples:
- `cardiomegaly` — prompt: "Where is the Cardiomegaly?"
- `lung_opacity` — prompt: "Where is the Lung Opacity?"
- `pneumothorax` — prompt: "Where is the Pneumothorax?"
- `nodule_mass` — prompt: "Where is the Nodule/Mass?"
- `calcification` — prompt: "Where is the Calcification?"

## Steps

```bash
# Run a specific sample headlessly (no browser needed)
uv run python run_samples.py <sample_name>

# Run all samples
uv run python run_samples.py
```

If the user passes arguments, use the first argument as the sample name (e.g. `/test-sample nodule_mass`). Default to `nodule_mass` if no argument given.

Report the pipeline steps, status, and any errors from the output.
