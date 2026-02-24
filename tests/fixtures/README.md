# Test Fixtures

This directory contains test data and fixtures for the segment2explain-poc test suite.

## VinDr-CXR-VQA Sample Data

The `sample_cxr_vqa/` directory will contain representative examples from the **VinDr-CXR-VQA dataset** for testing and demonstration purposes.

### Dataset Information

**Source:** VinDr-CXR-VQA: A Visual Question Answering Dataset for Explainable Chest X-Ray Analysis
**Paper:** https://arxiv.org/html/2511.00504v2
**Access:** Public release via Hugging Face

**Dataset Stats:**
- 4,394 CXRs
- 17,597 QA pairs
- Structured questions: "Where is X?", "What is abnormal?", "Is there...?", "How many...?"
- Radiologist-verified bounding boxes
- Clinical reasoning explanations

### Sample Cases to Include

Select 5-10 representative cases covering:

1. **Single finding with clear bbox** (`case_001_single_finding/`)
   - Example: "Where is the infiltration?"
   - Ground truth bbox
   - Expected answer with segment reference

2. **Multiple findings** (`case_002_multi_finding/`)
   - Example: "What abnormalities are present?"
   - Multiple bboxes (2-3 findings)
   - Tests multi-segment grounding

3. **Negative case** (`case_003_negative/`)
   - Example: "Is there any abnormality?"
   - No bbox (normal image)
   - Tests "no evidence → no claim" principle

4. **Ambiguous/subtle finding** (`case_004_subtle/`)
   - Example: "What is the opacity in the right lung?"
   - Bbox with lower confidence
   - Tests uncertainty handling

5. **Follow-up scenario** (`case_005_followup/`)
   - Two time points (baseline + follow-up)
   - Same anatomical region
   - Tests measurement tracking

### File Structure per Case

```
sample_cxr_vqa/
└── case_001_single_finding/
    ├── image.png                  # CXR image
    ├── qa.json                    # Question-answer pairs
    ├── bboxes.json                # Ground truth bounding boxes
    └── metadata.json              # Case metadata
```

### JSON Formats

**qa.json:**
```json
{
  "questions": [
    {
      "question_id": "Q001",
      "question_type": "Where",
      "question_text": "Where is the infiltration?",
      "answer": "Right lower lobe",
      "reasoning": "Dense consolidation visible in right lower lobe with air bronchograms"
    }
  ]
}
```

**bboxes.json:**
```json
{
  "annotations": [
    {
      "finding_id": "F001",
      "label": "infiltration",
      "bbox_px": [320, 250, 480, 410],
      "confidence": "high"
    }
  ],
  "image_size": {
    "width": 512,
    "height": 512
  }
}
```

**metadata.json:**
```json
{
  "case_id": "001",
  "dataset": "VinDr-CXR-VQA",
  "modality": "chest_xray",
  "view": "PA",
  "use_case": "single_finding_grounding",
  "difficulty": "easy"
}
```

## Downloading Sample Data

To populate this directory with actual VinDr-CXR-VQA samples:

1. **Access the dataset** via Hugging Face or the VinDr portal
2. **Select representative cases** matching the 5 scenarios above
3. **Convert to the expected format** (image.png, qa.json, bboxes.json, metadata.json)
4. **Place in respective subdirectories** under `sample_cxr_vqa/`

## License & Attribution

When using VinDr-CXR-VQA data:
- Follow the dataset's license terms
- Cite the original paper in publications
- Use only for research/educational purposes as specified

## Usage in Tests

```python
@pytest.fixture
def vindr_sample_001(sample_cxr_vqa_dir):
    """Load single finding case from VinDr-CXR-VQA."""
    case_dir = sample_cxr_vqa_dir / "case_001_single_finding"
    return {
        "image": Image.open(case_dir / "image.png"),
        "qa": json.load((case_dir / "qa.json").open()),
        "bboxes": json.load((case_dir / "bboxes.json").open()),
        "metadata": json.load((case_dir / "metadata.json").open()),
    }
```

## TODO

- [ ] Download 5-10 representative cases from VinDr-CXR-VQA
- [ ] Convert to standardized JSON format
- [ ] Create fixture loaders in conftest.py
- [ ] Add integration tests using these fixtures
