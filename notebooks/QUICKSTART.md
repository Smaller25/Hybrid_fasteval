# Quick Start Guide - Hybrid_fasteval Notebooks

## 5-Minute Setup

### 1. Prepare Data (One-time setup)
```bash
cd /Users/smaller225/code/Hybrid_fasteval
python project/data/prepare_counterfact.py --n 200 --out data/output/short_conflict.jsonl --seed 42
```

Expected output:
```
✓ Downloaded CounterFact dataset
✓ Prepared 200 conflict samples
✓ Saved to data/output/short_conflict.jsonl
```

### 2. Start Jupyter
```bash
cd /Users/smaller225/code/Hybrid_fasteval/notebooks
jupyter notebook
```

This opens a browser with the notebook interface.

---

## Notebook 1: Debug Conflict Responses

**Purpose**: Inspect individual model responses to conflict prompts

**Steps**:
1. Open `debug_conflict_responses.ipynb`
2. Run all cells (Cell → Run All)
3. Wait ~5-10 minutes for 5 samples on CPU
4. Check outputs in `results/notebook_outputs/`

**Key Outputs**:
- Verdict distribution (incontext/parametric/other)
- Context-following rate (CFR)
- Logit gap statistics
- Visualizations saved as PNG

**To customize**:
```python
# Cell 1.2: Change configuration
CONFIG = {
    'model_name': 'qwen3.5-2b',  # Use smaller model
    'n_samples': 3,              # Process fewer samples
    'max_new_tokens': 20,        # Shorter responses
}
```

**To inspect specific sample**:
```python
# Cell 2.1: Change sample index
SAMPLE_IDX = 2  # 0 to n_samples-1
```

---

## Notebook 2: Qwen Architecture Tutorial

**Purpose**: Understand Qwen 3.5 hybrid architecture

**Steps**:
1. Open `qwen_architecture_tutorial.ipynb`
2. Run all cells (Cell → Run All)
3. Wait ~10-15 minutes on CPU
4. Check visualizations inline and saved PNGs

**Key Outputs**:
- Layer pattern visualization (GDN vs Attention)
- Hidden state evolution plots
- Logit lens predictions
- Conflict resolution tracking

**To customize**:
```python
# Cell 1.2: Change model
MODEL_NAME = 'qwen3.5-2b'  # Use smaller model

# Cell 4.1: Change prompt
PROMPT = "Your custom prompt here"

# Cell 3.1: Inspect different layer
INSPECT_LAYER = 5  # 0 to num_layers-1
```

---

## Expected Results

### Debug Notebook
```
Summary Statistics
==================================================
Verdict Distribution:
  incontext: 3 (60.0%)
  parametric: 2 (40.0%)

Context-Following Rate (CFR): 0.600

Logit Gap Statistics:
  Mean: 0.1234
  Median: 0.0987
  ...
```

### Architecture Notebook
```
Layer Type Distribution:
==================================================
Total Layers: 40
Attention Layers: 10 (25.0%)
GDN/Linear Layers: 30 (75.0%)
==================================================

Attention Layer Positions: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
```

---

## Troubleshooting

### Data file missing
```bash
# Run preparation script
cd /Users/smaller225/code/Hybrid_fasteval
python project/data/prepare_counterfact.py --n 200 --out data/output/short_conflict.jsonl
```

### Memory error
```python
# Use smaller model in Cell 1.2
CONFIG = {
    'model_name': 'qwen3.5-2b',  # Instead of qwen3.5-4b
    'n_samples': 3,              # Reduce samples
}
```

### Slow inference
This is **expected on CPU**:
- 4B model: ~20-30 sec per sample
- 2B model: ~10-15 sec per sample

**Tip**: Start with `n_samples=3` for quick testing

### Import errors
```bash
# Ensure correct environment
conda activate hybrid

# Check project path in first cell
sys.path.insert(0, '/Users/smaller225/code/Hybrid_fasteval/project')
```

---

## What to Look For

### Debug Notebook
✓ **High CFR (>0.5)** = Model follows in-context info
✓ **Positive logit gap** = Prefers in-context
✓ **Few "other"** = Clean classifications

⚠️ **Low CFR (<0.3)** = Model ignores context
⚠️ **Negative logit gap** = Prefers parametric
⚠️ **Many "other"** = Unexpected responses

### Architecture Notebook
✓ **Clear 3:1 pattern** = Hybrid architecture working
✓ **Increasing L2 norm** = Hidden states evolving
✓ **Convergent predictions** = Logit lens shows refinement

⚠️ **Flat L2 norm** = May indicate issues
⚠️ **Unstable predictions** = Layer outputs inconsistent

---

## Next Steps

After running both notebooks:

1. **Compare models**
   - Run with `qwen3.5-4b` and `qwen3.5-2b`
   - Compare CFR and logit gaps

2. **Test custom prompts**
   - Cell 4.1 in architecture notebook
   - Create your own conflict scenarios

3. **Inspect failure cases**
   - Cell 4.1 in debug notebook
   - Analyze "other" responses

4. **Run full experiments**
   - Use `scripts/stage1_conflict_eval.sh`
   - Compare with notebook results

---

## Time Estimates

| Task | CPU Time | Output |
|------|----------|--------|
| Data preparation | 2-3 min | 200 samples |
| Debug notebook (5 samples) | 5-10 min | 3 files |
| Architecture notebook | 10-15 min | 4 files |
| **Total first run** | **20-30 min** | **7+ files** |

---

## File Locations

**Notebooks**:
- `/Users/smaller225/code/Hybrid_fasteval/notebooks/debug_conflict_responses.ipynb`
- `/Users/smaller225/code/Hybrid_fasteval/notebooks/qwen_architecture_tutorial.ipynb`

**Data**:
- `/Users/smaller225/code/Hybrid_fasteval/data/output/short_conflict.jsonl`

**Results**:
- `/Users/smaller225/code/Hybrid_fasteval/results/notebook_outputs/`

**Documentation**:
- `README.md` - Comprehensive guide
- `QUICKSTART.md` - This file

---

## Tips

💡 **Start small**: Use `n_samples=3` and `qwen3.5-2b` for first run
💡 **Run sequentially**: Execute cells one-by-one to understand each step
💡 **Save often**: Results are automatically saved to `results/notebook_outputs/`
💡 **Customize**: Change prompts, models, and samples to explore
💡 **Compare**: Run same notebook with different configurations

---

Ready to start? Run the data preparation command above, then launch Jupyter! 🚀
