# Hybrid_fasteval Notebooks

Interactive Jupyter notebooks for debugging and understanding Qwen 3.5 hybrid architecture models.

## Notebooks

### 1. debug_conflict_responses.ipynb
**Purpose**: Interactive debugging tool to inspect model responses to conflict prompts

**Features**:
- Load and inspect conflict dataset samples
- Deep dive into single samples (prompt, response, verdict, logits)
- Batch process multiple samples with statistics
- Visualize verdict distribution and logit gaps
- Save results to JSON/CSV for further analysis

**Use cases**:
- Verify experimental results
- Debug unexpected model behavior
- Understand "other" classifications
- Validate conflict resolution metrics (CFR, logit gap)

**Runtime**: ~5-10 minutes for 5 samples on CPU

---

### 2. qwen_architecture_tutorial.ipynb
**Purpose**: Comprehensive tutorial on Qwen 3.5's hybrid GDN architecture

**Features**:
- Model configuration and layer structure analysis
- Layer-by-layer breakdown (GDN vs Attention)
- Hidden state extraction and visualization
- Logit lens analysis (layer-wise predictions)
- Conflict resolution tracking (parametric vs in-context)

**Topics covered**:
- Hybrid architecture (75% GDN, 25% attention)
- Layer type distribution and patterns
- Module inspection (attention vs GDN layers)
- Hidden state evolution across layers
- Prediction confidence trajectories

**Runtime**: ~10-15 minutes on CPU

---

## Setup

### Prerequisites
```bash
cd /Users/smaller225/code/Hybrid_fasteval
conda activate hybrid  # or your environment
```

### Required Packages
All required packages should already be installed:
- torch
- transformers
- pandas
- matplotlib
- seaborn
- psutil

### Data Preparation

If `data/output/short_conflict.jsonl` doesn't exist, run:

```bash
cd /Users/smaller225/code/Hybrid_fasteval
python project/data/prepare_counterfact.py --n 200 --out data/output/short_conflict.jsonl --seed 42
```

This downloads the CounterFact dataset and prepares 200 conflict samples.

---

## Usage

### Start Jupyter
```bash
cd /Users/smaller225/code/Hybrid_fasteval/notebooks
jupyter notebook
```

### Run Notebooks
1. Open a notebook in Jupyter
2. Run cells sequentially (Shift+Enter)
3. Modify configuration cells as needed:
   - Change `MODEL_NAME` to use different models
   - Adjust `n_samples` to process more/fewer samples
   - Set `SAMPLE_IDX` to inspect different samples

### Configuration Options

**Models**:
- `qwen3.5-4b` - 4B parameters (default, ~8GB RAM)
- `qwen3.5-2b` - 2B parameters (fallback, ~4GB RAM)

**Sample size**:
- Start with `n_samples=5` for quick testing
- Increase to 10-20 for more comprehensive analysis
- CPU inference is slow (~10-30 sec per sample)

---

## Output Files

Results are saved to `results/notebook_outputs/`:

**From debug_conflict_responses.ipynb**:
- `debug_summary_{model_name}.png` - Visualization summary
- `debug_results_{model_name}.json` - Detailed results with statistics
- `debug_results_{model_name}.csv` - Results table

**From qwen_architecture_tutorial.ipynb**:
- `qwen_layer_pattern_{model_name}.png` - Layer architecture visualization
- `qwen_hidden_states_{model_name}.png` - Hidden state analysis
- `qwen_logit_lens_{model_name}.png` - Logit lens predictions
- `qwen_conflict_resolution_{model_name}.png` - Conflict resolution tracking

---

## Memory Management

**Mac CPU (24GB RAM)**:
- Qwen3.5-4B: ~8GB model + ~4GB overhead = 12GB total
- Qwen3.5-2B: ~4GB model + ~2GB overhead = 6GB total

**If memory issues occur**:
1. Switch to `qwen3.5-2b` in configuration
2. Reduce `n_samples` to 3
3. Reduce `max_new_tokens` to 20
4. Close other applications

---

## Troubleshooting

**Issue**: `Data file not found`
- **Solution**: Run data preparation command (see Setup section)

**Issue**: `Model loading takes too long`
- **Expected**: 30-60 seconds on CPU is normal
- **Solution**: Use smaller model (qwen3.5-2b)

**Issue**: `Memory error`
- **Solution**: Switch to smaller model, reduce n_samples

**Issue**: `KISTI-specific issues`
- **Note**: These notebooks are for local Mac use only
- For KISTI: use batch scripts in `scripts/` directory

**Issue**: `Import errors`
- **Solution**: Ensure you're in the correct conda environment
- Check that `project/` directory is in Python path

---

## Technical Details

### Shared Utilities
Located in `notebooks/utils/notebook_helpers.py`:
- `setup_project_path()` - Add project to sys.path
- `format_model_info()` - Format model info as DataFrame
- `display_prompt_preview()` - Display prompts with truncation
- `color_verdict()` - Styling for verdict display
- `extract_hidden_states()` - Extract hidden states using hooks
- `compute_logit_lens()` - Project hidden states to vocabulary
- `get_final_norm()`, `get_lm_head()` - Get model components

### Architecture Notes

**Qwen 3.5 Hybrid Architecture**:
- **GDN layers** (75%): Linear recurrent processing, O(n) complexity
- **Attention layers** (25%): Full self-attention, O(n²) complexity
- **Pattern**: Attention every 4th layer (e.g., layers 3, 7, 11, 15...)

**Logit Lens**:
- Projects intermediate hidden states to vocabulary
- Reveals what model "thinks" at each layer
- Shows how predictions evolve and converge

---

## Next Steps

**After running notebooks**:
1. Compare results across different models
2. Test with custom prompts
3. Analyze failure cases
4. Extend to longer contexts
5. Compare with pure attention models (Llama)

**Related Experiments**:
- `scripts/stage1_conflict_eval.sh` - Basic conflict evaluation
- `scripts/stage2_thinking_eval.sh` - Thinking mode evaluation
- Results in `results/` directory

---

## References

- [Qwen 3.5 Model Card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [GDN/Lightning Attention Paper](https://arxiv.org/abs/2401.04695)
- [Logit Lens Analysis](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- [CounterFact Dataset](https://github.com/kmeng01/rome)

---

## Contact

For issues or questions:
- Check existing results in `results/` directory
- Review KISTI setup in `/Users/smaller225/.claude/projects/-Users-smaller225/memory/MEMORY.md`
- Consult project documentation
