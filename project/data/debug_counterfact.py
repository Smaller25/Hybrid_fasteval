"""
Debug CounterFact dataset structure
"""
from datasets import load_dataset
import json

print("Loading CounterFact dataset...")
ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
print(f"Dataset size: {len(ds)}")
print()

# Check first record
print("=" * 50)
print("First record structure:")
print("=" * 50)
record = ds[0]
print(json.dumps(record, indent=2, default=str))
print()

# Check field names
print("=" * 50)
print("Available fields:")
print("=" * 50)
print(list(record.keys()))
print()

# Check requested_rewrite structure
if "requested_rewrite" in record:
    print("=" * 50)
    print("requested_rewrite structure:")
    print("=" * 50)
    rr = record["requested_rewrite"]
    print(json.dumps(rr, indent=2, default=str))
    print()

    # Check target_true and target_new
    if "target_true" in rr:
        print("target_true:", rr["target_true"])
    if "target_new" in rr:
        print("target_new:", rr["target_new"])
