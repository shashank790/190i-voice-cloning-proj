# Test the added preprocessing function from utils/text_preprocessing.py
import time
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from utils.text_preprocessing import preprocess_text


ORIGINALS = [
    "Dr. José bought 3,475 shares of Ångström-Tech for $1,299.50 each on 12/06/25. 🚀",
    "“Mr. O’Neill” lives at 123 St. Catherine Rd.",
    "He said, “That’s $3,003, okay?”",
    "On 01/02/23, Dr. Smith and Dr. Jones arrived.",
    "The total is $12,000 and the date is 07/04/22.",
]


SAMPLES = [
    {"orig": txt, "exp": preprocess_text(txt)}
    for txt in ORIGINALS
]


rows = []
for i, s in enumerate(SAMPLES, 1):
    t0 = time.perf_counter()
    out = preprocess_text(s["orig"])
    rt_ms = (time.perf_counter() - t0) * 1_000
    rows.append({
        "id":         i,
        "runtime_ms": round(rt_ms, 3),
        "len_before": len(s["orig"]),
        "len_after":  len(out),
        "reduction":  len(s["orig"]) - len(out),   # + = shorter, − = longer
        "correct":    out == s["exp"],             # should all be True
    })

df = pd.DataFrame(rows)

print("\nTest-set results\n")
print(tabulate(
    df[["id", "correct", "runtime_ms", "len_before", "len_after", "reduction"]],
    headers="keys", tablefmt="github"
))

print(f"\nOverall accuracy  : {df.correct.mean()*100:.1f}%")
print(f"Mean runtime      : {df.runtime_ms.mean():.3f} ms")
print(f"Mean length delta : {df.reduction.mean():.1f} chars\n")

def _zero_axis():
    ax = plt.gca()
    ax.axhline(0, color="black", linewidth=0.8)

plt.figure(figsize=(6, 3))
plt.bar(df.id, df.runtime_ms)
plt.title("Runtime per sample")
plt.xlabel("Sample")
plt.ylabel("ms")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3))
plt.bar(df.id, df.reduction)
plt.title("Length reduction per sample")
plt.xlabel("Sample")
plt.ylabel("chars ( + shorter / – longer )")
_zero_axis()
plt.tight_layout()
plt.show()
