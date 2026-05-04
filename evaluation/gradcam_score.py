import pandas as pd

df = pd.read_csv("evaluation/gradcam_results/gradcam_results.csv")
total = len(df)
correct = df["correct"].sum()
accuracy = correct / total * 100

print(f"Total cases: {total}")
print(f"Correct localizations: {int(correct)}")
print(f"Localization accuracy: {accuracy:.1f}%")
print(f"Target: ≥ 80%")
print(f"Result: {'✅ PASS' if accuracy >= 80 else '❌ FAIL'}")

# Per pathology breakdown
print("\nPer pathology:")
for path, group in df.groupby("pathology"):
    acc = group["correct"].mean() * 100
    print(f"  {path}: {acc:.0f}% ({int(group['correct'].sum())}/{len(group)})")