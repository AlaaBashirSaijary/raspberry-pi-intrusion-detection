"""يولّد results/comparison.md من نتائج اللابتوب وPi."""
import os, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
lap = pd.read_csv(os.path.join(R, "laptop_results.csv"))
pi_p = os.path.join(R, "pi_results.csv")
pi = pd.read_csv(pi_p) if os.path.exists(pi_p) else None
lap["Accuracy"] = lap.accuracy_mean.map("{:.3f}".format) + " ± " + lap.accuracy_std.map("{:.3f}".format)
lap["F1"] = lap.f1_mean.map("{:.3f}".format) + " ± " + lap.f1_std.map("{:.3f}".format)
lap["TPR"] = lap.tpr_mean.map("{:.3f}".format)
lap["FPR"] = lap.fpr_mean.map("{:.3f}".format) + " ± " + lap.fpr_std.map("{:.3f}".format)
t = lap[["model", "mode", "Accuracy", "F1", "TPR", "FPR"]]
lines = ["## الأداء (متوسط ± انحراف على 5 بذور)\n", t.to_markdown(index=False) if hasattr(t, "to_markdown") else t.to_string(index=False)]
if pi is not None:
    lines += ["\n## الموارد على Raspberry Pi\n", pi.drop_duplicates("model", keep="last").to_markdown(index=False)]
else:
    lines += ["\n## الموارد على Raspberry Pi\n", "بانتظار الجهاز."]
open(os.path.join(R, "comparison.md"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
