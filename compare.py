"""يولّد results/comparison.md من نتائج اللابتوب وPi.
يُؤخذ آخر تشغيل لكل (نموذج، وضع)، والتشغيلات الوهمية تُفصل عن الحقيقية."""
import os, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
lap = pd.read_csv(os.path.join(R, "laptop_results.csv"))
lap = lap.drop_duplicates(["model", "mode"], keep="last")
pi_p = os.path.join(R, "pi_results.csv")
pi = pd.read_csv(pi_p) if os.path.exists(pi_p) else None


def pm(m, s):
    return lap[m].map("{:.3f}".format) + " ± " + lap[s].map("{:.3f}".format)


for c in ["accuracy", "f1", "tpr", "fpr"]:
    lap[c.upper() if c != "accuracy" else "Accuracy"] = pm(c + "_mean", c + "_std")
lap = lap.rename(columns={"F1": "F1"})
cols = ["model", "mode", "Accuracy", "F1", "TPR", "FPR"]
tbl = lambda d: d[cols].to_markdown(index=False)
syn = lap["mode"].str.contains("synthetic")
lines = []
if (~syn).any():
    lines += ["## الأداء (متوسط ± انحراف على 5 بذور)\n", tbl(lap[~syn])]
if syn.any():
    lines += ["\n## تجارب الخط ببيانات وهمية (ليست نتائج بحثية)\n", tbl(lap[syn])]
lines += ["\n## الموارد على Raspberry Pi\n",
          pi.drop_duplicates("model", keep="last").to_markdown(index=False)
          if pi is not None else "بانتظار الجهاز."]
open(os.path.join(R, "comparison.md"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
