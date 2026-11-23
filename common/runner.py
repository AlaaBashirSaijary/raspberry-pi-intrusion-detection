"""مشغّل موحد: كل بحث يمرّر دالة بناء النموذج فقط، والباقي ثابت."""
import argparse, os, time, csv
import numpy as np, joblib
from . import load_data as L
from .metrics import compute

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2, 3, 4]


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="CSV للتدريب والاختبار")
    p.add_argument("--test-data", help="CSV ثانٍ: تدريب على --data واختبار على هذا (التعميم)")
    p.add_argument("--label-col", default="label")
    p.add_argument("--benign", default="0")
    p.add_argument("--max-rows", type=int, default=None,
                   help="عيّنة طبقية ثابتة من كل CSV (لتسريع SVM وNetFlow الكبير)")
    p.add_argument("--synthetic", action="store_true", help="تجربة الخط ببيانات وهمية")
    return p.parse_args()


def _append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def run(name, build_model, fit=None):
    a = cli()
    if a.synthetic:
        dA = L.synthetic(seed=0); dB = L.synthetic(seed=1, shift=0.4)
        XA, yA = dA.drop(columns="label"), dA["label"].values
        XB, yB = dB.drop(columns="label"), dB["label"].values
        tag = "synthetic"
    else:
        XA, yA = L.subsample(*L.read(a.data, a.label_col, a.benign), a.max_rows)
        tag = os.path.basename(a.data)
        if a.test_data:
            XB, yB = L.subsample(*L.read(a.test_data, a.label_col, a.benign), a.max_rows)

    res = {k: [] for k in ["accuracy", "f1", "tpr", "fpr"]}
    times, removed_all = [], []
    for s in SEEDS:
        if a.test_data:
            Xtr, Xte, ytr, yte, sc, rem = L.prepare_cross(XA, yA, XB, yB, s)
            mode = f"cross:{tag}->{os.path.basename(a.test_data)}"
        else:
            Xtr, Xte, ytr, yte, sc, rem = L.prepare(XA, yA, s)
            mode = f"within:{tag}"
        m = build_model(s)
        removed_all.append(rem)
        t0 = time.time(); m.fit(Xtr, ytr); times.append(time.time() - t0)
        for k, v in compute(yte, m.predict(Xte)).items():
            res[k].append(v)
    row = {"model": name, "mode": mode, "seeds": len(SEEDS), "train_sec_mean": round(float(np.mean(times)), 2),
           "dup_removed_mean": round(float(np.mean(removed_all)), 1)}
    for k, v in res.items():
        row[k + "_mean"] = round(float(np.mean(v)), 4)
        row[k + "_std"] = round(float(np.std(v)), 4)
    _append(os.path.join(ROOT, "results", "laptop_results.csv"), row)
    mp = os.path.join(ROOT, "models")
    if hasattr(m, "save_artifacts"):      # نماذج TFLite: النموذج في ملف مستقل
        m.save_artifacts(os.path.join(mp, f"{name}.tflite"))
        joblib.dump({"scaler": sc}, os.path.join(mp, f"{name}.joblib"))
    else:
        joblib.dump({"model": m, "scaler": sc}, os.path.join(mp, f"{name}.joblib"))
    # عيّنة مُطبَّعة من الاختبار الحقيقي (أو الوهمي) لقياس Pi؛ تُحفظ في كل تشغيل
    np.save(os.path.join(mp, "sample_X.npy"), np.asarray(Xte[:500], dtype=np.float32))
    print(row)
    return row
