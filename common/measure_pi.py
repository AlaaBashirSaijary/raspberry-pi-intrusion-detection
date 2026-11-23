"""قياس الموارد على Raspberry Pi (أو أي جهاز). الطاقة تُقرأ يدوياً من مقياس USB."""
import argparse, os, time, csv, glob
import numpy as np, joblib, psutil


def load_predictor(path):
    """يعيد دالة predict لصف واحد. يدعم joblib (sklearn/lightgbm) و.tflite."""
    if path.endswith(".tflite"):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        it = Interpreter(model_path=path, num_threads=1); it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]

        def predict(x):
            it.set_tensor(i["index"], x[..., None].astype(np.float32))
            it.invoke()
            return it.get_tensor(o["index"])
        return predict
    m = joblib.load(path)["model"]
    try:
        m.set_params(n_jobs=1)   # صف واحد: تعدد الخيوط يضيف كلفة لا تمثل الاستدلال الفعلي
    except (ValueError, AttributeError):
        pass
    return m.predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cpu_temp():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except OSError:
        return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="models/<name>.joblib")
    p.add_argument("--sample", default=os.path.join(ROOT, "models", "sample_X.npy"))
    p.add_argument("--watts", type=float, default=float("nan"), help="قراءة مقياس الطاقة أثناء التشغيل")
    a = p.parse_args()
    predict = load_predictor(a.model)
    X = np.load(a.sample)   # مُطبَّعة مسبقاً؛ تُنتَج في كل تشغيل تدريب
    for row in X[:5]:
        predict(row[None, :])  # إحماء
    proc = psutil.Process(); proc.cpu_percent(None)
    lat = []
    for row in X[:300]:
        t = time.perf_counter(); predict(row[None, :]); lat.append((time.perf_counter() - t) * 1000)
    out = {"model": os.path.splitext(os.path.basename(a.model))[0],
           "latency_ms_mean": round(float(np.mean(lat)), 3),
           "latency_ms_p95": round(float(np.percentile(lat, 95)), 3),
           "cpu_percent": proc.cpu_percent(None),
           "rss_mb": round(proc.memory_info().rss / 1e6, 1),
           "model_size_kb": round(os.path.getsize(a.model) / 1e3, 1),
           "temp_c": cpu_temp(), "watts": a.watts}
    path = os.path.join(ROOT, "results", "pi_results.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out))
        if new: w.writeheader()
        w.writerow(out)
    print(out)


if __name__ == "__main__":
    main()
