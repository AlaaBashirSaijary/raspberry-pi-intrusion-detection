"""تحميل البيانات: الفصل بين التدريب والاختبار يتم قبل أي معالجة."""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def synthetic(n=6000, d=30, seed=0, shift=0.0):
    """بيانات تجريبية لاختبار الخط فقط. ليست نتائج بحثية."""
    r = np.random.RandomState(seed)
    y = r.randint(0, 2, n)
    X = r.randn(n, d) + y[:, None] * (0.8 - shift) * (np.arange(d) < 8)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    df["label"] = y
    return df


def _binary_labels(col, benign=0):
    """0 = طبيعي، 1 = هجوم. المقارنة رقمية إن أمكن (0 و0.0 متساويان)."""
    num = pd.to_numeric(col, errors="coerce")
    try:
        b = float(benign)
        if num.notna().all():
            return (num != b).astype(int).values
    except (TypeError, ValueError):
        pass
    return (col.astype(str).str.strip() != str(benign)).astype(int).values


def subsample(X, y, n, seed=0):
    """عيّنة طبقية ثابتة (نفس البذرة لكل الأبحاث) لتناسب النماذج البطيئة مثل SVM."""
    if not n or len(X) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs, ys


def read(path, label_col, benign=0):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    y = _binary_labels(df[label_col], benign)
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    return X, y


def _clean_split(X, y, seed, test_size=0.3):
    # 1) الفصل أولاً، قبل أي scaler أو اختيار خصائص
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          stratify=y, random_state=seed)
    # 2) إزالة صفوف الاختبار المطابقة تماماً لصفوف التدريب (منع التسرب)
    key = lambda d: pd.util.hash_pandas_object(d, index=False)
    mask = ~key(Xte).isin(set(key(Xtr))).values
    removed = int((~mask).sum())
    return Xtr, Xte.loc[mask], ytr, yte[mask], removed


def prepare(X, y, seed):
    Xtr, Xte, ytr, yte, removed = _clean_split(X, y, seed)
    sc = StandardScaler().fit(Xtr)            # يُضبط على التدريب فقط
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte, sc, removed


def prepare_cross(XA, yA, XB, yB, seed, train_frac=0.7):
    """تدريب على A واختبار على B بالخصائص المشتركة بالاسم فقط.
    البذرة تحدد عيّنة التدريب (طبقية من A)، فيعكس الانحراف تباين التدريب أيضاً.
    تُزال من B الصفوف المطابقة تماماً لأي صف في A."""
    cols = [c for c in XA.columns if c in set(XB.columns)]
    if not cols:
        raise ValueError("لا توجد خصائص مشتركة بالاسم. طابقي الأسماء أولاً.")
    XA, XB = XA[cols], XB[cols]
    Xtr, _, ytr, _ = train_test_split(XA, yA, train_size=train_frac,
                                      stratify=yA, random_state=seed)
    key = lambda d: pd.util.hash_pandas_object(d, index=False)
    mask = ~key(XB).isin(set(key(XA))).values
    removed = int((~mask).sum())
    XBc, yBc = XB.loc[mask], yB[mask]
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(XBc), ytr, yBc, sc, removed
