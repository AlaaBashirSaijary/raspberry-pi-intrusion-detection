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


def read(path, label_col, benign=0):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    y = (df[label_col].astype(str) != str(benign)).astype(int).values
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


def prepare_cross(XA, yA, XB, yB):
    """تدريب على A واختبار على B بالخصائص المشتركة بالاسم فقط."""
    cols = [c for c in XA.columns if c in set(XB.columns)]
    if not cols:
        raise ValueError("لا توجد خصائص مشتركة بالاسم. طابقي الأسماء أولاً.")
    sc = StandardScaler().fit(XA[cols])
    return sc.transform(XA[cols]), sc.transform(XB[cols]), yA, yB, sc, cols
