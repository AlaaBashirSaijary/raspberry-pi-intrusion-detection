"""حلقة Sai: SVM مع اختيار خصائص. (طبّقي ما في الورقة وسجّلي الفروق في README)"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from common.runner import run

# الاختيار داخل الـ pipeline، فيُحسب على التدريب فقط (منع التسرب)
run("sai_svm", lambda s: make_pipeline(SelectKBest(f_classif, k=20),
                                       SVC(kernel="rbf", C=1.0, random_state=s)))
