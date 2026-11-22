"""حلقة Diab: LightGBM."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from lightgbm import LGBMClassifier
from common.runner import run

run("diab_lightgbm", lambda s: LGBMClassifier(n_estimators=200, random_state=s, verbose=-1))
