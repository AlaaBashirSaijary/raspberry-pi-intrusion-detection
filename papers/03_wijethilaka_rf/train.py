"""حلقة Wijethilaka: Random Forest."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sklearn.ensemble import RandomForestClassifier
from common.runner import run

run("wijethilaka_rf", lambda s: RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1))
