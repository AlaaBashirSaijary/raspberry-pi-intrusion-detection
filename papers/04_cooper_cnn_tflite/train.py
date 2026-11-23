"""حلقة Cooper: CNN صغير مضغوط بصيغة TFLite. يحتاج tensorflow على اللابتوب."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import tensorflow as tf
from common.runner import run


class KerasCNN:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        tf.keras.utils.set_random_seed(self.seed)
        X = X[..., None]
        self.m = tf.keras.Sequential([
            tf.keras.layers.Input(X.shape[1:]),
            tf.keras.layers.Conv1D(16, 3, activation="relu"),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")])
        self.m.compile("adam", "binary_crossentropy")
        self.m.fit(X, y, epochs=5, batch_size=256, verbose=0)
        return self

    def predict(self, X):
        return (self.m.predict(X[..., None], verbose=0).ravel() > 0.5).astype(int)

    def save_artifacts(self, path):
        """تحويل النموذج إلى TFLite (يستدعيه المشغّل). measure_pi يقيس هذا الملف."""
        conv = tf.lite.TFLiteConverter.from_keras_model(self.m)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        open(path, "wb").write(conv.convert())


run("cooper_cnn", lambda s: KerasCNN(s))
