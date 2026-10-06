from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from src.models.cv_splitter import PurgedKFold


class MetaClassifier:
    def __init__(self, probability_cutoff: float = 0.62) -> None:
        self.probability_cutoff = probability_cutoff
        self.model: CalibratedClassifierCV | None = None
        self.feature_names: list[str] = []

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_splitter: PurgedKFold,
    ) -> dict[str, float]:
        self.feature_names = list(X.columns)

        base_lgbm = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )

        samples_info = pd.DataFrame(
            {"entry_time": X.index, "exit_time": X.index + pd.Timedelta(minutes=120)}
        )

        cv_splits = list(cv_splitter.split(samples_info))

        self.model = CalibratedClassifierCV(
            estimator=base_lgbm,
            method="sigmoid",
            cv=cv_splits,
        )

        self.model.fit(X, y)

        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= self.probability_cutoff).astype(int)

        accuracy = float(np.mean(preds == y))
        filtered_win_rate = (
            float(np.mean(y[preds == 1])) if np.sum(preds == 1) > 0 else 0.0
        )

        return {
            "train_samples": float(len(X)),
            "accuracy": round(accuracy, 4),
            "baseline_win_rate": round(float(y.mean()), 4),
            "filtered_win_rate": round(filtered_win_rate, 4),
            "trade_retention_pct": round(float(np.mean(preds == 1)), 4),
        }

    def predict_probability(self, feature_series: pd.Series | dict[str, float]) -> float:
        if self.model is None:
            return 1.0

        if isinstance(feature_series, dict):
            df_input = pd.DataFrame([feature_series])
        else:
            df_input = pd.DataFrame([feature_series.to_dict()])

        df_input = df_input.reindex(columns=self.feature_names, fill_value=0.0)
        prob = float(self.model.predict_proba(df_input)[0, 1])
        return round(prob, 4)

    def should_execute(self, feature_series: pd.Series | dict[str, float]) -> tuple[bool, float]:
        prob = self.predict_probability(feature_series)
        authorized = prob >= self.probability_cutoff
        return authorized, prob

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)

    def load(self, filepath: str | Path) -> None:
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
