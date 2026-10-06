import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml

from src.alpha.orb_breakout import ORBBreakoutStrategy
from src.core.events import BarEvent
from src.models.cv_splitter import PurgedKFold
from src.models.labeler import TripleBarrierLabeler
from src.models.meta_classifier import MetaClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo Meta-Labeling")
    parser.add_argument("--data", type=str, default="data/raw/minute/cme_mnq_1m.parquet")
    parser.add_argument("--meta-config", type=str, default="configs/strategy/meta_model.yaml")
    parser.add_argument("--output-weights", type=str, default="src/models/weights/lgbm_orb_meta_v1.booster")
    args = parser.parse_args()

    with open(args.meta_config) as f:
        config = yaml.safe_load(f)

    print(f"[*] Cargando dataset histórico desde {args.data}...")
    df_raw = pd.read_parquet(args.data)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

    strategy = ORBBreakoutStrategy(symbol="MNQ")
    labeler = TripleBarrierLabeler(
        pt_multiplier=config["triple_barrier_labeling"]["pt_multiplier"],
        sl_multiplier=config["triple_barrier_labeling"]["sl_multiplier"],
        max_holding_minutes=120,
    )

    print("[*] Generando señales base del ORB y aplicando Triple Barrera...")
    signals = []
    features_list = []
    labels = []

    for i, row in df_raw.iterrows():
        bar = BarEvent(
            timestamp=row["timestamp"],
            symbol=row["symbol"],
            timeframe="1m",
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=float(row["volume"]),
        )
        sig = strategy.on_bar(bar)
        if sig is not None:
            future_bars = df_raw.iloc[i + 1 : i + 150]
            current_atr = sig.metadata.get("atr", 15.0)

            labeled_result = labeler.label_signal(
                entry_time=sig.timestamp,
                entry_price=sig.entry_price,
                direction=sig.direction,
                atr=current_atr,
                future_bars=future_bars,
            )

            if labeled_result is not None:
                signals.append(sig)
                labels.append(labeled_result.meta_label)
                norm_range = (sig.metadata["orb_high"] - sig.metadata["orb_low"]) / (current_atr * np.sqrt(15))
                features_list.append({
                    "entry_time": sig.timestamp,
                    "rvol_opening_range": sig.metadata.get("rvol", 1.0),
                    "range_to_atr_ratio": round(norm_range, 4),
                    "distance_to_vwap_zscore": round(abs(sig.entry_price - sig.metadata["vwap"]) / current_atr, 4),
                    "confidence_score": sig.confidence_score,
                })

    if not features_list:
        print("[!] No se generaron suficientes señales para entrenar.")
        return

    df_features = pd.DataFrame(features_list).set_index("entry_time")
    y = pd.Series(labels, index=df_features.index)

    print(f"[+] Total de señales identificadas: {len(y)}")
    print(f"[+] Win Rate base sin filtrar: {y.mean() * 100:.2f}%")

    cv_splitter = PurgedKFold(
        n_splits=config["training"]["cv_splits"],
        embargo_pct=config["training"]["embargo_pct"],
    )

    classifier = MetaClassifier(probability_cutoff=config["meta_labeling"]["execution_probability_cutoff"])
    print("[*] Entrenando LightGBM con Purged K-Fold y Calibración Isotónica...")
    metrics = classifier.train(X=df_features, y=y, cv_splitter=cv_splitter)

    print("\n" + "=" * 50)
    print("      RESULTADOS DE ENTRENAMIENTO META-LABELING")
    print("=" * 50)
    print(f" Muestras entrenadas:          {int(metrics['train_samples'])}")
    print(f" Exactitud (Accuracy):         {metrics['accuracy'] * 100:.2f}%")
    print(f" Win Rate Base:                {metrics['baseline_win_rate'] * 100:.2f}%")
    print(f" Win Rate Filtrado (P>=0.62):  {metrics['filtered_win_rate'] * 100:.2f}%")
    print(f" Retención de Trades:          {metrics['trade_retention_pct'] * 100:.2f}%")
    print("=" * 50)

    classifier.save(args.output_weights)
    print(f"[+] Pesos guardados en: {args.output_weights}\n")


if __name__ == "__main__":
    main()
