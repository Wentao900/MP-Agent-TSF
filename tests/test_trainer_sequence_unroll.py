from __future__ import annotations

import numpy as np
from pathlib import Path

from imtsa.train.trainer import train_model


def _bundle(n: int = 16, seq_len: int = 8, p_dim: int = 4, t_dim: int = 3) -> dict[str, np.ndarray]:
    return {
        "x_price": np.random.randn(n, seq_len, p_dim).astype(np.float32),
        "x_text": np.random.randn(n, seq_len, t_dim).astype(np.float32),
        "y_action": np.random.randint(0, 3, size=(n,), dtype=np.int64),
        "y_return": np.random.randn(n).astype(np.float32),
    }


def test_trainer_sequence_unroll_runs(tmp_path: Path) -> None:
    cfg = {
        "model": {"hidden_dim": 16, "memory_dim": 8, "num_actions": 3, "reflector_k": 4},
        "ablation": {"use_memory": True, "use_reflector": True, "use_explain_loss": True},
        "train": {
            "batch_size": 4,
            "epochs": 1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "sequence_len": 2,
            "lambda_act": 1.0,
            "lambda_pred": 0.3,
            "lambda_exp": 0.1,
            "lambda_smooth": 0.05,
            "lambda_reflect_corr": 0.1,
            "lambda_exp_faith": 0.1,
            "lambda_exp_stability": 0.1,
            "exp_perturb_ratio": 0.2,
            "reward_norm_scale": 100.0,
        },
    }
    train_bundle = _bundle()
    val_bundle = _bundle()
    artifacts = train_model(train_bundle, val_bundle, cfg, tmp_path)
    assert artifacts.model_path.exists()
    assert artifacts.metrics_path.exists()
