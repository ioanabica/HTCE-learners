import random
from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from htce_learners.constants import DEFAULT_SEED, DEFAULT_VAL_SPLIT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

TRAIN_STRING = "training"
VALIDATION_STRING = "validation"


def flatten(_list):
    return [item for sublist in _list for item in sublist]


def check_tensor(X: torch.Tensor) -> torch.Tensor:
    if isinstance(X, torch.Tensor):
        return X.to(DEVICE)
    else:
        return torch.from_numpy(np.asarray(X)).to(DEVICE)


def make_target_val_split(
    X_target_specific: torch.Tensor,
    X_target_shared: torch.Tensor,
    y: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    seed: int = DEFAULT_SEED,
    stratify_w: bool = True,
) -> Any:
    if val_split_prop == 0:
        # return original data
        X = None
        if w is None:
            return X_target_specific, X_target_shared, y, X, y, TRAIN_STRING

        return X_target_specific, X_target_shared, y, w, X, y, w, TRAIN_STRING

    X_target_specific = X_target_specific.cpu()
    X_target_shared = X_target_shared.cpu()
    y = y.cpu()
    # make actual split
    if w is None:
        (
            X_target_specific_t,
            X_target_specific_val,
            X_target_shared_t,
            X_target_shared_val,
            y_t,
            y_val,
        ) = train_test_split(
            X_target_specific, X_target_shared, y, test_size=val_split_prop, random_state=seed, shuffle=True
        )
        return (
            X_target_specific_t.to(DEVICE),  # type: ignore
            X_target_shared_t.to(DEVICE),  # type: ignore
            y_t.to(DEVICE),  # type: ignore
            X_target_specific_val.to(DEVICE),  # type: ignore
            X_target_shared_val.to(DEVICE),  # type: ignore
            y_val.to(DEVICE),  # type: ignore
            VALIDATION_STRING,
        )

    w = w.cpu()
    if stratify_w:
        # split to stratify by group
        (
            X_target_specific_t,
            X_target_specific_val,
            X_target_shared_t,
            X_target_shared_val,
            y_t,
            y_val,
            w_t,
            w_val,
        ) = train_test_split(
            X_target_specific,
            X_target_shared,
            y,
            w,
            test_size=val_split_prop,
            random_state=seed,
            stratify=w,
            shuffle=True,
        )
    else:
        (
            X_target_specific_t,
            X_target_specific_val,
            X_target_shared_t,
            X_target_shared_val,
            y_t,
            y_val,
            w_t,
            w_val,
        ) = train_test_split(
            X_target_specific, X_target_shared, y, w, test_size=val_split_prop, random_state=seed, shuffle=True
        )

    return (
        X_target_specific_t.to(DEVICE),  # type: ignore
        X_target_shared_t.to(DEVICE),  # type: ignore
        y_t.to(DEVICE),  # type: ignore
        w_t.to(DEVICE),  # type: ignore
        X_target_specific_val.to(DEVICE),  # type: ignore
        X_target_shared_val.to(DEVICE),  # type: ignore
        y_val.to(DEVICE),  # type: ignore
        w_val.to(DEVICE),  # type: ignore
        VALIDATION_STRING,
    )


def enable_reproducible_results(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def compute_pehe(cate_true, cate_pred):
    pehe = np.sqrt(mean_squared_error(cate_true, cate_pred.detach().cpu().numpy()))
    return pehe


def compute_cate_metrics(cate_true, y_true, w_true, mu0_pred, mu1_pred):
    mu0_pred = mu0_pred.detach().cpu().numpy()
    mu1_pred = mu1_pred.detach().cpu().numpy()

    cate_pred = mu1_pred - mu0_pred

    pehe = np.sqrt(mean_squared_error(cate_true, cate_pred))

    y_pred = w_true.reshape(len(cate_true),) * mu1_pred.reshape(len(cate_true),) + (
        1
        - w_true.reshape(
            len(cate_true),
        )
    ) * mu0_pred.reshape(
        len(cate_true),
    )
    factual_rmse = np.sqrt(
        mean_squared_error(
            y_true.reshape(
                len(cate_true),
            ),
            y_pred,
        )
    )
    return pehe, factual_rmse
