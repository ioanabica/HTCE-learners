# stdlib
from typing import Tuple

# third party
import numpy as np
import torch
from scipy.special import expit
from scipy.stats import zscore

from utils import enable_reproducible_results

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore


class SyntheticSimulatorBase:
    def __init__(self, seed=42):
        enable_reproducible_results(seed=seed)
        self.seed = seed

    def predict_Y0(self, X_source_specific, X_source_shared, X_target_specific, X_target_shared):
        raise NotImplementedError

    def predict_Y1(self, X_source_specific, X_source_shared, X_target_specific, X_target_shared):
        raise NotImplementedError

    def simulate_dataset(
        self,
        X_source_specific,
        X_source_shared,
        X_target_specific,
        X_target_shared,
        treatment_assign="random",
        noise=True,
        err_std=0.001,
        binary_outcome=False,
        prop_scale_source=1,
        pred_scale_target=1,
    ) -> Tuple:

        enable_reproducible_results(self.seed)
        self.treatment_assign = treatment_assign
        self.noise = noise
        self.err_std = err_std
        self.prop_scale_source = prop_scale_source
        self.pred_scale_target = pred_scale_target

        Y0_source, Y0_target = self.predict_Y0(X_source_specific, X_source_shared, X_target_specific, X_target_shared)
        Y1_source, Y1_target = self.predict_Y1(X_source_specific, X_source_shared, X_target_specific, X_target_shared)

        if self.treatment_assign == "random":
            self.propensity_source = 0.5 * np.ones(len(X_source_shared))
            self.propensity_target = 0.5 * np.ones(len(X_target_shared))
        elif self.treatment_assign == "pred":
            pred_score_source = zscore(Y1_source - Y0_source)
            self.propensity_source = expit(self.prop_scale_source * pred_score_source)

            pred_score_target = zscore(Y1_target - Y0_target)
            self.propensity_target = expit(self.pred_scale_target * pred_score_target)
        else:
            raise ValueError(f"{treatment_assign} is not a valid treatment assignment mechanism.")

        W_source = np.random.binomial(1, p=self.propensity_source)
        W_target = np.random.binomial(1, p=self.propensity_target)

        error_source = 0
        error_target = 0
        if self.noise:
            error_source = torch.empty(len(X_source_shared)).normal_(std=self.err_std).squeeze().detach().cpu().numpy()
            error_target = torch.empty(len(X_target_shared)).normal_(std=self.err_std).squeeze().detach().cpu().numpy()

        Y_source = W_source * Y1_source + (1 - W_source) * Y0_source + error_source
        Y_target = W_target * Y1_target + (1 - W_target) * Y0_target + error_target

        PO_source = (Y0_source, Y1_source)
        PO_target = (Y0_target, Y1_target)

        if binary_outcome:
            Y_prob_source = expit(2 * (Y_source - np.mean(Y_source)) / np.std(Y_source))
            Y_source = np.random.binomial(1, Y_prob_source)

            Y_prob_target = expit(2 * (Y_source - np.mean(Y_source)) / np.std(Y_source))
            Y_target = np.random.binomial(1, Y_prob_target)

        return W_source, Y_source, PO_source, W_target, Y_target, PO_target


class SyntheticSimulatorLinear(SyntheticSimulatorBase):
    def __init__(
        self,
        X_source_specific,
        X_source_shared,
        X_target_specific,
        X_target_shared,
        alpha=0.5,
        beta=0.5,
        seed=42,
    ) -> None:
        super(SyntheticSimulatorLinear, self).__init__(seed=seed)

        scale_factor = 10
        self.po_sharing_source = np.random.uniform(
            -scale_factor, scale_factor, size=(X_source_specific.shape[1] + X_source_shared.shape[1])
        )
        self.po_sharing_target = np.random.uniform(
            -scale_factor, scale_factor, size=(X_target_specific.shape[1] + X_target_shared.shape[1])
        )

        self.source_weights_Y0 = np.random.uniform(-scale_factor, scale_factor, size=(X_source_specific.shape[1]))
        self.shared_weights_Y0 = np.random.uniform(-scale_factor, scale_factor, size=(X_source_shared.shape[1]))
        self.target_weights_Y0 = np.random.uniform(-scale_factor, scale_factor, size=(X_target_specific.shape[1]))

        self.source_weights_Y1 = np.random.uniform(-scale_factor, scale_factor, size=(X_source_specific.shape[1]))
        self.shared_weights_Y1 = np.random.uniform(-scale_factor, scale_factor, size=(X_target_shared.shape[1]))
        self.target_weights_Y1 = np.random.uniform(-scale_factor, scale_factor, size=(X_target_specific.shape[1]))

        self.alpha = alpha
        self.beta = beta

    def predict_Y0(self, X_source_specific, X_source_shared, X_target_specific, X_target_shared):
        shared_source = np.dot(X_source_shared, self.shared_weights_Y0) / X_source_shared.shape[1]
        y0_specific_source = np.dot(X_source_specific, self.source_weights_Y0) / X_source_specific.shape[1]
        po_sharing_source = (
            np.dot(np.concatenate([X_source_shared, X_source_specific], axis=1), self.po_sharing_source)
            / np.concatenate([X_source_shared, X_source_specific], axis=1).shape[1]
        )
        Y0_source = self.alpha * shared_source + (1 - self.alpha) * (
            self.beta * y0_specific_source + (1 - self.beta) * po_sharing_source
        )

        shared_target = np.dot(X_target_shared, self.shared_weights_Y0) / X_target_shared.shape[1]
        yo_specific_target = np.dot(X_target_specific, self.target_weights_Y0) / X_target_specific.shape[1]
        po_sharing_target = (
            np.dot(np.concatenate([X_target_shared, X_target_specific], axis=1), self.po_sharing_target)
            / np.concatenate([X_target_shared, X_target_specific], axis=1).shape[1]
        )
        Y0_target = self.alpha * shared_target + (1 - self.alpha) * (
            self.beta * yo_specific_target + (1 - self.beta) * po_sharing_target
        )

        return Y0_source, Y0_target

    def predict_Y1(self, X_source_specific, X_source_shared, X_target_specific, X_target_shared):
        shared_source = np.dot(X_source_shared, self.shared_weights_Y1) / X_source_shared.shape[1]
        y1_specific_source = np.dot(X_source_specific, self.source_weights_Y1) / X_source_specific.shape[1]
        po_sharing_source = (
            np.dot(np.concatenate([X_source_shared, X_source_specific], axis=1), self.po_sharing_source)
            / np.concatenate([X_source_shared, X_source_specific], axis=1).shape[1]
        )
        Y1_source = self.alpha * shared_source + (1 - self.alpha) * (
            self.beta * y1_specific_source + (1 - self.beta) * po_sharing_source
        )

        shared_target = np.dot(X_target_shared, self.shared_weights_Y1) / X_target_shared.shape[1]
        y1_specific_target = np.dot(X_target_specific, self.target_weights_Y1) / X_target_specific.shape[1]
        po_sharing_target = (
            np.dot(np.concatenate([X_target_shared, X_target_specific], axis=1), self.po_sharing_target)
            / np.concatenate([X_target_shared, X_target_specific], axis=1).shape[1]
        )
        Y1_target = self.alpha * shared_target + (1 - self.alpha) * (
            self.beta * y1_specific_target + (1 - self.beta) * po_sharing_target
        )

        return Y1_source, Y1_target
