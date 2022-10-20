import abc
from typing import Optional, Tuple

import torch
from torch import nn

from contrib.catenets.models.torch.utils.transformations import dr_transformation_cate
from contrib.catenets.models.torch.utils.weight_utils import compute_importance_weights
from htce_learners.base_htce_layers import DEVICE, HTCEBaseEstimator
from htce_learners.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LAYERS_OUT,
    DEFAULT_LAYERS_OUT_T,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_MIN,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_NONLIN,
    DEFAULT_PATIENCE,
    DEFAULT_PENALTY_L2,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_STEP_SIZE_T,
    DEFAULT_UNITS_OUT,
    DEFAULT_UNITS_OUT_T,
    DEFAULT_VAL_SPLIT,
)
from htce_learners.htce_one_step_learners import HTCE_TLearner


class HTCE_PseudoOutcomeLearner(nn.Module):
    def __init__(
        self,
        name: str,
        n_unit_in_shared: int,
        n_unit_in_source_specific: int,
        n_unit_in_target_specific: int,
        binary_y: bool,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_layers_out_t: int = DEFAULT_LAYERS_OUT_T,
        n_units_out: int = DEFAULT_UNITS_OUT,
        n_units_out_t: int = DEFAULT_UNITS_OUT_T,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = 0,
        weight_decay: float = DEFAULT_PENALTY_L2,
        weight_decay_t: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        lr_t: float = DEFAULT_STEP_SIZE_T,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        weighting_strategy: Optional[str] = "prop",
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        early_stopping: bool = True,
    ):
        super(HTCE_PseudoOutcomeLearner, self).__init__()

        self.n_unit_in_shared = n_unit_in_shared
        self.n_unit_in_source_specific = n_unit_in_source_specific
        self.n_unit_in_target_specific = n_unit_in_target_specific
        self.binary_y = binary_y
        self.n_layers_out = n_layers_out
        self.n_units_out = n_units_out
        self.n_units_out_prop = n_units_out_prop
        self.n_layers_out_prop = n_layers_out_prop
        self.weight_decay_t = weight_decay_t
        self.weight_decay = weight_decay
        self.weighting_strategy = weighting_strategy
        self.lr = lr
        self.lr_t = lr_t
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_split_prop = val_split_prop
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.nonlin = nonlin
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.n_layers_out_t = n_layers_out_t
        self.n_units_out_t = n_units_out_t
        self.n_layers_out = n_layers_out
        self.n_units_out = n_units_out
        self.early_stopping = early_stopping

        self._te_estimator = self._generate_te_estimator()
        self._po_estimator = self._generate_po_estimator()
        if weighting_strategy is not None:
            self._propensity_estimator = self._generate_propensity_estimator()

    def _generate_te_estimator(self, name: str = "te_estimator") -> nn.Module:
        return HTCEBaseEstimator(
            name=name,
            n_unit_in_shared=self.n_unit_in_shared,
            n_unit_in_source_specific=self.n_unit_in_source_specific,
            n_unit_in_target_specific=self.n_unit_in_target_specific,
            binary_y=self.binary_y,
            batch_size=128,
            n_iter=self.n_iter,
            nonlin="relu",
        ).to(DEVICE)

    def _generate_po_estimator(self, name: str = "po_estimator") -> nn.Module:
        return HTCE_TLearner(
            name=name,
            n_unit_in_shared=self.n_unit_in_shared,
            n_unit_in_source_specific=self.n_unit_in_source_specific,
            n_unit_in_target_specific=self.n_unit_in_target_specific,
            binary_y=self.binary_y,
            batch_size=128,
            n_iter=self.n_iter,
            nonlin="relu",
        ).to(DEVICE)

    def _generate_propensity_estimator(self, name: str = "propensity_estimator") -> nn.Module:
        if self.weighting_strategy is None:
            raise ValueError("Invalid weighting_strategy for PropensityNet")
        return HTCEBaseEstimator(
            name=name,
            n_unit_in_shared=self.n_unit_in_shared,
            n_unit_in_source_specific=self.n_unit_in_source_specific,
            n_unit_in_target_specific=self.n_unit_in_target_specific,
            binary_y=True,
            batch_size=128,
            n_iter=self.n_iter,
            nonlin="relu",
        ).to(DEVICE)

    def train(
        self,
        X_source_specific,
        X_source_shared,
        X_target_specific,
        X_target_shared,
        y_source,
        y_target,
        w_source,
        w_target,
    ) -> "PseudoOutcomeLearner":

        # STEP 1: fit plug-in estimators
        (
            mu_0_pred_source,
            mu_0_pred_target,
            mu_1_pred_source,
            mu_1_pred_target,
            p_pred_source,
            p_pred_target,
        ) = self._first_step(
            X_source_specific,
            X_source_shared,
            X_target_specific,
            X_target_shared,
            y_source,
            y_target,
            w_source,
            w_target,
        )

        # use estimated propensity scores
        if self.weighting_strategy is not None:
            p_source = p_pred_source
            p_target = p_pred_target

        # STEP 2: direct TE estimation
        self._second_step(
            X_source_specific,
            X_source_shared,
            X_target_specific,
            X_target_shared,
            y_source,
            y_target,
            w_source,
            w_target,
            p_source,
            p_target,
            mu_0_pred_source,
            mu_0_pred_target,
            mu_1_pred_source,
            mu_1_pred_target,
        )

        return self

    def predict(self, X_specific, X_shared, return_po=False, training=False, env="target") -> torch.Tensor:
        if return_po:
            raise NotImplementedError("PseudoOutcomeLearners have no Potential outcome predictors.")
        outcome, y0_preds, y1_preds = self._po_estimator.predict(X_specific, X_shared, return_po=True, env=env)
        cate_pred = outcome

        cate_pred = self._te_estimator._forward(X_specific, X_shared, env)

        return cate_pred

    @abc.abstractmethod
    def _first_step(
        self,
        X_source_specific,
        X_source_shared,
        X_target_specific,
        X_target_shared,
        y_source,
        y_target,
        w_source,
        w_target,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def _second_step(
        self,
        X_source_specific,
        X_source_shared,
        X_target_specific,
        X_target_shared,
        y_source,
        y_target,
        w_source,
        w_target,
        p_source,
        p_target,
        mu_0_pred_source,
        mu_0_pred_target,
        mu_1_pred_source,
        mu_1_pred_target,
    ) -> None:
        pass

    def _impute_pos(
        self,
        X_source_specific,
        X_source_shared,
        X_target_specific,
        X_target_shared,
        y_source,
        y_target,
        w_source,
        w_target,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # fit two separate (standard) models - TLearner
        self._po_estimator.train(
            X_source_specific,
            X_source_shared,
            X_target_specific,
            X_target_shared,
            y_source,
            y_target,
            w_source,
            w_target,
        )

        _, mu_0_pred_source, mu_1_pred_source = self._po_estimator.predict(
            X_specific=X_source_specific, X_shared=X_source_shared, return_po=True, env="source"
        )
        _, mu_0_pred_target, mu_1_pred_target = self._po_estimator.predict(
            X_specific=X_target_specific, X_shared=X_target_shared, return_po=True, env="target"
        )

        return mu_0_pred_source, mu_0_pred_target, mu_1_pred_source, mu_1_pred_target

    def _impute_propensity(
        self, X_source_specific, X_source_shared, X_target_specific, X_target_shared, w_source, w_target
    ) -> torch.Tensor:
        # split sample
        # fit propensity estimator

        self._propensity_estimator = self._generate_propensity_estimator("prop_estimator_impute_propensity")
        self._propensity_estimator.train(
            X_source_specific, X_source_shared, X_target_specific, X_target_shared, w_source, w_target
        )

        p_pred_source = self._propensity_estimator.predict(X_source_specific, X_source_shared, env="source")
        p_pred_source = compute_importance_weights(p_pred_source, w_source, self.weighting_strategy, {})

        p_pred_target = self._propensity_estimator.predict(X_target_specific, X_target_shared, env="target")
        p_pred_target = compute_importance_weights(p_pred_target, w_target, self.weighting_strategy, {})

        return p_pred_source, p_pred_target


class HTCE_DRLearner(HTCE_PseudoOutcomeLearner):
    """
    DR-learner for CATE estimation, based on doubly robust AIPW pseudo-outcome
    """

    def _first_step(
        self,
        X_source_specific,
        X_source_shared,
        X_target_specific,
        X_target_shared,
        y_source,
        y_target,
        w_source,
        w_target,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_0_pred_source, mu_0_pred_target, mu_1_pred_source, mu_1_pred_target = self._impute_pos(
            X_source_specific,
            X_source_shared,
            X_target_specific,
            X_target_shared,
            y_source,
            y_target,
            w_source,
            w_target,
        )
        p_pred_source, p_pred_target = self._impute_propensity(
            X_source_specific, X_source_shared, X_target_specific, X_target_shared, w_source, w_target
        )
        return (
            mu_0_pred_source.squeeze(),
            mu_0_pred_target.squeeze(),
            mu_1_pred_source.squeeze(),
            mu_1_pred_target.squeeze(),
            p_pred_source.squeeze(),
            p_pred_target.squeeze(),
        )

    def _second_step(
        self,
        X_source_specific,
        X_source_shared,
        X_target_specific,
        X_target_shared,
        y_source,
        y_target,
        w_source,
        w_target,
        p_source,
        p_target,
        mu_0_source,
        mu_0_target,
        mu_1_source,
        mu_1_target,
    ) -> None:
        y_source_tensor = torch.Tensor(y_source).squeeze().to(DEVICE)
        y_target_tensor = torch.Tensor(y_target).squeeze().to(DEVICE)

        w_source_tensor = torch.Tensor(w_source).squeeze().long().to(DEVICE)
        w_target_tensor = torch.Tensor(w_target).squeeze().long().to(DEVICE)

        pseudo_outcome_source = dr_transformation_cate(
            y_source_tensor, w_source_tensor, p_source, mu_0_source, mu_1_source
        )
        pseudo_outcome_target = dr_transformation_cate(
            y_target_tensor, w_target_tensor, p_target, mu_0_target, mu_1_target
        )

        pseudo_outcome_source = pseudo_outcome_source.cpu().detach().numpy()
        pseudo_outcome_target = pseudo_outcome_target.cpu().detach().numpy()

        self._te_estimator.train(
            X_source_specific,
            X_source_shared,
            X_target_specific,
            X_target_shared,
            pseudo_outcome_source,
            pseudo_outcome_target,
        )
