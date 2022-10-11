import numpy as np
import torch
from torch import nn
import contrib.catenets.logger as log

from htce_learners.base_htce_layers import DEVICE, FlexTENetHTCE, SharedRepresentationNet, RepresentationNet
from htce_learners.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LAYERS_R,
    DEFAULT_LAYERS_R_SHARED,
    DEFAULT_UNITS_R_SHARED,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_MIN,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_NONLIN,
    DEFAULT_PATIENCE,
    DEFAULT_PENALTY_L2,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_UNITS_R,
    DEFAULT_VAL_SPLIT,
    LARGE_VAL,
)
from utils import make_target_val_split, flatten, check_tensor

EPS = 1e-8

NONLIN = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
}


class BaseHTCE_Learner(nn.Module):
    def __init__(
        self,
        binary_y: bool = False,
        weight_decay: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        penalty_disc: float = 0,
        early_stopping: bool = True,
        prop_loss_multiplier: float = 1,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        patience: int = DEFAULT_PATIENCE,
        clipping_value: int = 1,
    ) -> None:
        super(BaseHTCE_Learner, self).__init__()

        self.val_split_prop = val_split_prop
        self.seed = seed
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.lr = lr
        self.weight_decay = weight_decay
        self.binary_y = binary_y
        self.penalty_disc = penalty_disc
        self.early_stopping = early_stopping
        self.prop_loss_multiplier = prop_loss_multiplier
        self.n_iter_min = n_iter_min
        self.patience = patience
        self.clipping_value = clipping_value

    def train(
        self,
        X_source_specific, X_source_shared, X_target_specific, X_target_shared,
        y_source, y_target,
        w_source, w_target):
        X_source_specific = torch.Tensor(X_source_specific).to(DEVICE)
        X_source_shared = torch.Tensor(X_source_shared).to(DEVICE)
        X_target_specific = torch.Tensor(X_target_specific).to(DEVICE)
        X_target_shared = torch.Tensor(X_target_shared).to(DEVICE)

        y_source = torch.Tensor(y_source).squeeze().to(DEVICE)
        y_target = torch.Tensor(y_target).squeeze().to(DEVICE)

        w_source = torch.Tensor(w_source).squeeze().long().to(DEVICE)
        w_target = torch.Tensor(w_target).squeeze().long().to(DEVICE)

        X_target_specific, X_target_shared, y_target, w_target, X_target_specific_val, X_target_shared_val, y_target_val, w_target_val, val_string = make_target_val_split(
            X_target_specific, X_target_shared, y_target, w=w_target, val_split_prop=self.val_split_prop, seed=self.seed
        )

        n_target = X_target_specific.shape[0]
        n_source = X_source_specific.shape[0]

        # calculate number of batches per epoch
        batch_size = self.batch_size if self.batch_size < n_target else n_target
        n_batches_target = int(np.round(n_target / batch_size)) if batch_size < n_target else 1
        train_indices_target = np.arange(n_target)

        n_batches_source = int(np.round(n_source / batch_size)) if batch_size < n_source else 1
        train_indices_source = np.arange(n_source)

        params = []
        for module in self.model_modules:
            params.append(list(module.parameters()))

        optimizer = torch.optim.Adam(flatten(params), lr=self.lr, weight_decay=self.weight_decay)

        # training
        val_loss_best = LARGE_VAL
        patience = 0
        b_source = 0
        for i in range(self.n_iter):
            # shuffle data for minibatches
            np.random.shuffle(train_indices_target)
            np.random.shuffle(train_indices_source)

            train_loss_source, train_loss_target = [], []

            for b_target in range(n_batches_target):
                idx_next_target = train_indices_target[
                                  (b_target * batch_size): min((b_target + 1) * batch_size, n_target - 1)
                                  ]

                idx_next_source = train_indices_source[
                                  (b_source * batch_size): min((b_source + 1) * batch_size, n_source - 1)
                                  ]
                b_source = (b_source + 1) % n_batches_source

                X_target_specific_next = X_target_specific[idx_next_target]
                X_target_shared_next = X_target_shared[idx_next_target]
                w_target_next = w_target[idx_next_target].squeeze()
                y_target_next = y_target[idx_next_target].squeeze()

                X_source_specific_next = X_source_specific[idx_next_source]
                X_source_shared_next = X_source_shared[idx_next_source]
                w_source_next = w_source[idx_next_source].squeeze()
                y_source_next = y_source[idx_next_source].squeeze()

                po_preds_source, po_preds_target = self._step(X_source_specific_next, X_source_shared_next,
                                                              X_target_specific_next, X_target_shared_next)

                batch_loss_source = self.loss(po_preds_source, y_source_next, w_source_next)
                batch_loss_target = self.loss(po_preds_target, y_target_next, w_target_next)

                batch_loss = batch_loss_source + batch_loss_target
                ortho_penalty_shared = self._get_ortho_penalty_shared(X_source_specific_next, X_source_shared_next,
                                                                          X_target_specific_next, X_target_shared_next)
                batch_loss += ortho_penalty_shared

                ortho_penalty_flex = self._get_ortho_penalty_flex()
                batch_loss += ortho_penalty_flex

                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
                optimizer.step()

                train_loss_source.append(batch_loss_source.detach())
                train_loss_target.append(batch_loss_target.detach())

            train_loss_source = torch.Tensor(train_loss_source).to(DEVICE)
            train_loss_target = torch.Tensor(train_loss_target).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    po_preds = self._forward(X_target_specific_val, X_target_shared_val, env='target')
                    val_loss = self.loss(po_preds, y_target_val, w_target_val)
                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            patience = 0
                        else:
                            patience += 1
                        if patience > self.patience and ((i + 1) * n_batches_target > self.n_iter_min):
                            break
                    if i % self.n_iter_print == 0:
                        log.info(
                            f"[{self.name}] Epoch: {i}, current {val_string} loss: {val_loss} train_loss_target: "
                            f"{torch.mean(train_loss_target)} train_loss_source: {torch.mean(train_loss_source)} "
                        )

        return self

    def loss(
        self,
        po_pred: torch.Tensor,
        y_true: torch.Tensor,
        t_true: torch.Tensor,
    ) -> torch.Tensor:
        def head_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            if self.binary_y:
                return nn.BCELoss()(y_pred, y_true)
            else:
                return (y_pred - y_true) ** 2

        def po_loss(
            po_pred: torch.Tensor, y_true: torch.Tensor, t_true: torch.Tensor
        ) -> torch.Tensor:
            y0_pred = po_pred[:, 0]
            y1_pred = po_pred[:, 1]

            loss0 = torch.mean((1.0 - t_true) * head_loss(y0_pred, y_true))
            loss1 = torch.mean(t_true * head_loss(y1_pred, y_true))

            return loss0 + loss1

        return (
            po_loss(po_pred, y_true, t_true)
        )

    def _get_ortho_penalty_shared(self, X_source_specific, X_source_shared,
                                  X_target_specific, X_target_shared):
        raise NotImplementedError

    def _get_ortho_penalty_flex(self):
        raise NotImplementedError

    def _step(self, X_source_specific, X_source_shared, X_target_specific, X_target_shared):

        po_preds_source = self._forward(X_source_specific, X_source_shared, env='source')
        po_preds_target = self._forward(X_target_specific, X_target_shared, env='target')

        return po_preds_source, po_preds_target

    def _forward(self, X_specific, X_shared, env):
        raise NotImplementedError

    def predict(self, X_specific, X_shared, return_po=False, env='target'):
        preds = self._forward(X_specific, X_shared, env)
        y0_preds = preds[:, 0]
        y1_preds = preds[:, 1]

        outcome = y1_preds - y0_preds

        if return_po:
            return outcome, y0_preds, y1_preds

        return outcome

class HTCE_SLearner(nn.Module):
    def __init__(
        self,
        name: str,
        n_unit_in_shared: int,
        n_unit_in_source_specific: int,
        n_unit_in_target_specific: int,
        binary_y: bool = False,
        n_layers_r_shared: int = DEFAULT_LAYERS_R_SHARED,
        n_units_r_shared: int = DEFAULT_UNITS_R_SHARED,
        weight_decay: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        penalty_disc: float = 0,
        early_stopping: bool = True,
        prop_loss_multiplier: float = 1,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        patience: int = DEFAULT_PATIENCE,
    ) -> None:
        super(HTCE_SLearner, self).__init__()

        self._shared_repr_estimator = SharedRepresentationNet(
            n_unit_in_shared=n_unit_in_shared + 1, n_unit_in_source_specific=n_unit_in_source_specific,
            n_unit_in_target_specific=n_unit_in_target_specific,
            n_layers=n_layers_r_shared, n_units=n_units_r_shared, nonlin=nonlin)

        self._po_estimator = FlexTENetHTCE(
            f"{name}_po_estimator",
            n_units_r_shared * 2,
            binary_y=binary_y,
        )

        # return final architecture
        self.binary_y = binary_y

        self.name = name
        self.val_split_prop = val_split_prop
        self.seed = seed
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.lr = lr
        self.weight_decay = weight_decay
        self.binary_y = binary_y
        self.penalty_disc = penalty_disc
        self.early_stopping = early_stopping
        self.prop_loss_multiplier = prop_loss_multiplier
        self.n_iter_min = n_iter_min
        self.patience = patience

    def train(
        self,
        X_source_specific, X_source_shared, X_target_specific, X_target_shared,
        y_source, y_target,
        w_source, w_target):

        X_source_specific = torch.Tensor(X_source_specific).to(DEVICE)
        X_source_shared = torch.Tensor(X_source_shared).to(DEVICE)
        X_target_specific = torch.Tensor(X_target_specific).to(DEVICE)
        X_target_shared = torch.Tensor(X_target_shared).to(DEVICE)

        y_source = torch.Tensor(y_source).squeeze().to(DEVICE)
        y_target = torch.Tensor(y_target).squeeze().to(DEVICE)

        w_source = torch.Tensor(w_source).squeeze().long().to(DEVICE)
        w_target = torch.Tensor(w_target).squeeze().long().to(DEVICE)

        X_target_specific, X_target_shared, y_target, w_target, X_target_specific_val, X_target_shared_val, y_target_val, w_target_val, val_string = make_target_val_split(
            X_target_specific, X_target_shared, y_target, w=w_target, val_split_prop=self.val_split_prop, seed=self.seed
        )

        n_target = X_target_specific.shape[0]  # could be different from before due to split
        n_source = X_source_specific.shape[0]

        # calculate number of batches per epoch
        batch_size = self.batch_size if self.batch_size < n_target else n_target
        n_batches_target = int(np.round(n_target / batch_size)) if batch_size < n_target else 1
        train_indices_target = np.arange(n_target)

        n_batches_source = int(np.round(n_source / batch_size)) if batch_size < n_source else 1
        train_indices_source = np.arange(n_source)

        params = (
            list(self._shared_repr_estimator.parameters())
            + list(self._po_estimator.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        # training
        val_loss_best = LARGE_VAL
        patience = 0
        b_source = 0
        for i in range(self.n_iter):
            # shuffle data for minibatches
            np.random.shuffle(train_indices_target)
            np.random.shuffle(train_indices_source)

            train_loss_source, train_loss_target = [], []

            for b_target in range(n_batches_target):
                idx_next_target = train_indices_target[
                                  (b_target * batch_size): min((b_target + 1) * batch_size, n_target - 1)
                                  ]

                idx_next_source = train_indices_source[
                                  (b_source * batch_size): min((b_source + 1) * batch_size, n_source - 1)
                                  ]
                b_source = (b_source + 1) % n_batches_source

                X_target_specific_next = X_target_specific[idx_next_target]
                X_target_shared_next = X_target_shared[idx_next_target]
                w_target_next = w_target[idx_next_target].squeeze()
                y_target_next = y_target[idx_next_target].squeeze()

                X_source_specific_next = X_source_specific[idx_next_source]
                X_source_shared_next = X_source_shared[idx_next_source]
                w_source_next = w_source[idx_next_source].squeeze()
                y_source_next = y_source[idx_next_source].squeeze()

                preds_source, preds_target = self._step(X_source_specific_next, X_source_shared_next, w_source_next,
                                                        X_target_specific_next, X_target_shared_next, w_target_next)

                loss = nn.BCELoss() if self.binary_y else nn.MSELoss()

                batch_loss_source = loss(preds_source, y_source_next)
                batch_loss_target = loss(preds_target, y_target_next)

                batch_loss = batch_loss_source + batch_loss_target

                ortho_penalty_shared = self._get_ortho_penalty_shared(X_source_specific_next, X_source_shared_next, w_source_next,
                                                                          X_target_specific_next, X_target_shared_next, w_target_next)
                batch_loss += ortho_penalty_shared

                ortho_penalty_flex = self._get_ortho_penalty_flex()
                batch_loss += ortho_penalty_flex

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                train_loss_source.append(batch_loss_source.detach())
                train_loss_target.append(batch_loss_target.detach())


            train_loss_source = torch.Tensor(train_loss_source).to(DEVICE)
            train_loss_target = torch.Tensor(train_loss_target).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    preds = self._forward(X_target_specific_val, X_target_shared_val, w_target_val, env='target')
                    val_loss = loss(preds, y_target_val)
                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            patience = 0
                        else:
                            patience += 1
                        if patience > self.patience and ((i + 1) * n_batches_target > self.n_iter_min):
                            break
                    if i % self.n_iter_print == 0:
                        log.info(
                            f"[{self.name}] Epoch: {i}, current {val_string} loss: {val_loss} train_loss_target: "
                            f"{torch.mean(train_loss_target)} train_loss_source: {torch.mean(train_loss_source)} "
                        )

        return self

    def _step(self, X_source_specific, X_source_shared, w_source, X_target_specific, X_target_shared, w_target):

        po_preds_source = self._forward(X_source_specific, X_source_shared, w_source, env='source')
        po_preds_target = self._forward(X_target_specific, X_target_shared, w_target, env='target')

        return po_preds_source, po_preds_target

    def _forward(self, X_specific, X_shared, w, env):
        X_shared = check_tensor(X_shared)
        X_shared_with_w = torch.cat((X_shared, w.reshape((-1, 1))), dim=1).to(DEVICE)

        repr_preds = self._shared_repr_estimator(X_specific, X_shared_with_w, env).squeeze()

        y_preds = self._po_estimator(repr_preds, env).squeeze()

        return y_preds

    def _get_ortho_penalty_shared(self, X_source_specific, X_source_shared, w_source,
                                  X_target_specific, X_target_shared, w_target):
        X_source_shared_with_w = torch.cat((X_source_shared, w_source.reshape((-1, 1))), dim=1)
        X_target_shared_with_w = torch.cat((X_target_shared, w_target.reshape((-1, 1))), dim=1)

        return self._shared_repr_estimator._get_loss_diff(X_source_specific, X_source_shared_with_w, env='source') + \
               self._shared_repr_estimator._get_loss_diff(X_target_specific, X_target_shared_with_w, env='target')

    def _get_ortho_penalty_flex(self):
        return self._po_estimator._ortho_penalty_asymmetric() + \
               self._po_estimator._ortho_penalty_asymmetric()

    def predict(self, X_specific, X_shared, return_po=False, env='target'):
        n = X_specific.shape[0]

        w_0 = torch.zeros((n, 1)).to(DEVICE)
        w_1 = torch.ones((n, 1)).to(DEVICE)

        preds_0 = self._forward(X_specific, X_shared, w_0, env)
        preds_1 = self._forward(X_specific, X_shared, w_1, env)

        y0_preds = preds_0
        y1_preds = preds_1

        outcome = y1_preds - y0_preds

        if return_po:
            return outcome, y0_preds, y1_preds

        return outcome

class HTCE_TLearner(BaseHTCE_Learner):
    def __init__(
        self,
        name: str,
        n_unit_in_shared: int,
        n_unit_in_source_specific: int,
        n_unit_in_target_specific: int,
        binary_y: bool = False,
        n_layers_r_shared: int = DEFAULT_LAYERS_R_SHARED,
        n_units_r_shared: int = DEFAULT_UNITS_R_SHARED,
        weight_decay: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        penalty_disc: float = 0,
        early_stopping: bool = True,
        prop_loss_multiplier: float = 1,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        patience: int = DEFAULT_PATIENCE,

    ) -> None:
        super(HTCE_TLearner, self).__init__()

        self._shared_repr_estimators = []
        self._po_estimators = []

        for idx in range(2):
            self._shared_repr_estimators.append(
                SharedRepresentationNet(
                    n_unit_in_shared=n_unit_in_shared, n_unit_in_source_specific=n_unit_in_source_specific,
                    n_unit_in_target_specific=n_unit_in_target_specific,
                    n_layers=n_layers_r_shared, n_units=n_units_r_shared, nonlin=nonlin)
            )
            self._po_estimators.append(
                FlexTENetHTCE(
                    f"{name}_po_estimator_{idx}",
                    n_units_r_shared * 2,
                    binary_y=binary_y,
                )
            )
        self.model_modules = [self._shared_repr_estimators[0], self._shared_repr_estimators[1],
                              self._po_estimators[0], self._po_estimators[1]]
        self.name = name
        self.val_split_prop = val_split_prop
        self.seed = seed
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.lr = lr
        self.weight_decay = weight_decay
        self.binary_y = binary_y
        self.penalty_disc = penalty_disc
        self.early_stopping = early_stopping
        self.prop_loss_multiplier = prop_loss_multiplier
        self.n_iter_min = n_iter_min
        self.patience = patience

    def _forward(self, X_specific, X_shared, env):
        repr0_preds = self._shared_repr_estimators[0](X_specific, X_shared, env).squeeze()
        y0_preds = self._po_estimators[0](repr0_preds, env).squeeze()

        repr1_preds = self._shared_repr_estimators[1](X_specific, X_shared, env).squeeze()
        y1_preds = self._po_estimators[1](repr1_preds, env).squeeze()

        return torch.vstack((y0_preds, y1_preds)).T

    def _get_ortho_penalty_shared(self, X_source_specific, X_source_shared,
                                  X_target_specific, X_target_shared):
        return self._shared_repr_estimators[0]._get_loss_diff(X_source_specific, X_source_shared, env='source') + \
               self._shared_repr_estimators[0]._get_loss_diff(X_target_specific, X_target_shared, env='target') + \
               self._shared_repr_estimators[1]._get_loss_diff(X_source_specific, X_source_shared, env='source') + \
               self._shared_repr_estimators[1]._get_loss_diff(X_target_specific, X_target_shared, env='target')

    def _get_ortho_penalty_flex(self):
        return self._po_estimators[0]._ortho_penalty_asymmetric() + \
               self._po_estimators[1]._ortho_penalty_asymmetric()




class HTCE_TARNet(BaseHTCE_Learner):
    def __init__(
        self,
        name: str,
        n_unit_in_shared: int,
        n_unit_in_source_specific: int,
        n_unit_in_target_specific: int,
        binary_y: bool = False,
        n_units_r: int = DEFAULT_UNITS_R,
        n_layers_r_shared: int = DEFAULT_LAYERS_R_SHARED,
        n_units_r_shared: int = DEFAULT_UNITS_R_SHARED,
        weight_decay: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        penalty_disc: float = 0,
        early_stopping: bool = True,
        prop_loss_multiplier: float = 1,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        patience: int = DEFAULT_PATIENCE,
    ) -> None:
        super(HTCE_TARNet, self).__init__()

        self._shared_repr_estimator = SharedRepresentationNet(
            n_unit_in_shared=n_unit_in_shared, n_unit_in_source_specific=n_unit_in_source_specific,
            n_unit_in_target_specific=n_unit_in_target_specific,
            n_layers=n_layers_r_shared, n_units=n_units_r_shared, nonlin=nonlin
        )

        self._repr_estimator_source = RepresentationNet(
            n_units_r * 2, n_units=n_units_r, n_layers=DEFAULT_LAYERS_R, nonlin=nonlin)

        self._repr_estimator_target = RepresentationNet(
            n_units_r * 2, n_units=n_units_r, n_layers=DEFAULT_LAYERS_R, nonlin=nonlin)


        self._po_estimators = []
        for idx in range(2):
            self._po_estimators.append(
                FlexTENetHTCE(
                    f"{name}_po_estimator_{idx}",
                    n_units_r,
                    binary_y=binary_y,
                    n_layers_r=1,
                    n_layers_out=2,
                )
            )

        self.model_modules = [self._shared_repr_estimator, self._repr_estimator_source, self._repr_estimator_target,
                              self._po_estimators[0], self._po_estimators[1]]

        self.name = name
        self.val_split_prop = val_split_prop
        self.seed = seed
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.lr = lr
        self.weight_decay = weight_decay
        self.binary_y = binary_y
        self.penalty_disc = penalty_disc
        self.early_stopping = early_stopping
        self.prop_loss_multiplier = prop_loss_multiplier
        self.n_iter_min = n_iter_min
        self.patience = patience

    def _forward(self, X_specific, X_shared, env):
        repr_preds_shared = self._shared_repr_estimator(X_specific, X_shared, env).squeeze()

        if env == 'source':
            repr_preds = self._repr_estimator_source(repr_preds_shared)
        else:
            repr_preds = self._repr_estimator_target(repr_preds_shared)

        y0_preds = self._po_estimators[0](repr_preds, env).squeeze()
        y1_preds = self._po_estimators[1](repr_preds, env).squeeze()

        return torch.vstack((y0_preds, y1_preds)).T

    def _get_ortho_penalty_shared(self, X_source_specific, X_source_shared,
                                  X_target_specific, X_target_shared):
        return self._shared_repr_estimator._get_loss_diff(X_source_specific, X_source_shared, env='source') + \
               self._shared_repr_estimator._get_loss_diff(X_target_specific, X_target_shared, env='target')

    def _get_ortho_penalty_flex(self):
        return self._po_estimators[0]._ortho_penalty_asymmetric() + \
               self._po_estimators[1]._ortho_penalty_asymmetric()
