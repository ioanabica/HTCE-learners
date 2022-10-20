from typing import Any, Callable, List

import numpy as np
import torch
from torch import nn

import contrib.catenets.logger as log
from htce_learners.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIM_P_OUT,
    DEFAULT_DIM_P_R,
    DEFAULT_DIM_S_OUT,
    DEFAULT_DIM_S_R,
    DEFAULT_LAYERS_OUT,
    DEFAULT_LAYERS_R,
    DEFAULT_LAYERS_R_SHARED,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_MIN,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_NONLIN,
    DEFAULT_PATIENCE,
    DEFAULT_PENALTY_L2,
    DEFAULT_PENALTY_ORTHOGONAL,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_UNITS_R,
    DEFAULT_UNITS_R_SHARED,
    DEFAULT_VAL_SPLIT,
    LARGE_VAL,
)
from utils import check_tensor, make_target_val_split

EPS = 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NONLIN = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
}


class FlexTESplitLayer(nn.Module):
    """
    Code adapted from: https://github.com/AliciaCurth/CATENets.
    Create multitask layer has shape [shared, private_0, private_1]
    """

    def __init__(
        self,
        name: str,
        n_units_in: int,
        n_units_in_p: int,
        n_units_s: int,
        n_units_p: int,
        first_layer: bool,
    ) -> None:
        super(FlexTESplitLayer, self).__init__()
        self.name = name
        self.first_layer = first_layer
        self.n_units_in = n_units_in
        self.n_units_in_p = n_units_in_p
        self.n_units_s = n_units_s
        self.n_units_p = n_units_p

        self.net_shared = nn.Sequential(nn.Linear(n_units_in, n_units_s)).to(DEVICE)
        self.net_psource = nn.Sequential(nn.Linear(n_units_in_p, n_units_p)).to(DEVICE)
        self.net_ptarget = nn.Sequential(nn.Linear(n_units_in_p, n_units_p)).to(DEVICE)

    def forward(self, tensors: List[torch.Tensor]) -> List:
        if self.first_layer and len(tensors) != 2:
            raise ValueError("Invalid number of tensor for the FlexSplitLayer layer. It requires the features vector.")
        if not self.first_layer and len(tensors) != 4:
            raise ValueError(
                "Invalid number of tensor for the FlexSplitLayer layer. It requires X_s, X_psource, X_ptarget, env as input."
            )

        if self.first_layer:
            X = tensors[0]
            rep_s = self.net_shared(X)
            rep_psource = self.net_psource(X)
            rep_ptarget = self.net_ptarget(X)
            env = tensors[1]

        else:
            X_s = tensors[0]
            X_psource = tensors[1]
            X_ptarget = tensors[2]
            env = tensors[3]

            rep_s = self.net_shared(X_s)
            rep_psource = self.net_psource(torch.cat([X_s, X_psource], dim=1))
            rep_ptarget = self.net_ptarget(torch.cat([X_s, X_ptarget], dim=1))

        return [rep_s, rep_psource, rep_ptarget, env]


class FlexTEOutputLayer(nn.Module):
    """
    Code adapted from: https://github.com/AliciaCurth/CATENets.
    """

    def __init__(self, n_units_in: int, n_units_in_p: int, private: bool) -> None:
        super(FlexTEOutputLayer, self).__init__()
        self.private = private
        self.net_shared = nn.Sequential(nn.Linear(n_units_in, 1)).to(DEVICE)
        self.net_psource = nn.Sequential(nn.Linear(n_units_in_p, 1)).to(DEVICE)
        self.net_ptarget = nn.Sequential(nn.Linear(n_units_in_p, 1)).to(DEVICE)

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        if len(tensors) != 4:
            raise ValueError(
                "Invalid number of tensor for the FlexSplitLayer layer. It requires X_s, X_psource, X_ptarget, env."
            )
        X_s = tensors[0]
        X_psource = tensors[1]
        X_ptarget = tensors[2]
        env = tensors[3]

        if self.private:
            rep_psource = self.net_psource(torch.cat([X_s, X_psource], dim=1)).squeeze()
            rep_ptarget = self.net_ptarget(torch.cat([X_s, X_ptarget], dim=1)).squeeze()

            if env == "source":
                return rep_psource
            elif env == "target":
                return rep_ptarget
            else:
                raise Exception("Unknown env.")
        else:
            rep_s = self.net_shared(X_s).squeeze()
            rep_psource = self.net_psource(torch.cat([X_s, X_psource], dim=1)).squeeze()
            rep_ptarget = self.net_ptarget(torch.cat([X_s, X_ptarget], dim=1)).squeeze()

            # These could also be concatenated instead of added
            if env == "source":
                return rep_psource + rep_s
            elif env == "target":
                return rep_ptarget + rep_s
            else:
                raise Exception("Unknown env.")


class ElementWiseParallelActivation(nn.Module):
    """
    Code adapted from: https://github.com/AliciaCurth/CATENets.
    Layer that applies a scalar function elementwise on its inputs.

    Input looks like: X_s, X_p0, X_p1 = inputs
    """

    def __init__(self, act: Callable, **act_kwargs: Any) -> None:
        super(ElementWiseParallelActivation, self).__init__()
        self.act = act
        self.act_kwargs = act_kwargs

    def forward(self, tensors: List[torch.Tensor]) -> List:
        if len(tensors) != 4:
            raise ValueError(
                "Invalid number of tensor for the ElementWiseParallelActivation layer. It requires X_s, X_psource, X_ptarget, env as input"
            )

        return [
            self.act(tensors[0], **self.act_kwargs),
            self.act(tensors[1], **self.act_kwargs),
            self.act(tensors[2], **self.act_kwargs),
            tensors[3],
        ]


class ElementWiseSplitActivation(nn.Module):
    """
    Code adapted from: https://github.com/AliciaCurth/CATENets.
    Layer that applies a scalar function elementwise on its inputs.

    Input looks like: X = inputs
    """

    def __init__(self, act: Callable, **act_kwargs: Any) -> None:
        super(ElementWiseSplitActivation, self).__init__()
        self.act = act
        self.act_kwargs = act_kwargs

    def forward(self, tensors: List[torch.Tensor]) -> List:
        if len(tensors) != 1:
            raise ValueError(
                "Invalid number of tensor for the ElementWiseSplitActivation layer. It requires X as input"
            )

        return [
            self.act(tensors[0], **self.act_kwargs),
        ]


class FlexTENetHTCE(nn.Module):
    """
    Code adapted from: https://github.com/AliciaCurth/CATENets.
    """

    def __init__(
        self,
        name: str,
        n_unit_in: int,
        binary_y: bool,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_s_out: int = DEFAULT_DIM_S_OUT,
        n_units_p_out: int = DEFAULT_DIM_P_OUT,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_s_r: int = DEFAULT_DIM_S_R,
        n_units_p_r: int = DEFAULT_DIM_P_R,
        private_out: bool = False,
        weight_decay: float = DEFAULT_PENALTY_L2,
        penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        early_stopping: bool = True,
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        normalize_ortho: bool = False,
        mode: int = 1,
        clipping_value: int = 1,
    ) -> None:
        super(FlexTENetHTCE, self).__init__()

        self.name = name

        self.binary_y = binary_y
        self.n_layers_r = n_layers_r if n_layers_r else 1
        self.n_layers_out = n_layers_out
        self.n_units_s_out = n_units_s_out
        self.n_units_p_out = n_units_p_out
        self.n_units_s_r = n_units_s_r
        self.n_units_p_r = n_units_p_r
        self.private_out = private_out
        self.mode = mode

        self.penalty_orthogonal = penalty_orthogonal
        self.weight_decay = weight_decay
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_split_prop = val_split_prop
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.normalize_ortho = normalize_ortho
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping

        self.seed = seed
        self.n_iter_print = n_iter_print

        layers = []

        layers.extend(
            [
                FlexTESplitLayer(
                    f"{self.name}_shared_private_layer_0",
                    n_unit_in,
                    n_unit_in,
                    n_units_s_r,
                    n_units_p_r,
                    first_layer=True,
                ),
                ElementWiseParallelActivation(nn.SELU(inplace=True)),
            ]
        )
        # These representation layers can probably be removed
        # add required number of layers
        for i in range(n_layers_r - 1):
            layers.extend(
                [
                    FlexTESplitLayer(
                        f"{self.name}_shared_private_layer_{i + 1}",
                        n_units_s_r,
                        n_units_s_r + n_units_p_r,
                        n_units_s_r,
                        n_units_p_r,
                        first_layer=False,
                    ),
                    ElementWiseParallelActivation(nn.SELU(inplace=True)),
                ]
            )
        # add output layers
        layers.extend(
            [
                FlexTESplitLayer(
                    f"{self.name}_output_layer_0",
                    n_units_s_r,
                    n_units_s_r + n_units_p_r,
                    n_units_s_out,
                    n_units_p_out,
                    first_layer=False,
                ),
                ElementWiseParallelActivation(nn.SELU(inplace=True)),
            ]
        )

        # add required number of layers
        for i in range(n_layers_out - 1):
            layers.extend(
                [
                    FlexTESplitLayer(
                        f"{self.name}_output_layer_{i + 1}",
                        n_units_s_out,
                        n_units_s_out + n_units_p_out,
                        n_units_s_out,
                        n_units_p_out,
                        first_layer=False,
                    ),
                    ElementWiseParallelActivation(nn.SELU(inplace=True)),
                ]
            )

        # append final layer
        layers.append(FlexTEOutputLayer(n_units_s_out, n_units_s_out + n_units_p_out, private=self.private_out))
        if binary_y:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers).to(DEVICE)

    def _ortho_penalty_asymmetric(self) -> torch.Tensor:
        def _get_cos_reg(params_source: torch.Tensor, params_target: torch.Tensor, normalize: bool) -> torch.Tensor:
            if normalize:
                params_source = params_source / torch.linalg.norm(params_source, dim=0)
                params_target = params_target / torch.linalg.norm(params_target, dim=0)

            x_min = min(params_source.shape[0], params_target.shape[0])
            y_min = min(params_source.shape[1], params_target.shape[1])

            return (
                torch.linalg.norm(
                    torch.matmul(torch.transpose(params_source[:x_min, :y_min], 0, 1), params_target[:x_min, :y_min]),
                    "fro",
                )
                ** 2
            )

        def _apply_reg_split_layer(layer: FlexTESplitLayer, full: bool = True) -> torch.Tensor:
            _ortho_body = 0
            if full:
                _ortho_body = _get_cos_reg(
                    layer.net_psource[-1].weight,
                    layer.net_ptarget[-1].weight,
                    self.normalize_ortho,
                )
            _ortho_body += torch.sum(
                _get_cos_reg(
                    layer.net_shared[-1].weight,
                    layer.net_psource[-1].weight,
                    self.normalize_ortho,
                )
                + _get_cos_reg(
                    layer.net_shared[-1].weight,
                    layer.net_ptarget[-1].weight,
                    self.normalize_ortho,
                )
            )
            return _ortho_body

        ortho_body = 0
        for layer in self.model:
            if not isinstance(layer, (FlexTESplitLayer, FlexTEOutputLayer)):
                continue

            if isinstance(layer, FlexTESplitLayer):
                ortho_body += _apply_reg_split_layer(layer, full=True)

            if self.private_out:
                continue

            ortho_body += _apply_reg_split_layer(layer, full=False)

        return self.penalty_orthogonal * ortho_body

    def forward(self, X: torch.Tensor, env: str) -> torch.Tensor:
        X = check_tensor(X).float()
        mu = self.model([X, env])
        return mu

    def predict(self, X: torch.Tensor, return_po: bool = False) -> torch.Tensor:
        """
        Predict treatment effects and potential outcomes

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        Returns
        -------
        y: array-like of shape (n_samples,)
        """

        X = check_tensor(X).float()
        W0 = torch.zeros(X.shape[0]).to(DEVICE)
        W1 = torch.ones(X.shape[0]).to(DEVICE)

        mu0 = self.model([X, W0])
        mu1 = self.model([X, W1])

        te = mu1 - mu0

        if return_po:
            return te, mu0, mu1

        return te


class HTCEBaseEstimator(nn.Module):
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
        early_stopping: bool = True,
        prop_loss_multiplier: float = 1,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        patience: int = DEFAULT_PATIENCE,
        clipping_value: int = 1,
    ) -> None:
        super(HTCEBaseEstimator, self).__init__()

        self._shared_repr_estimator = SharedRepresentationNet(
            n_unit_in_shared=n_unit_in_shared,
            n_unit_in_source_specific=n_unit_in_source_specific,
            n_unit_in_target_specific=n_unit_in_target_specific,
            n_layers=n_layers_r_shared,
            n_units=n_units_r_shared,
            nonlin=nonlin,
        )

        self._estimator = FlexTENetHTCE(
            f"{name}_estimator",
            n_units_r_shared * 2,
            binary_y=binary_y,
        )

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
        self.early_stopping = early_stopping
        self.prop_loss_multiplier = prop_loss_multiplier
        self.n_iter_min = n_iter_min
        self.patience = patience
        self.clipping_value = clipping_value

    def train(self, X_source_specific, X_source_shared, X_target_specific, X_target_shared, y_source, y_target):

        X_source_specific = torch.Tensor(X_source_specific).to(DEVICE)
        X_source_shared = torch.Tensor(X_source_shared).to(DEVICE)
        X_target_specific = torch.Tensor(X_target_specific).to(DEVICE)
        X_target_shared = torch.Tensor(X_target_shared).to(DEVICE)

        y_source = torch.Tensor(y_source).squeeze().to(DEVICE)
        y_target = torch.Tensor(y_target).squeeze().to(DEVICE)

        (
            X_target_specific,
            X_target_shared,
            y_target,
            X_target_specific_val,
            X_target_shared_val,
            y_target_val,
            val_string,
        ) = make_target_val_split(
            X_target_specific=X_target_specific,
            X_target_shared=X_target_shared,
            y=y_target,
            w=None,
            val_split_prop=self.val_split_prop,
            seed=self.seed,
        )

        n_target = X_target_specific.shape[0]  # could be different from before due to split
        n_source = X_source_specific.shape[0]

        # calculate number of batches per epoch
        batch_size = self.batch_size if self.batch_size < n_target else n_target
        n_batches_target = int(np.round(n_target / batch_size)) if batch_size < n_target else 1
        train_indices_target = np.arange(n_target)

        n_batches_source = int(np.round(n_source / batch_size)) if batch_size < n_source else 1
        train_indices_source = np.arange(n_source)

        params = list(self._shared_repr_estimator.parameters()) + list(self._estimator.parameters())

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
                    (b_target * batch_size) : min((b_target + 1) * batch_size, n_target - 1)
                ]

                idx_next_source = train_indices_source[
                    (b_source * batch_size) : min((b_source + 1) * batch_size, n_source - 1)
                ]
                b_source = (b_source + 1) % n_batches_source

                X_target_specific_next = X_target_specific[idx_next_target]
                X_target_shared_next = X_target_shared[idx_next_target]
                y_target_next = y_target[idx_next_target].squeeze()

                X_source_specific_next = X_source_specific[idx_next_source]
                X_source_shared_next = X_source_shared[idx_next_source]
                y_source_next = y_source[idx_next_source].squeeze()

                po_preds_source, po_preds_target = self._step(
                    X_source_specific_next, X_source_shared_next, X_target_specific_next, X_target_shared_next
                )

                batch_loss_source = self.loss(po_preds_source, y_source_next)
                batch_loss_target = self.loss(po_preds_target, y_target_next)

                batch_loss = batch_loss_source + batch_loss_target

                ortho_penalty_shared = self._get_ortho_penalty_shared(
                    X_source_specific_next, X_source_shared_next, X_target_specific_next, X_target_shared_next
                )
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
                    preds = self._forward(X_target_specific_val, X_target_shared_val, env="target")
                    val_loss = self.loss(preds, y_target_val)
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
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        def head_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            if self.binary_y:
                return nn.BCELoss()(y_pred, y_true)
            else:
                return nn.MSELoss()(y_pred, y_true)

        return head_loss(y_pred, y_true)

    def _step(self, X_source_specific, X_source_shared, X_target_specific, X_target_shared):

        preds_source = self._forward(X_source_specific, X_source_shared, env="source")
        preds_target = self._forward(X_target_specific, X_target_shared, env="target")

        return preds_source, preds_target

    def _forward(self, X_specific, X_shared, env):
        repr_preds = self._shared_repr_estimator(X_specific, X_shared, env).squeeze()
        preds = self._estimator(repr_preds, env).squeeze()

        return preds

    def _get_ortho_penalty_shared(self, X_source_specific, X_source_shared, X_target_specific, X_target_shared):
        return self._shared_repr_estimator._get_loss_diff(
            X_source_specific, X_source_shared, env="source"
        ) + self._shared_repr_estimator._get_loss_diff(X_target_specific, X_target_shared, env="target")

    def _get_ortho_penalty_flex(self):
        return self._estimator._ortho_penalty_asymmetric() + self._estimator._ortho_penalty_asymmetric()

    def predict(self, X_specific, X_shared, env="target"):
        preds = self._forward(X_specific, X_shared, env)
        return preds


class RepresentationNet(nn.Module):
    def __init__(
        self,
        n_unit_in: int,
        n_layers: int = DEFAULT_LAYERS_R,
        n_units: int = DEFAULT_UNITS_R,
        nonlin: str = DEFAULT_NONLIN,
    ) -> None:

        super(RepresentationNet, self).__init__()
        if nonlin not in list(NONLIN.keys()):
            raise ValueError("Unknown nonlinearity")

        NL = NONLIN[nonlin]
        layers = [nn.Linear(n_unit_in, n_units), NL()]

        # add required number of layers
        for i in range(n_layers - 1):
            layers.extend([nn.Linear(n_units, n_units), NL()])

        self.model = nn.Sequential(*layers).to(DEVICE)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


class SharedRepresentationNet(nn.Module):
    def __init__(
        self,
        n_unit_in_shared: int,
        n_unit_in_source_specific: int,
        n_unit_in_target_specific: int,
        n_layers: int = DEFAULT_LAYERS_R,
        n_units: int = DEFAULT_UNITS_R,
        nonlin: str = DEFAULT_NONLIN,
    ) -> None:
        super(SharedRepresentationNet, self).__init__()
        if nonlin not in list(NONLIN.keys()):
            raise ValueError("Unknown nonlinearity")

        self.shared_rep_model = RepresentationNet(
            n_unit_in=n_unit_in_shared, n_units=n_units, n_layers=n_layers, nonlin=nonlin
        )
        self.source_specific_rep_model = RepresentationNet(
            n_unit_in=n_unit_in_source_specific, n_units=n_units, n_layers=n_layers, nonlin=nonlin
        )
        self.target_specific_rep_model = RepresentationNet(
            n_unit_in=n_unit_in_target_specific, n_units=n_units, n_layers=n_layers, nonlin=nonlin
        )

    def forward(self, X_specific, X_shared, env) -> torch.Tensor:
        X_specific = self._check_tensor(X_specific)
        X_shared = self._check_tensor(X_shared)
        shared_rep = self.shared_rep_model(X_shared)

        if env == "source":
            specific_rep = self.source_specific_rep_model(X_specific)
        elif env == "target":
            specific_rep = self.target_specific_rep_model(X_specific)
        else:
            raise Exception("Unknown env.")

        return torch.cat([shared_rep, specific_rep], dim=1)

    def _get_loss_diff(self, X_specific, X_shared, env, normalize=True):
        X_specific = self._check_tensor(X_specific)
        X_shared = self._check_tensor(X_shared)
        shared_rep = self.shared_rep_model(X_shared)

        if env == "source":
            specific_rep = self.source_specific_rep_model(X_specific)
        elif env == "target":
            specific_rep = self.target_specific_rep_model(X_specific)
        else:
            raise Exception("Unknown env.")

        if normalize:
            specific_rep = specific_rep / torch.linalg.norm(specific_rep)
            shared_rep = shared_rep / torch.linalg.norm(shared_rep)

        loss_scale = 0.01

        return loss_scale * torch.linalg.norm(torch.matmul(torch.transpose(specific_rep, 0, 1), shared_rep), "fro") ** 2

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.float().to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).float().to(DEVICE)
