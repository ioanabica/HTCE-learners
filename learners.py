import contrib.catenets.models as cate_models

from htce_learners.htce_one_step_learners import HTCE_TLearner, HTCE_SLearner, HTCE_TARNet

from htce_learners.htce_two_step_learners import HTCE_DRLearner


def get_flex_transfer_learner(name, X_shared_size, X_source_specific_size, X_target_specific_size, binary_Y, n_iter):
    transfer_learner = {
        "TLearner": HTCE_TLearner(
            "TLearner",
            n_unit_in_shared=X_shared_size,
            n_unit_in_source_specific=X_source_specific_size,
            n_unit_in_target_specific=X_target_specific_size,
            binary_y=binary_Y,
            batch_size=128,
            n_iter=n_iter,
            nonlin="relu",
        ),
        "SLearner": HTCE_SLearner(
            "SLearner",
            n_unit_in_shared=X_shared_size,
            n_unit_in_source_specific=X_source_specific_size,
            n_unit_in_target_specific=X_target_specific_size,
            binary_y=binary_Y,
            batch_size=128,
            n_iter=n_iter,
            nonlin="relu",
        ),
        "TARNet": HTCE_TARNet(
            "TARNet",
            n_unit_in_shared=X_shared_size,
            n_unit_in_source_specific=X_source_specific_size,
            n_unit_in_target_specific=X_target_specific_size,
            binary_y=binary_Y,
            n_units_r=100,
            batch_size=128,
            n_iter=n_iter,
            nonlin="relu",
        ),
        "DRLearner": HTCE_DRLearner(
            "DRLearner",
            n_unit_in_shared=X_shared_size,
            n_unit_in_source_specific=X_source_specific_size,
            n_unit_in_target_specific=X_target_specific_size,
            binary_y=binary_Y,
            batch_size=128,
            n_iter=n_iter,
            nonlin="relu",
        ),
    }

    return transfer_learner[name]


def get_learner(name, X_size, binary_Y, n_iter):
    learners = {
        "TLearner": cate_models.torch.TLearner(
            X_size,
            binary_y=binary_Y,
            n_layers_out=5,
            n_units_out=100,
            batch_size=128,
            n_iter=n_iter,
            batch_norm=False,
            nonlin="relu",
        ),
        "SLearner": cate_models.torch.SLearner(
            X_size,
            binary_y=binary_Y,
            n_layers_out=5,
            n_units_out=100,
            n_iter=n_iter,
            batch_size=128,
            batch_norm=False,
            nonlin="relu",
        ),
        "TARNet": cate_models.torch.TARNet(
            X_size,
            binary_y=binary_Y,
            n_layers_r=2,
            n_layers_out=3,
            n_units_out=100,
            n_units_r=100,
            batch_size=128,
            n_iter=n_iter,
            batch_norm=False,
            nonlin="relu",
        ),
        "DRLearner": cate_models.torch.DRLearner(
            X_size,
            binary_y=binary_Y,
            n_layers_out=5,
            n_units_out=100,
            batch_size=128,
            n_iter=n_iter,
            batch_norm=False,
            nonlin="relu",
        ),
        "PWLearner": cate_models.torch.PWLearner(
            X_size,
            binary_y=binary_Y,
            n_layers_out=5,
            n_units_out=100,
            batch_size=128,
            n_iter=n_iter,
            batch_norm=False,
            nonlin="relu",
        ),
        "RALearner": cate_models.torch.RALearner(
            X_size,
            binary_y=binary_Y,
            n_layers_out=5,
            n_units_out=100,
            batch_size=128,
            n_iter=n_iter,
            batch_norm=False,
            nonlin="relu",
        ),
    }

    return learners[name]
