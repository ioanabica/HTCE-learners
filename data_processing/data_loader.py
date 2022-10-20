import numpy as np

from contrib.catenets.datasets import load as catenets_load


def sample_source_and_target_datasets(
    X,
    seed,
    num_examples_source=None,
    num_examples_target=None,
    num_features_source=None,
    num_features_shared=None,
    num_features_target=None,
):
    np.random.seed(seed)

    if num_examples_target == None:
        num_examples_target = np.random.randint(100, 500)

    if num_examples_source == None:
        num_examples_source = np.random.randint(1000, X.shape[0] - num_examples_target)

    if num_features_shared == None:
        num_features_shared = np.random.randint(5, int(X.shape[1] / 3))
        num_features_source = np.random.randint(5, int(X.shape[1] / 3))
        num_features_target = np.random.randint(5, int(X.shape[1] / 3))
    elif num_features_shared != None:
        num_features_source = np.random.randint(1, int((X.shape[1] - num_features_shared) / 2))
        num_features_target = np.random.randint(1, int((X.shape[1] - num_features_shared) / 2))

    assert num_examples_source + num_examples_target < X.shape[0]
    assert num_features_source + num_features_shared + num_features_target < X.shape[1]

    all_indices_features = np.array(range(X.shape[1]))
    np.random.shuffle(all_indices_features)

    all_indices_examples = np.array(range(X.shape[0]))
    np.random.shuffle(all_indices_examples)

    X_source_specific = X[all_indices_examples[:num_examples_source]][:, all_indices_features[:num_features_source]]

    X_source_shared = X[all_indices_examples[:num_examples_source]][
        :, all_indices_features[num_features_source : num_features_source + num_features_shared]
    ]

    X_target_shared = X[all_indices_examples[num_examples_source : num_examples_source + num_examples_target]][
        :, all_indices_features[num_features_source : num_features_source + num_features_shared]
    ]

    X_target_specific = X[all_indices_examples[num_examples_source : num_examples_source + num_examples_target]][
        :,
        all_indices_features[
            num_features_source + num_features_shared : num_features_source + num_features_shared + num_features_target
        ],
    ]

    return X_source_specific, X_source_shared, X_target_specific, X_target_shared


def load(
    dataset_name,
    seed,
    train_ratio=1.0,
    num_examples_source=None,
    num_examples_target=None,
    num_features_source=None,
    num_features_shared=None,
    num_features_target=None,
    return_full_data=False,
):
    if "twins" in dataset_name:
        X_raw, _, _, _, _, _ = catenets_load(dataset_name, train_ratio=1.0)
        if return_full_data:
            return X_raw
        X_source_specific, X_source_shared, X_target_specific, X_target_shared = sample_source_and_target_datasets(
            X_raw,
            seed,
            num_examples_source,
            num_examples_target,
            num_features_source,
            num_features_shared,
            num_features_target,
        )
    else:
        print("Unknown dataset " + str(dataset_name))

    if train_ratio == 1.0:
        return (X_source_specific, X_source_shared, X_target_specific, X_target_shared)
    else:

        X_source_specific_train = X_source_specific[: int(train_ratio * X_source_specific.shape[0])]
        X_source_specific_test = X_source_specific[int(train_ratio * X_source_specific.shape[0]) :]

        X_source_shared_train = X_source_shared[: int(train_ratio * X_source_shared.shape[0])]
        X_source_shared_test = X_source_shared[int(train_ratio * X_source_shared.shape[0]) :]

        X_target_specific_train = X_target_specific[: int(train_ratio * X_target_specific.shape[0])]
        X_target_specific_test = X_target_specific[int(train_ratio * X_target_specific.shape[0]) :]

        X_target_shared_train = X_target_shared[: int(train_ratio * X_target_shared.shape[0])]
        X_target_shared_test = X_target_shared[int(train_ratio * X_target_shared.shape[0]) :]

        return (X_source_specific_train, X_source_shared_train, X_target_specific_train, X_target_shared_train), (
            X_source_specific_test,
            X_source_shared_test,
            X_target_specific_test,
            X_target_shared_test,
        )
