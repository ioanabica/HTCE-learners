from pathlib import Path

import numpy as np
import pandas as pd

import logger as log
from contrib.radial_gan.RadialGAN_treatment import RadialGAN_Treatment
from data_processing.data_loader import load
from simulate_dataset import SyntheticSimulatorLinear
from learners import get_learner, get_flex_transfer_learner
from utils import compute_pehe


class BaselineExperiment:
    def __init__(
        self,
        n_iter: int = 1000,
        seed: int = 42,
        save_path: Path = Path.cwd(),
        synthetic_simulator_type: str = "linear",
    ) -> None:
        self.n_iter = n_iter
        self.seed = seed
        self.save_path = save_path
        self.synthetic_simulator_type = synthetic_simulator_type

    def run_benchmarks_new(
        self,
        learner_names_list,
        methods_list,
        X_source_shared_train,
        X_target_shared_train,
        X_source_specific_train,
        X_target_specific_train,
        W_source_train,
        W_target_train,
        Y_source_train,
        Y_target_train,
        X_target_specific_test,
        X_target_shared_test,
        cate_test,
    ):
        experiment_data = []

        for method in methods_list:
            if method == "target":
                X_train = np.concatenate((X_target_specific_train, X_target_shared_train), axis=1)
                W_train = W_target_train
                Y_train = Y_target_train

                for learner_name in learner_names_list:
                    log.info(f"Fitting learner {learner_name} with method {method}.")
                    learner = get_learner(
                        learner_name,
                        X_size=X_train.shape[1],
                        binary_Y=(len(np.unique(Y_source_train)) == 2),
                        n_iter=self.n_iter,
                    )

                    learner.fit(X=X_train, y=Y_train, w=W_train)

                    X_target_test = np.concatenate((X_target_specific_test, X_target_shared_test), axis=1)
                    cate_pred = learner.predict(X=X_target_test)

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)
                    experiment_data.append(
                        [
                            learner_name,
                            method,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

            elif method == "shared_source_target":
                X_train = np.concatenate((X_source_shared_train, X_target_shared_train), axis=0)
                W_train = np.concatenate((W_source_train, W_target_train), axis=0)
                Y_train = np.concatenate((Y_source_train, Y_target_train), axis=0)

                for learner_name in learner_names_list:
                    log.info(f"Fitting learner {learner_name} with method {method}.")
                    learner = get_learner(
                        learner_name,
                        X_size=X_train.shape[1],
                        binary_Y=(len(np.unique(Y_source_train)) == 2),
                        n_iter=self.n_iter,
                    )
                    learner.fit(X=X_train, y=Y_train, w=W_train)

                    X_target_test = X_target_shared_test
                    cate_pred = learner.predict(X=X_target_test)

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)
                    experiment_data.append(
                        [
                            learner_name,
                            method,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

            elif method == "radial_gan":
                X_source_train = np.concatenate((X_source_specific_train, X_source_shared_train), axis=1)
                X_target_train = np.concatenate((X_target_specific_train, X_target_shared_train), axis=1)

                Train_X = (X_source_train, X_target_train)
                Train_W = (W_source_train, W_target_train)
                Train_Y = (Y_source_train, Y_target_train)

                new_X_train_target = RadialGAN_Treatment(Train_X=Train_X, Train_T=Train_W, Train_Y=Train_Y, alpha=1.0)

                X_train = np.concatenate((X_target_train, new_X_train_target), axis=0)
                W_train = np.concatenate((W_target_train, W_source_train), axis=0)
                Y_train = np.concatenate((Y_target_train, Y_source_train), axis=0)

                for learner_name in learner_names_list:
                    log.info(f"Fitting learner {learner_name} with method {method}.")
                    learner = get_learner(
                        learner_name,
                        X_size=X_train.shape[1],
                        binary_Y=(len(np.unique(Y_source_train)) == 2),
                        n_iter=self.n_iter,
                    )

                    learner.fit(X=X_train, y=Y_train, w=W_train)

                    X_target_test = np.concatenate((X_target_specific_test, X_target_shared_test), axis=1)
                    cate_pred = learner.predict(X=X_target_test)

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)
                    experiment_data.append(
                        [
                            learner_name,
                            method,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

            elif method == "htce":
                for learner_name in learner_names_list:
                    log.info(f"Fitting learner {learner_name} with method {method}.")
                    learner = get_flex_transfer_learner(
                        learner_name,
                        X_shared_size=X_source_shared_train.shape[1],
                        X_source_specific_size=X_source_specific_train.shape[1],
                        X_target_specific_size=X_target_specific_train.shape[1],
                        binary_Y=(len(np.unique(Y_source_train)) == 2),
                        n_iter=self.n_iter,
                    )

                    learner.train(
                        X_source_specific=X_source_specific_train,
                        X_source_shared=X_source_shared_train,
                        X_target_specific=X_target_specific_train,
                        X_target_shared=X_target_shared_train,
                        y_source=Y_source_train,
                        y_target=Y_target_train,
                        w_source=W_source_train,
                        w_target=W_target_train,
                    )

                    cate_pred = learner.predict(X_specific=X_target_specific_test, X_shared=X_target_shared_test)
                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)
                    experiment_data.append(
                        [
                            learner_name,
                            method,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )
            else:
                raise Exception("Unknown method type.")

        return experiment_data

    def run(
        self,
        dataset="tcga",
        train_ratio=0.8,
        binary_outcome=False,
        learner_names_list=["TLearner"],
        methods_list=["target"],
        experiment_id=0,
    ) -> None:
        log.info(f"Using dataset {dataset} with learners = {learner_names_list} and methods = {methods_list}.")

        X_train, X_test = load(dataset, self.seed, train_ratio=train_ratio)

        (X_source_specific_train, X_source_shared_train, X_target_specific_train, X_target_shared_train) = X_train
        (X_source_specific_test, X_source_shared_test, X_target_specific_test, X_target_shared_test) = X_test

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_source_specific_train,
                X_source_shared_train,
                X_target_specific_train,
                X_target_shared_train,
                seed=self.seed,
            )
        else:
            raise Exception("Unknown simulator type.")

        W_source_train, Y_source_train, _, W_target_train, Y_target_train, _ = sim.simulate_dataset(
            X_source_specific_train,
            X_source_shared_train,
            X_target_specific_train,
            X_target_shared_train,
            binary_outcome=binary_outcome,
            treatment_assign="pred",
        )
        W_source_test, Y_source_test, _, W_target_test, Y_target_test, PO_target_test = sim.simulate_dataset(
            X_source_specific_test,
            X_source_shared_test,
            X_target_specific_test,
            X_target_shared_test,
            binary_outcome=binary_outcome,
            treatment_assign="pred",
        )

        cate_test = PO_target_test[1] - PO_target_test[0]

        experiment_data = self.run_benchmarks_new(
            learner_names_list,
            methods_list,
            X_source_shared_train,
            X_target_shared_train,
            X_source_specific_train,
            X_target_specific_train,
            W_source_train,
            W_target_train,
            Y_source_train,
            Y_target_train,
            X_target_specific_test,
            X_target_shared_test,
            cate_test,
        )

        metrics_df = pd.DataFrame(
            experiment_data,
            columns=[
                "Learner",
                "Method",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        dir_name = "results/baseline_experiment/experiment_id_" + str(experiment_id)

        results_path = self.save_path / dir_name
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"dataset_{dataset}_"
            f"{self.synthetic_simulator_type}_"
            f"binary_{binary_outcome}-seed{self.seed}.csv"
        )


class POSharingExperiment(BaselineExperiment):
    def __init__(
        self,
        n_iter: int = 1000,
        seed: int = 42,
        po_sharing_scales_list: list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        sharing_across_domains=True,
        save_path: Path = Path.cwd(),
        synthetic_simulator_type: str = "linear",
    ) -> None:
        super(BaselineExperiment, self).__init__()

        self.n_iter = n_iter
        self.seed = seed
        self.save_path = save_path
        self.synthetic_simulator_type = synthetic_simulator_type

        self.po_sharing_scales_list = po_sharing_scales_list
        self.sharing_across_domains = sharing_across_domains

    def run(
        self,
        dataset="tcga",
        train_ratio=0.8,
        binary_outcome=False,
        learner_names_list=["TLearner"],
        methods_list=["target"],
        experiment_id=0,
    ) -> None:
        log.info(f"Using dataset {dataset} with learners = {learner_names_list} and methods = {methods_list}.")

        X_train, X_test = load(dataset, self.seed, train_ratio=train_ratio)

        (X_source_specific_train, X_source_shared_train, X_target_specific_train, X_target_shared_train) = X_train
        (X_source_specific_test, X_source_shared_test, X_target_specific_test, X_target_shared_test) = X_test

        all_metrics_df = []

        for po_sharing_scale in self.po_sharing_scales_list:
            log.info(f"Now working with alpha = {po_sharing_scale}...")

            if self.synthetic_simulator_type == "linear":
                if self.sharing_across_domains:
                    sim = SyntheticSimulatorLinear(
                        X_source_specific_train,
                        X_source_shared_train,
                        X_target_specific_train,
                        X_target_shared_train,
                        alpha=po_sharing_scale,
                        seed=self.seed,
                    )
                else:
                    sim = SyntheticSimulatorLinear(
                        X_source_specific_train,
                        X_source_shared_train,
                        X_target_specific_train,
                        X_target_shared_train,
                        beta=po_sharing_scale,
                        seed=self.seed,
                    )
            else:
                raise Exception("Unknown simulator type.")

            W_source_train, Y_source_train, _, W_target_train, Y_target_train, _ = sim.simulate_dataset(
                X_source_specific_train,
                X_source_shared_train,
                X_target_specific_train,
                X_target_shared_train,
                binary_outcome=binary_outcome,
                treatment_assign="pred",
            )
            W_source_test, Y_source_test, _, W_target_test, Y_target_test, PO_target_test = sim.simulate_dataset(
                X_source_specific_test,
                X_source_shared_test,
                X_target_specific_test,
                X_target_shared_test,
                binary_outcome=binary_outcome,
                treatment_assign="pred",
            )

            cate_test = PO_target_test[1] - PO_target_test[0]

            experiment_data = self.run_benchmarks_new(
                learner_names_list,
                methods_list,
                X_source_shared_train,
                X_target_shared_train,
                X_source_specific_train,
                X_target_specific_train,
                W_source_train,
                W_target_train,
                Y_source_train,
                Y_target_train,
                X_target_specific_test,
                X_target_shared_test,
                cate_test,
            )

            metrics_df = pd.DataFrame(
                experiment_data,
                columns=[
                    "Learner",
                    "Method",
                    "PEHE",
                    "CATE true mean",
                    "CATE true var",
                    "Normalized PEHE",
                ],
            )
            metrics_df["PO Sharing Scale"] = po_sharing_scale
            all_metrics_df.append(metrics_df)

        all_metrics_df_all = pd.concat(all_metrics_df)
        print("all_metrics_df_all")
        print(all_metrics_df_all)
        if self.sharing_across_domains:
            dir_name = "results/po_sharing_across_domains/experiment_id_" + str(experiment_id)
        else:
            dir_name = "results/po_sharing_within_domain/experiment_id_" + str(experiment_id)

        results_path = self.save_path / dir_name
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        all_metrics_df_all.to_csv(
            results_path / f"dataset_{dataset}_"
            f"{self.synthetic_simulator_type}_"
            f"binary_{binary_outcome}-seed{self.seed}.csv"
        )


class TargetDatasetSizeExperiment(BaselineExperiment):
    def __init__(
        self,
        n_iter: int = 1000,
        seed: int = 42,
        target_dataset_size_list: list = [50, 100, 200, 300, 500, 1000, 2000, 4000, 6000, 8000],
        save_path: Path = Path.cwd(),
        synthetic_simulator_type: str = "linear",
    ) -> None:
        super(BaselineExperiment, self).__init__()

        self.n_iter = n_iter
        self.seed = seed
        self.save_path = save_path
        self.synthetic_simulator_type = synthetic_simulator_type

        self.target_dataset_size_list = target_dataset_size_list

    def run(
        self,
        dataset="tcga",
        train_ratio=0.8,
        binary_outcome=False,
        learner_names_list=["TLearner"],
        methods_list=["target"],
        experiment_id=0,
    ) -> None:
        log.info(f"Using dataset {dataset} with learners = {learner_names_list} and methods = {methods_list}.")

        all_metrics_df = []

        for target_dataset_size in self.target_dataset_size_list:
            log.info(f"Now working with target dataset size = {target_dataset_size}...")

            X_train, X_test = load(dataset, self.seed, train_ratio=train_ratio, num_examples_target=target_dataset_size)

            (X_source_specific_train, X_source_shared_train, X_target_specific_train, X_target_shared_train) = X_train
            (X_source_specific_test, X_source_shared_test, X_target_specific_test, X_target_shared_test) = X_test

            if self.synthetic_simulator_type == "linear":
                sim = SyntheticSimulatorLinear(
                    X_source_specific_train,
                    X_source_shared_train,
                    X_target_specific_train,
                    X_target_shared_train,
                    seed=self.seed,
                )
            else:
                raise Exception("Unknown simulator type.")

            W_source_train, Y_source_train, _, W_target_train, Y_target_train, _ = sim.simulate_dataset(
                X_source_specific_train,
                X_source_shared_train,
                X_target_specific_train,
                X_target_shared_train,
                binary_outcome=binary_outcome,
                treatment_assign="pred",
            )
            W_source_test, Y_source_test, _, W_target_test, Y_target_test, PO_target_test = sim.simulate_dataset(
                X_source_specific_test,
                X_source_shared_test,
                X_target_specific_test,
                X_target_shared_test,
                binary_outcome=binary_outcome,
                treatment_assign="pred",
            )

            cate_test = PO_target_test[1] - PO_target_test[0]

            experiment_data = self.run_benchmarks_new(
                learner_names_list,
                methods_list,
                X_source_shared_train,
                X_target_shared_train,
                X_source_specific_train,
                X_target_specific_train,
                W_source_train,
                W_target_train,
                Y_source_train,
                Y_target_train,
                X_target_specific_test,
                X_target_shared_test,
                cate_test,
            )

            metrics_df = pd.DataFrame(
                experiment_data,
                columns=[
                    "Learner",
                    "Method",
                    "PEHE",
                    "CATE true mean",
                    "CATE true var",
                    "Normalized PEHE",
                ],
            )
            metrics_df["Target dataset size"] = target_dataset_size
            all_metrics_df.append(metrics_df)

        all_metrics_df_all = pd.concat(all_metrics_df)
        print("all_metrics_df_all")
        print(all_metrics_df_all)

        dir_name = "results/target_dataset_size/experiment_id_" + str(experiment_id)

        results_path = self.save_path / dir_name
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        all_metrics_df_all.to_csv(
            results_path / f"dataset_{dataset}_"
            f"{self.synthetic_simulator_type}_"
            f"binary_{binary_outcome}-seed{self.seed}.csv"
        )


class SelectionBiasExperiment(BaselineExperiment):
    def __init__(
        self,
        n_iter: int = 1000,
        seed: int = 42,
        selection_bias_source_list: list = [0, 2, 10],
        selection_bias_target_list: list = [0, 0.5, 1, 2, 5, 10],
        save_path: Path = Path.cwd(),
        synthetic_simulator_type: str = "linear",
    ) -> None:
        super(BaselineExperiment, self).__init__()

        self.n_iter = n_iter
        self.seed = seed
        self.save_path = save_path
        self.synthetic_simulator_type = synthetic_simulator_type

        self.selection_bias_source_list = selection_bias_source_list
        self.selection_bias_target_list = selection_bias_target_list

    def run(
        self,
        dataset="tcga",
        train_ratio=0.8,
        binary_outcome=False,
        learner_names_list=["TLearner"],
        methods_list=["target"],
        experiment_id=0,
    ) -> None:
        log.info(f"Using dataset {dataset} with learners = {learner_names_list} and methods = {methods_list}.")

        all_metrics_df = []
        for selection_bias_source in self.selection_bias_source_list:
            for selection_bias_target in self.selection_bias_target_list:

                log.info(
                    f"Now working with slection bias source = {selection_bias_source} target = {selection_bias_target} ..."
                )

                X_train, X_test = load(dataset, self.seed, train_ratio=train_ratio)

                (
                    X_source_specific_train,
                    X_source_shared_train,
                    X_target_specific_train,
                    X_target_shared_train,
                ) = X_train
                (X_source_specific_test, X_source_shared_test, X_target_specific_test, X_target_shared_test) = X_test

                if self.synthetic_simulator_type == "linear":
                    sim = SyntheticSimulatorLinear(
                        X_source_specific_train,
                        X_source_shared_train,
                        X_target_specific_train,
                        X_target_shared_train,
                        seed=self.seed,
                    )
                else:
                    raise Exception("Unknown simulator type.")

                W_source_train, Y_source_train, _, W_target_train, Y_target_train, _ = sim.simulate_dataset(
                    X_source_specific_train,
                    X_source_shared_train,
                    X_target_specific_train,
                    X_target_shared_train,
                    binary_outcome=binary_outcome,
                    treatment_assign="pred",
                    prop_scale_source=selection_bias_source,
                    pred_scale_target=selection_bias_target,
                )
                W_source_test, Y_source_test, _, W_target_test, Y_target_test, PO_target_test = sim.simulate_dataset(
                    X_source_specific_test,
                    X_source_shared_test,
                    X_target_specific_test,
                    X_target_shared_test,
                    binary_outcome=binary_outcome,
                    treatment_assign="pred",
                    prop_scale_source=selection_bias_source,
                    pred_scale_target=selection_bias_target,
                )

                cate_test = PO_target_test[1] - PO_target_test[0]

                experiment_data = self.run_benchmarks_new(
                    learner_names_list,
                    methods_list,
                    X_source_shared_train,
                    X_target_shared_train,
                    X_source_specific_train,
                    X_target_specific_train,
                    W_source_train,
                    W_target_train,
                    Y_source_train,
                    Y_target_train,
                    X_target_specific_test,
                    X_target_shared_test,
                    cate_test,
                )

                metrics_df = pd.DataFrame(
                    experiment_data,
                    columns=[
                        "Learner",
                        "Method",
                        "PEHE",
                        "CATE true mean",
                        "CATE true var",
                        "Normalized PEHE",
                    ],
                )
                metrics_df["Selection bias source"] = selection_bias_source
                metrics_df["Selection bias target"] = selection_bias_target

                all_metrics_df.append(metrics_df)

        all_metrics_df_all = pd.concat(all_metrics_df)
        print("all_metrics_df_all")
        print(all_metrics_df_all)

        dir_name = "results/selection_bias/experiment_id_" + str(experiment_id)

        results_path = self.save_path / dir_name
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        all_metrics_df_all.to_csv(
            results_path / f"dataset_{dataset}_"
            f"{self.synthetic_simulator_type}_"
            f"binary_{binary_outcome}-seed{self.seed}.csv"
        )
