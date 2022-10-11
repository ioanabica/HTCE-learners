import argparse
import sys
from typing import Any

import logger as log
from experiments import (BaselineExperiment, POSharingExperiment, TargetDatasetSizeExperiment,
                         SelectionBiasExperiment)


def init_arg() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="po_sharing_across_domains", type=str)
    parser.add_argument("--experiment_id", default=0, type=int)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--synthetic_simulator_type", default='linear', type=str)

    parser.add_argument(
        "--dataset_list",
        nargs="+",
        type=str,
        default=["twins"],
    )
    # Arguments for Propensity Sensitivity Experiment
    parser.add_argument("--treatment_assgn", default="pred", type=str)

    parser.add_argument("--methods_list", nargs="+",
                        default=['target', 'htce', 'shared_source_target', 'radial_gan',], type=str)

    parser.add_argument("--learner_names_list", nargs="+",
                        default=['TLearner', 'SLearner', 'DRLearner', 'TARNet', ], type=str)

    parser.add_argument(
        "--seed_list", nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int
    )
    return parser.parse_args()


if __name__ == "__main__":
    log.add(sink=sys.stderr, level="INFO")
    args = init_arg()
    for seed in args.seed_list:
        log.info(f"Experiment {args.experiment_name} with simulator {args.synthetic_simulator_type} and seed {seed}.")
        if args.experiment_name == "baseline_experiment":
            exp = BaselineExperiment(seed=seed, synthetic_simulator_type=args.synthetic_simulator_type)
        elif args.experiment_name == "po_sharing_across_domains":
            exp = POSharingExperiment(seed=seed, synthetic_simulator_type=args.synthetic_simulator_type,
                                      sharing_across_domains=True)
        elif args.experiment_name == 'target_size':
            exp = TargetDatasetSizeExperiment(seed=seed, synthetic_simulator_type=args.synthetic_simulator_type)
        elif args.experiment_name == 'selection_bias':
            exp = SelectionBiasExperiment(seed=seed, synthetic_simulator_type=args.synthetic_simulator_type)

        log.info(
            f"Running experiment {args.experiment_name} for {args.dataset_list}.")

        for experiment_id in range(len(args.dataset_list)):
            log.info(
                f"Running experiment for {args.dataset_list[experiment_id]}")

            exp.run(
                dataset=args.dataset_list[experiment_id],
                train_ratio=args.train_ratio,
                methods_list=args.methods_list,
                learner_names_list=args.learner_names_list,
                experiment_id=args.experiment_id, )
