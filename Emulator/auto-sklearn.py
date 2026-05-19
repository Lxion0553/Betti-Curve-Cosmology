from pickle import FALSE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append('..')
from Emulator.GPRemulator import load_emulator_data
import joblib
from autosklearn.regression import AutoSklearnRegressor
import numpy as np
from pprint import pprint
import os
import argparse
import types

from autosklearn.metrics import make_scorer
from Emulator.chi2_metric import (
    reduced_chi2_metric,
    set_chi2_cov,
    set_chi2_cov_path,
)

rsd = True


def main(trial_index, dim, data_path="../Data/EmulatorData/nwLH_fof_emulator_dimensionless_[(1,6),(2,15),(9,19)].bc"):
    cuts = [range(1, 6), range(2, 15), range(9, 19)]

    if not os.path.exists(f'./trial{trial_index}'):
        os.makedirs(f'./trial{trial_index}')
    # temp directory
    tempdir = f'./trial{trial_index}/autosklearn_tmp{dim}'

    # data_path = "../Data/EmulatorData/nwLH_fof_emulator_dimensionless_rsdz_[(1,6),(2,15),(9,19)].bc"
    # data_path = '../Data/EmulatorData/nwLH_fof_emulator_dimensionless_[(1,5),(4,14),(9,19)].bc'
    dataset = load_emulator_data(data_path)
    input = np.array([np.array(d[0]) for d in dataset], dtype=np.float32)
    # input = input[:, :-1]  # exclude l
    output = [np.array([d[1][dim] for d in dataset], dtype=np.float32) for dim in range(3)]

    print("-------------------------------------------------------------------")
    print(f"Training file : {data_path}")
    print("Optimizing for dimension: ", dim)
    X_train, X_test, y_train, y_test = train_test_split(input, output[dim], test_size=0.1, random_state=42)

    cov_path = '../Data/EmulatorData/cov_fof_rsdz.npy' if rsd else '../Data/EmulatorData/cov_fof.npy'
    lens = [len(list(c)) for c in cuts]
    offsets = np.cumsum([0] + lens[:-1]) 
    cov = np.load(cov_path)[offsets[dim]:offsets[dim]+lens[dim], :][:, offsets[dim]:offsets[dim]+lens[dim]]

    # Ensure forkserver/spawn workers load the sliced covariance (shape matches y_true).
    cov_slice_path = f'./trial{trial_index}/cov_dim{dim}.npy'
    np.save(cov_slice_path, cov)

    set_chi2_cov_path(cov_slice_path)
    set_chi2_cov(cov)

    chi2_metric = make_scorer(
        name="reduced_chi2",
        score_func=reduced_chi2_metric,
        optimum=0.0,          
        worst_possible_result=np.inf,
        greater_is_better=False,
    )

    automl = AutoSklearnRegressor(
    time_left_for_this_task=int(24 * 3600),
    tmp_folder=tempdir,
    seed=42,
    memory_limit=12*1024,
    n_jobs=60,
    metric=chi2_metric,
    # resampling_strategy='cv',
    # resampling_strategy_arguments={'folds': 5},
    include={'regressor': ['gaussian_process']},
    )
    automl.fit(X_train, y_train, dataset_name="BettiCurve")
    # automl.fit(X_train, y_train, dataset_name="BettiCurve")
    # automl.refit(X_train.copy(), y_train.copy()) # call refit when using cv

    print(automl.leaderboard())
    pprint(automl.show_models(), indent=4)
    print(automl.sprint_statistics())
    predictions = automl.predict(X_test)
    print("R2 score:", r2_score(y_test, predictions))
    print(automl.get_configuration_space(X_train, y_train))

    joblib.dump(automl, f'./trial{trial_index}/autosklearn_model_{dim}.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto-sklearn training script')
    parser.add_argument('--trial_index', type=int, help='Trial index for saving the model')
    parser.add_argument('--dim', type=int, help='Dimension to optimize')
    parser.add_argument('--data_path', type=str, default="../Data/EmulatorData/nwLH_fof_emulator_dimensionless_[(1,6),(2,15),(9,19)].bc", help='Path to the training data file')
    parser.add_argument('--rsd', action='store_true', help='Whether to use redshift space distortion covariance')
    
    args = parser.parse_args()
    
    main(args.trial_index, args.dim, args.data_path)



    
    
