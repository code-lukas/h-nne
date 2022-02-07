from pathlib import Path
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.preprocessing import StandardScaler
import typer

from hnne.benchmarking.utils import time_function_call
from hnne.benchmarking.data_loading import dataset_loaders, DatasetGroup, \
    dataset_validation_knn_values, load_extracted_finch_partitions
from hnne.hierarchical_projection import multi_step_projection
from hnne.benchmarking.evaluation import format_metric, dim_reduction_benchmark
from hnne.finch_clustering import FINCH


def main(
        experiment_name: str,
        output_path: Path,
        data_path: Path,
        dataset_group: DatasetGroup = DatasetGroup.small,
        output_partitions_path: Optional[Path] = None,
        points_plot_limit: int = 50000,
        figsize: Tuple[int, int] = (20, 20),
        large_datasets: bool = False,
        skip_done: bool = True,
        scale_data: bool = False,
        dim: int = 2,
        continue_on_error: bool = False,
        inflate_pointclouds: bool = True,
        radius_shrinking: float = 0.66,
        load_finch_from_disk: bool = False,
        finch_distance: Optional[str] = None,
        validate_only_1nn: bool = True,
        ann_threshold: int = 20000,
        compute_trustworthiness: bool = False,
        projection_type: str = 'pca',
        remove_partitions_above_pca_partition: bool = False,
        project_first_partition_pca: bool = False,
        decompress_points: bool = True
):
    if large_datasets:
        print('Hold tight, processing large datasets...')

    loaders = dataset_loaders(dataset_group=dataset_group)

    for dataset_name, loader in loaders.items():
        try:
            knn_values = [1] if validate_only_1nn else dataset_validation_knn_values[dataset_name]

            filename = f'{dataset_name}_{experiment_name}_dim_{dim}'
            if output_partitions_path is None:
                output_partitions_path = output_path / 'partitions'
            in_partitions_path = output_partitions_path / f'{dataset_name}_finch.pkl'
            in_partitions_performance_path = output_partitions_path / f'{dataset_name}_performance.csv'
            out_projection_path = output_path / 'projections' / f'{filename}.npz'
            out_score_path = output_path / 'scores' / f'{filename}.csv'
            out_plot_path = output_path / 'plots' / f'{filename}.png'

            output_path.mkdir(exist_ok=True)
            out_projection_path.parent.mkdir(exist_ok=True)
            out_score_path.parent.mkdir(exist_ok=True)
            out_plot_path.parent.mkdir(exist_ok=True)

            if skip_done and out_score_path.exists() and out_projection_path.exists():
                print(f'Skipping extraction of existing partitions for {dataset_name}.')
                continue

            print(f'Loading {dataset_name}...')
            data, targets = loader(data_path)
            if scale_data:
                print('Scaling data...')
                data = StandardScaler().fit_transform(data)

            if load_finch_from_disk and finch_distance is None:
                print(f'Loading FINCH partitions for {dataset_name}...')
                [
                    partitions,
                    partition_sizes,
                    _,
                    adjacency_matrices,
                    partition_labels,
                    cluster_dists,
                    first_neighbors_list
                ] = load_extracted_finch_partitions(in_partitions_path)

                time_elapsed_finch = pd.read_csv(in_partitions_performance_path)['time_finch'].iloc[0]
            else:
                distance = 'cosine' if finch_distance is None else finch_distance
                print(f'Extracting FINCH partitions with {distance} distance...')
                [
                    partitions,
                    partition_sizes,
                    req_c,
                    adjacency_matrices,
                    partition_labels,
                    cluster_dists,
                    first_neighbors_list
                ], time_elapsed_finch = time_function_call(
                    FINCH,
                    data,
                    ensure_early_exit=True,
                    verbose=large_datasets,
                    low_memory_nndescent=large_datasets,
                    distance=distance,
                    ann_threshold=ann_threshold
                )

            if partition_sizes[-1] < 3:
                partition_sizes = partition_sizes[:-1]
                partitions = partitions[:, :-1]
                adjacency_matrices = adjacency_matrices[:-1]
                partition_labels = partition_labels[:-1]
                first_neighbors_list = first_neighbors_list[:-1]

            print(f'Projecting {dataset_name} to {dim} dimensions...')
            [projection, projected_centroid_radii, projected_centroids, _, _, _, _,
             _], time_elapsed = time_function_call(
                multi_step_projection,
                data,
                partitions,
                adjacency_matrices,
                partition_labels,
                first_neighbors_list,
                inflate_pointclouds=inflate_pointclouds,
                radius_shrinking=radius_shrinking,
                dim=dim,
                partition_sizes=partition_sizes,
                real_nn_threshold=ann_threshold,
                projection_type=projection_type,
                remove_partitions_above_pca_partition=remove_partitions_above_pca_partition,
                project_first_partition_pca=project_first_partition_pca,
                decompress_points=decompress_points
            )

            np.savez(
                out_projection_path,
                projection=projection,
                projected_centroid_radii=projected_centroid_radii,
                projected_centroids=projected_centroids)

            print(f'Finch time: {time_elapsed_finch}, projection time: {time_elapsed}')

            print(f'Validating {dataset_name} on {knn_values} nearest neighbors...')
            proj_knn_acc, tw = dim_reduction_benchmark(knn_values, data, projection, targets,
                                                       compute_trustworthiness=compute_trustworthiness)
            scores = pd.DataFrame({
                'proj_KNN_ACC': format_metric(proj_knn_acc),
                'trustworthiness': len(knn_values) * [tw],
                'proj_time': [time_elapsed] + (len(knn_values) - 1) * [''],
                'finch_time': [time_elapsed_finch] + (len(knn_values) - 1) * ['']
            },
                index=pd.Series(knn_values, name='k_value'))

            scores.to_csv(out_score_path)

            if dim == 2:
                print(f'Exporting projection plots...')
                if projection.shape[0] > points_plot_limit:
                    idx = np.random.choice(projection.shape[0], points_plot_limit, replace=False)
                    projection = projection[idx]
                    targets = targets[idx]

                plt.figure(figsize=figsize)
                plt.scatter(*projection.T, s=1, c=targets, cmap='Spectral')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig(out_plot_path)

        except Exception as e:
            if continue_on_error:
                print(f'Failed to run experiment on {dataset_name}.')
                print(f'Error: {e}')
            else:
                raise e


if __name__ == '__main__':
    typer.run(main)