from typing import Tuple
import numpy as np


def feature_extraction(simulator) -> Tuple[np.ndarray, np.ndarray]:
    # === GLOBAL FEATURES ===
    num_sim_features = 5
    simulator_features = np.zeros((num_sim_features,), dtype=np.float32)

    job_num = len(simulator.jobs_manager.waiting_queue)
    simulator_features[0] = job_num

    arrival_rate = len(simulator.Monitor.jobs_submission_log) / (
        simulator.current_time - simulator.start_time
        + 1e-8)

    simulator_features[1] = arrival_rate

    mean_runtime_jobs_in_queue = sum(
        job["runtime"] for job in simulator.jobs_manager.waiting_queue) / (len(simulator.jobs_manager.waiting_queue) + 1e-8)

    simulator_features[2] = mean_runtime_jobs_in_queue

    total_energy_waste = sum(e["energy_waste"]
                             for e in simulator.Monitor.energy)

    simulator_features[3] = total_energy_waste

    mean_requested_walltime_jobs_in_queue = mean_runtime_jobs_in_queue
    simulator_features[4] = mean_requested_walltime_jobs_in_queue

    # expand simulator features for concatenation
    simulator_features = simulator_features[np.newaxis, ...]

# === NODE FEATURES ===
    num_node_features = 6
    hosts = list(simulator.PlatformControl.get_state())
    num_nodes = len(hosts)

    # Generate random values per node per feature
    node_features = np.random.uniform(0.0, 10.0, size=(
        num_nodes, num_node_features)).astype(np.float32)

    # Broadcast simulator features to match node_features rows
    simulator_features_broadcast = np.broadcast_to(
        simulator_features, (num_nodes, simulator_features.shape[1])
    )

    # Concatenate along features axis
    features = np.concatenate(
        (simulator_features_broadcast, node_features), axis=1)

    return features
