# SPARS/Gym/features/global_node_11d.py
# NOTE: Logic unchanged; only moved here.
import numpy as np

FEATURE_DIM = 11  # NOTE: keep feature length explicit


def feature_extraction(simulator) -> np.ndarray:
    """
    Returns a 1D global feature vector of shape [11].
    The timestep axis is *implicit* (you append this per step to history_features).
    """

    # === GLOBAL (simulator-level) FEATURES ===
    tq = simulator.jobs_manager.waiting_queue
    tnow = simulator.current_time
    t0 = simulator.start_time
    dt = max(tnow - t0, 1e-8)

    job_num = float(len(tq))
    arrival_rate = float(len(simulator.Monitor.jobs_submission_log)) / dt
    mean_runtime_q = (sum(job.get("runtime", 0.0)
                      for job in tq) / max(len(tq), 1e-8))
    total_waste = float(sum(e.get("energy_waste", 0.0)
                        for e in simulator.Monitor.energy))
    mean_req_wt_q = mean_runtime_q
    avg_req_nodes = (sum(job.get("res", 0.0)
                     for job in tq) / max(len(tq), 1e-8))

    sim_feats = np.array([
        job_num,
        arrival_rate,
        float(mean_runtime_q),
        total_waste,
        float(mean_req_wt_q),
        float(avg_req_nodes),
    ], dtype=np.float32)  # [6]

    # === AGGREGATED NODE FEATURES ===
    state = list(simulator.PlatformControl.get_state())
    computing_nodes = [n["id"] for n in state if n.get(
        "state") == "active" and n.get("job_id") is not None]
    idle_nodes = [n["id"] for n in state if n.get(
        "state") == "active" and n.get("job_id") is None]
    sleeping_nodes = [n["id"] for n in state if n.get("state") == "sleeping"]

    transitions_info = getattr(getattr(
        simulator.PlatformControl, "machines", object()), "machines_transition", [])
    sleeping_set, idle_set = set(sleeping_nodes), set(idle_nodes)
    switch_on_times, switch_off_times = [], []
    for node_info in transitions_info:
        nid = node_info.get("node_id")
        for tr in node_info.get("transitions", []):
            frm = tr.get("from")
            to = tr.get("to")
            tt = float(tr.get("transition_time", 0.0))
            if frm == "switching_on" and to == "active" and nid in sleeping_set:
                switch_on_times.append(tt)
            if frm == "switching_off" and to == "sleeping" and nid in idle_set:
                switch_off_times.append(tt)

    avg_switch_on = (sum(switch_on_times) /
                     max(len(switch_on_times),  1)) if switch_on_times else 0.0
    avg_switch_off = (sum(switch_off_times) /
                      max(len(switch_off_times), 1)) if switch_off_times else 0.0

    node_feats = np.array([
        float(len(computing_nodes)),
        float(len(idle_nodes)),
        float(len(sleeping_nodes)),
        float(avg_switch_on),
        float(avg_switch_off),
    ], dtype=np.float32)  # [5]

    features = np.concatenate(
        [sim_feats, node_feats], axis=0).astype(np.float32)  # [11]

    features = features.reshape(1, 11)
    return features
