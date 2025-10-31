from SPARS.Simulator.Algo.easy_adaptive import EASYAdaptive
from SPARS.Simulator.Algo.easy_auto_switch_on import EASYAuto
from SPARS.Simulator.Algo.easy_balanced import EASYBalanced
from SPARS.Simulator.Algo.easy_normal import EASYNormal
from SPARS.Simulator.Algo.fcfs_auto_switch_on import FCFSAuto
from SPARS.Simulator.Algo.fcfs_normal import FCFSNormal

ALGO_MAP = {
    'fcfs_auto': FCFSAuto,
    'fcfs_normal': FCFSNormal,
    'easy_normal': EASYNormal,
    'easy_auto': EASYAuto,
    'easy_balanced': EASYBalanced,
    'easy_adaptive': EASYAdaptive,
}


class Scheduler:
    def __init__(self, state, waiting_queue, algorithm, start_time, jobs_manager, timeout=None):
        AlgorithmClass = ALGO_MAP[algorithm.lower()]
        self.algorithm = AlgorithmClass(
            state,
            waiting_queue,
            start_time,
            jobs_manager,
            timeout
        )
        self.jobs_manager = jobs_manager

    def schedule(self, current_time, new_state, waiting_queue, scheduled_queue, resources_agenda):
        self.algorithm.set_time(current_time)
        events = self.algorithm.schedule(
            new_state, waiting_queue, scheduled_queue, resources_agenda)
        return events
