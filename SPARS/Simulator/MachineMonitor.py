import pandas as pd
import logging

logger = logging.getLogger("runner")


class Monitor:
    def __init__(self, platform_info, start_time):
        self.ecr = [
            {
                "id": m["id"],
                "dvfs_profiles": {
                    dvfs_name: dvfs_info["power"]
                    for dvfs_name, dvfs_info in m["dvfs_profiles"].items()
                },
                "states": {
                    state: m["states"][state]["power"]
                    for state in m["states"]
                }
            }
            for m in platform_info["machines"]
        ]

        self.num_nodes = len(platform_info['machines'])
        self.node_ids = [node['id'] for node in platform_info['machines']]
        self.energy = [
            {'id': i, 'energy_consumption': 0, 'energy_effective': 0,
                'energy_waste': 0, 'last_update': start_time}
            for i in range(self.num_nodes)
        ]
        self.nodes_state = [{'id': i, 'state': 'active', 'dvfs_mode': 'base',
                             'start_time': start_time, 'duration': 0, 'job_id': None} for i in range(self.num_nodes)]

        self.states_hist = [{'id': i, 'state_history': [{'state': 'active', 'dvfs_mode': 'base',
                                                         'start_time': start_time, 'finish_time': 0}]} for i in range(self.num_nodes)]
        self.states_dur = []

        self.jobs_arrival_log = []
        self.jobs_submission_log = []
        self.jobs_execution_log = []

        for i, node in enumerate(platform_info['machines']):
            entry = {'id': i}
            for state in node['states'].keys():
                if state == 'active':
                    entry['active_idle'] = {
                        dvfs_mode: 0 for dvfs_mode in node['dvfs_profiles'].keys()}
                    entry['active_compute'] = {
                        dvfs_mode: 0 for dvfs_mode in node['dvfs_profiles'].keys()}
                else:
                    entry[state] = {
                        dvfs_mode: 0 for dvfs_mode in node['dvfs_profiles'].keys()}
            self.states_dur.append(entry)

    def print_energy(self):
        for entry in self.energy:
            logger.info(
                f"Node {entry['id']}: Energy Consumption = {entry['energy_consumption']}, Energy Effective = {entry['energy_effective']}, Energy Waste = {entry['energy_waste']}")

    def print_states_dur(self):
        for state_dur in self.states_dur:
            logger.info(f"Node {state_dur['id']}: {state_dur}")

    def on_finish(self):
        for node_state in self.nodes_state:
            state_hist = next(
                (h for h in self.states_hist if h['id'] == node_state['id']), None)
            if state_hist is not None:
                state_hist['state_history'].append({
                    'state': node_state['state'],
                    'start_time': node_state['start_time'],
                    'finish_time': node_state['start_time'] + node_state['duration']
                })

        self.print_energy()
        self.print_states_dur()

    def record(self, mode, current_time=None, machines=None, record_job_arrival=None, record_job_submission=None, record_job_execution=None):
        if mode not in ('before', 'after'):
            raise ValueError(
                f"Invalid mode '{mode}'. Expected 'before' or 'after'.")

        if mode == 'before':
            if current_time is None:
                raise ValueError(
                    "`current_time` is required for mode 'before'.")
            self.update_node_state_duration(current_time)
            self.update_energy(current_time)

        elif mode == 'after':
            if machines is None:
                raise ValueError("`machines` is required for mode 'after'.")
            if current_time is None:
                raise ValueError(
                    "`current_time` is required for mode 'before'.")
            if len(record_job_arrival) > 0:
                for job in record_job_arrival:
                    self.jobs_arrival_log.append(job)

            if len(record_job_submission) > 0:
                for job in record_job_submission:
                    self.jobs_submission_log.append(job)

            if len(record_job_execution) > 0:
                for job in record_job_execution:
                    job['finish_time'] = current_time
                    self.jobs_execution_log.append(job)

            self.update_node_state(machines, current_time)

    def update_energy(self, current_time):
        for node in self.nodes_state:
            node_id = node['id']
            node_state = node['state']

            energy_entry = next(
                (e for e in self.energy if e['id'] == node_id), None)
            ecr_entry = next((e for e in self.ecr if e['id'] == node_id), None)

            if not energy_entry or not ecr_entry:
                continue

            timespan = node['duration'] + \
                node['start_time'] - energy_entry['last_update']
            energy_entry['last_update'] = current_time

            ecr_value = ecr_entry['states'][node_state]
            if ecr_value == 'from_dvfs':
                dvfs_mode = node['dvfs_mode']
                ecr_value = ecr_entry['dvfs_profiles'][dvfs_mode]

            if node['job_id'] is None:
                """" Node is active but not computing"""
                energy_entry['energy_waste'] += ecr_value * timespan
            else:
                energy_entry['energy_effective'] += ecr_value * timespan

            energy_entry['energy_consumption'] = energy_entry['energy_effective'] + \
                energy_entry['energy_waste']

    def update_node_state_duration(self, current_time):
        for node in self.nodes_state:
            delta = current_time - node['start_time'] - node['duration']
            node['duration'] += delta
            state = node['state']
            dvfs_mode = node['dvfs_mode']

            if state == 'active' and node['job_id'] == None:
                state = 'active_idle'
            elif state == 'active' and node['job_id'] != None:
                state = 'active_compute'

            for state_dur in self.states_dur:
                if state_dur['id'] == node['id']:
                    state_dur[state][dvfs_mode] += delta
                    break

    def update_node_state(self, machines, current_time):
        # Build a quick lookup of machine states by ID
        machine_state_by_id = {node['id']: node['state']
                               for node in machines.nodes}
        machine_job_id_by_id = {node['id']: node['job_id']
                                for node in machines.nodes}
        machine_dvfs_mode_by_id = {
            node['id']: node['dvfs_mode'] for node in machines.nodes}
        for node_state in self.nodes_state:
            node_id = node_state['id']
            new_state = machine_state_by_id.get(node_id)
            new_job_id = machine_job_id_by_id.get(node_id)
            new_dvfs_mode = machine_dvfs_mode_by_id.get(node_id)
            if new_state is None and new_dvfs_mode is None:
                continue

            if node_state['state'] != new_state or node_state['job_id'] != new_job_id or node_state['dvfs_mode'] != new_dvfs_mode:
                # Append to state history
                state_hist = next(
                    (h for h in self.states_hist if h['id'] == node_id), None)
                if state_hist is not None or new_dvfs_mode is not None:
                    state_hist['state_history'].append({
                        'state': node_state['state'],
                        'start_time': node_state['start_time'],
                        'finish_time': node_state['start_time'] + node_state['duration'],
                        'dvfs_mode': node_state['dvfs_mode']
                    })

                # Update current state

                node_state['state'] = new_state
                node_state['job_id'] = new_job_id
                node_state['start_time'] = current_time
                node_state['duration'] = 0
                node_state['dvfs_mode'] = new_dvfs_mode
