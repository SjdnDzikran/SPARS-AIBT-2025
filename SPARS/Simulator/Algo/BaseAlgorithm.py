import copy


class BaseAlgorithm():
    def __init__(self, state, waiting_queue, start_time, jobs_manager, timeout=None):
        self.state = state
        self.waiting_queue = waiting_queue
        self.events = []
        self.current_time = start_time
        self.timeout = timeout

        self.jobs_manager = jobs_manager

        self.available = []
        self.inactive = []
        self.compute_speeds = []
        self.allocated = []
        self.scheduled = []
        self.call_me_later = []
        self.timeout_list = []

    def push_event(self, timestamp, event):
        found = next(
            (x for x in self.events if x['timestamp'] == timestamp), None)
        if found:
            found['events'].append(event)
        else:
            self.events.append({'timestamp': timestamp, 'events': [event]})
        self.events.sort(key=lambda x: x['timestamp'])

    def set_time(self, current_time):
        self.current_time = current_time

    def remove_from_timeout_list(self, node_ids):
        ids = set(node_ids)
        self.timeout_list[:] = [ti for ti in self.timeout_list
                                if ti.get('node_id') not in ids]

    def timeout_policy(self):
        now = self.current_time

        t_exp = now + self.timeout

        # Fast membership on existing timeouts
        allocated_ids = [node['id'] for node in self.allocated]
        timeout_node_ids = {t['node_id'] for t in self.timeout_list}

        # Add timeouts for newly idle, active nodes (no duplicates)
        for node in self.state:
            if (
                node['job_id'] is None
                and node['state'] == 'active'
                and node['reserved'] == False
                and node['id'] not in allocated_ids
                and node['id'] not in timeout_node_ids
            ):
                self.timeout_list.append(
                    {'node_id': node['id'], 'time': t_exp})
                timeout_node_ids.add(node['id'])

        # Build lookups once
        state_by_id = {n['id']: n for n in self.state}
        allocated_ids = {n['id'] for n in self.allocated}

        switch_off = []
        keep_timeouts = []
        next_earliest = None  # track the earliest remaining timeout

        for t in self.timeout_list:
            node = state_by_id.get(t['node_id'])
            if not node:
                # stale timeout for a node that no longer exists
                continue

            # keep if not yet expired or node not active
            if now < t['time'] or node['state'] != 'active':
                keep_timeouts.append(t)
                if next_earliest is None or t['time'] < next_earliest:
                    next_earliest = t['time']
                continue

            # expired:
            if node['job_id'] is None and node['id'] not in allocated_ids:
                switch_off.append(node['id'])
            # else: drop the timeout silently

        # swap in the filtered list
        self.timeout_list = keep_timeouts

        if switch_off:
            self.push_event(now, {'type': 'switch_off', 'nodes': switch_off})

        # Schedule exactly one wake-up at the earliest pending timeout
        if next_earliest is not None and getattr(self, 'next_timeout_at', None) != next_earliest:
            self.push_event(next_earliest, {'type': 'call_me_later'})
            self.next_timeout_at = next_earliest

    def prep_schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        self.state = new_state
        self.waiting_queue = waiting_queue
        self.scheduled_queue = scheduled_queue
        self.resources_agenda = copy.deepcopy(resources_agenda)
        self.resources_agenda = sorted(
            self.resources_agenda,
            key=lambda x: (x['release_time'], x['id'])
        )

        self.events = []
        self.compute_speeds = [node['compute_speed']
                               for node in self.state]

        """ 
        GET ALL AVAILABLE NODES (INCLUDES THE RESERVED NODES)
        THEN CHECK IF A SCHEDULED JOB CAN BE EXECUTED
        """
        self.available = [
            node for node in self.state
            if node['state'] == 'active' and node['job_id'] is None and node['reserved'] == False
        ]

        self.inactive = [
            node for node in self.state
            if node['state'] == 'sleeping' and node['reserved'] == False
        ]

        self.allocated = []  # This tracks the list of allocated nodes in an instance of scheduling, since easy scheduler is executed after fcfs scheduler, we have to make sure easy scheduler doesn't realocate the nodes that already been allocated by fcfs, since fcfs doesnt immediately update simulator's machine, we have to store info of currently allocated nodes in the current instance of scheduling
        self.scheduled = []

        self.reserved_ids = []
        for job in scheduled_queue:
            self.reserved_ids.extend(job['nodes'])

        # if len(self.scheduled_queue) > 0:
        #     for scheduled_job in self.scheduled_queue:
        #         executable = True
        #         for reserved_nodes in scheduled_job['nodes']:
        #             if reserved_nodes not in self.available:
        #                 executable = False

        #         if executable:
        #             allocated_nodes = scheduled_job['nodes']
        #             self.available = [
        #                 node for node in self.available if node not in scheduled_job['nodes']]
        #             self.allocated.extend(scheduled_job['nodes'])

        #             self.scheduled_queue.remove(scheduled_job)
        #             event = {
        #                 'job_id': scheduled_job['job_id'],
        #                 'subtime': scheduled_job['subtime'],
        #                 'runtime': scheduled_job['runtime'],
        #                 'res': scheduled_job['res'],
        #                 'type': 'execution_start',
        #                 'nodes': allocated_nodes
        #             }
        #             self.push_event(self.current_time, event)
        """ 
        CONTINUE EXECUTE SCHEDULING LOGIC WITHOUT CONSIDERING THE RESERVED NODES
        """

        self.available = [
            node for node in self.available if node['id'] not in self.reserved_ids
        ]
