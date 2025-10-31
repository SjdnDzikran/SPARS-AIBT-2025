from SPARS.Simulator.Algo.BaseAlgorithm import BaseAlgorithm


class FCFSNormal(BaseAlgorithm):
    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        super().prep_schedule(new_state, waiting_queue, scheduled_queue, resources_agenda)
        self.FCFSNormal()
        if self.timeout is not None:
            super().timeout_policy()
        return self.events

    def FCFSNormal(self):
        # if self.current_time == 168:
        #     print('x')
        waiting_queue = [
            job for job in self.waiting_queue if job['job_id'] not in self.scheduled]
        for job in waiting_queue:
            if job['res'] <= len(self.available) + len(self.inactive):
                if job['res'] <= len(self.available):
                    allocated_nodes = self.available[:job['res']]
                    allocated_ids = [node['id'] for node in allocated_nodes]
                    self.available = self.available[job['res']:]
                    self.allocated.extend(allocated_nodes)
                    self.scheduled.append(job['job_id'])

                    event = {
                        'job_id': job['job_id'],
                        'subtime': job['subtime'],
                        'runtime': job['runtime'],
                        'reqtime': job['reqtime'],
                        'res': job['res'],
                        'type': 'execution_start',
                        'nodes': allocated_ids
                    }
                    if self.timeout:
                        super().remove_from_timeout_list(allocated_ids)
                    # if event['job_id'] == 25:
                    #     print('fn 36')
                    self.jobs_manager.add_job_to_scheduled_queue(
                        event['job_id'], allocated_ids, self.current_time)
                    super().push_event(self.current_time, event)

            else:
                break
