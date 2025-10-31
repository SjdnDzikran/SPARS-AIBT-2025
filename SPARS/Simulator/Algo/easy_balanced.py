from __future__ import annotations

from .easy_normal import EASYNormal


class EASYBalanced(EASYNormal):
    """
    Energy-aware EASY scheduler that keeps the low-power behaviour of EASYNormal,
    but selectively wakes sleeping nodes when the head job has waited "too long".

    Heuristics:
    - Prefer immediate dispatch on already-active nodes (no wake-ups).
    - For the FCFS head job, allow switching on additional nodes only if:
        * the job has waited past a dynamic threshold, and
        * the predicted release window for enough active nodes is still far away.
    - Once the head job is planned, reuse EASY backfilling to fill gaps with
      smaller jobs without additional wake-ups.
    """

    MIN_WAIT_BEFORE_WAKE = 300          # seconds (5 minutes)
    WAIT_FACTOR = 0.25                  # 25% of job reqtime
    RELEASE_GRACE_WINDOW = 120          # seconds

    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        super().prep_schedule(new_state, waiting_queue, scheduled_queue, resources_agenda)
        self._energy_aware_fcfs()
        self.backfill()

        if self.timeout is not None:
            super().timeout_policy()

        return self.events

    def _energy_aware_fcfs(self):
        waiting_queue = [
            job for job in self.waiting_queue if job['job_id'] not in self.scheduled
        ]

        for job in waiting_queue:
            available_now = len(self.available)
            if job['res'] <= available_now:
                self._start_now(job)
                continue

            shortfall = job['res'] - available_now
            if shortfall > len(self.inactive):
                # Cannot satisfy even with all sleeping nodes; respect FCFS ordering.
                break

            if not self._should_wake_for(job):
                break

            self._wake_and_reserve(job, shortfall)

    def _should_wake_for(self, job) -> bool:
        wait_time = max(0, self.current_time - job['subtime'])
        reqtime = job.get('reqtime') or job.get('runtime', 0)
        dynamic_wait = reqtime * self.WAIT_FACTOR
        wake_threshold = max(self.MIN_WAIT_BEFORE_WAKE, dynamic_wait)

        if wait_time < wake_threshold:
            return False

        release_delay = self._predict_release_delay(job['res'])
        if release_delay <= self.RELEASE_GRACE_WINDOW:
            # Enough active nodes become free soon, no need to wake extra ones yet.
            return False

        return True

    def _predict_release_delay(self, needed: int) -> float:
        state_by_id = {node['id']: node for node in self.state}
        eligible = [
            entry for entry in self.resources_agenda
            if state_by_id.get(entry['id'], {}).get('state') not in (
                'sleeping', 'switching_off', 'switching_on')
        ]
        if len(eligible) < needed:
            return float('inf')
        release_time = eligible[needed - 1]['release_time']
        return max(0.0, release_time - self.current_time)

    def _start_now(self, job):
        allocated_nodes = list(self.available[:job['res']])
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

        self.jobs_manager.add_job_to_scheduled_queue(
            event['job_id'], allocated_ids, self.current_time
        )
        super().push_event(self.current_time, event)

    def _wake_and_reserve(self, job, shortfall: int):
        to_activate = list(self.inactive[:shortfall])
        still_available = list(self.available)
        reserved_nodes = still_available + to_activate
        reserved_ids = [node['id'] for node in reserved_nodes]
        activate_ids = [node['id'] for node in to_activate]

        self.available = []
        self.inactive = self.inactive[shortfall:]
        self.allocated.extend(reserved_nodes)
        self.scheduled.append(job['job_id'])

        highest_release_time = max(
            (ra["release_time"] for ra in self.resources_agenda if ra["id"] in reserved_ids),
            default=self.current_time
        )
        start_predict_time = max(self.current_time, highest_release_time)

        event = {
            'job_id': job['job_id'],
            'subtime': job['subtime'],
            'runtime': job['runtime'],
            'reqtime': job['reqtime'],
            'res': job['res'],
            'type': 'execution_start',
            'nodes': reserved_ids
        }

        if self.timeout:
            super().remove_from_timeout_list(reserved_ids)

        self.jobs_manager.add_job_to_scheduled_queue(
            event['job_id'], reserved_ids, start_predict_time
        )

        if activate_ids:
            super().push_event(self.current_time, {'type': 'switch_on', 'nodes': activate_ids})
        super().push_event(self.current_time, {'type': 'reserve', 'nodes': reserved_ids})
        super().push_event(start_predict_time, event)
