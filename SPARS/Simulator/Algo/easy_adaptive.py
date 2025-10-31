from __future__ import annotations

from .easy_balanced import EASYBalanced


class EASYAdaptive(EASYBalanced):
    """
    Adaptive EASY variant that reacts to queue pressure and shortfall size.

    Goals:
    - Wake a few sleeping nodes quickly when the head job is blocked by a small
      shortfall, reducing latency.
    - Escalate the wake threshold dynamically when the waiting queue grows,
      so we trade a little more energy for shorter waits under load.
    - Reuse EASY backfilling once the head job is planned.
    """

    MIN_WAIT_BASE = 120         # base seconds before waking when queue is light
    MIN_WAIT_FLOOR = 20         # never require more than this when queue is large
    QUEUE_DEC = 25              # drop threshold per extra queued job
    WAIT_FACTOR = 0.2           # optional fraction of runtime as guardrail
    SMALL_SHORTFALL = 2         # wake immediately if shortfall <= this
    LARGE_SHORTFALL = 6         # require stronger signal if shortfall very large
    RELEASE_GRACE_BASE = 45     # seconds
    RELEASE_GRACE_SLOPE = 20    # per queued job

    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        super().prep_schedule(new_state, waiting_queue, scheduled_queue, resources_agenda)
        self._adaptive_fcfs()
        self.backfill()

        if self.timeout is not None:
            super().timeout_policy()

        return self.events

    def _adaptive_fcfs(self):
        waiting_queue = [
            job for job in self.waiting_queue if job['job_id'] not in self.scheduled
        ]
        queue_len = len(waiting_queue)

        for idx, job in enumerate(waiting_queue):
            available_now = len(self.available)
            if job['res'] <= available_now:
                self._start_now(job)
                continue

            shortfall = job['res'] - available_now
            if shortfall > len(self.inactive):
                break  # can't satisfy even by waking everything

            if not self._should_wake_for(job, shortfall, queue_len, idx):
                break  # stick with FCFS ordering

            self._wake_and_reserve(job, shortfall)

    def _should_wake_for(self, job, shortfall: int, queue_len: int, position: int) -> bool:
        # Small shortfall? wake immediately
        if shortfall <= self.SMALL_SHORTFALL:
            return True

        wait_time = max(0, self.current_time - job['subtime'])
        reqtime = job.get('reqtime') or job.get('runtime', 0) or 0

        dynamic_threshold = self.MIN_WAIT_BASE - self.QUEUE_DEC * max(0, queue_len - 1)
        dynamic_threshold = max(self.MIN_WAIT_FLOOR, dynamic_threshold)

        if reqtime > 0:
            dynamic_threshold = min(dynamic_threshold, reqtime * self.WAIT_FACTOR)

        if wait_time < dynamic_threshold:
            return False

        release_delay = self._predict_release_delay(job['res'])
        grace_window = self.RELEASE_GRACE_BASE + self.RELEASE_GRACE_SLOPE * max(0, queue_len - 1)

        # For large shortfalls require that the release delay is significant
        if shortfall >= self.LARGE_SHORTFALL and release_delay <= grace_window:
            return False

        if release_delay <= grace_window and position == 0:
            # Head job expects resources soon; keep waiting unless queue is heavily loaded
            if queue_len <= 2:
                return False

        return True
