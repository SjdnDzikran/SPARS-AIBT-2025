from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from SPARS.Simulator.Algo.BaseAlgorithm import BaseAlgorithm


@dataclass
class JobConfig:
    job_id: int
    start: float
    nodes: int
    dvfs: str


class Particle:
    """
    Lightweight particle representation:
    - position: mapping job_id -> JobConfig
    - velocity: mapping job_id -> (delta_start, delta_nodes)
    """

    def __init__(self, configs: Dict[int, JobConfig], velocity: Dict[int, Tuple[float, float]]):
        self.configs = configs
        self.velocity = velocity
        self.fitness = math.inf

    def copy(self) -> "Particle":
        return Particle(
            {jid: JobConfig(cfg.job_id, cfg.start, cfg.nodes, cfg.dvfs) for jid, cfg in self.configs.items()},
            {jid: (vel[0], vel[1]) for jid, vel in self.velocity.items()},
        )


class PSOMalleable(BaseAlgorithm):
    """
    Particle-swarm-inspired malleable scheduler.

    Each iteration adjusts candidate start times, node counts, and DVFS modes
    for the oldest waiting jobs, balancing energy consumption vs. stretch.
    """

    NUM_PARTICLES = 20
    NUM_ITER = 12
    INERTIA = 0.7
    COGNITIVE = 1.4
    SOCIAL = 1.4
    MAX_CANDIDATES = 6

    ALPHA = 0.5  # energy weight
    BETA = 0.5   # stretch weight

    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        super().prep_schedule(new_state, waiting_queue, scheduled_queue, resources_agenda)

        candidates = [
            job for job in self.waiting_queue
            if job['job_id'] not in self.scheduled
        ][:self.MAX_CANDIDATES]

        if not candidates:
            if self.timeout is not None:
                super().timeout_policy()
            return self.events

        job_lookup = {job['job_id']: job for job in candidates}

        machine_ids = [node['id'] for node in self.state]
        total_nodes = len(machine_ids)
        dvfs_profiles = self._get_dvfs_profiles()

        particles = self._initialise_particles(candidates, dvfs_profiles, total_nodes)
        personal_best = [p.copy() for p in particles]
        global_best = None

        for particle in particles:
            self._evaluate_particle(particle, job_lookup, dvfs_profiles, total_nodes)
            if global_best is None or particle.fitness < global_best.fitness:
                global_best = particle.copy()

        # PSO loop
        for _ in range(self.NUM_ITER):
            for idx, particle in enumerate(particles):
                self._evaluate_particle(particle, job_lookup, dvfs_profiles, total_nodes)
                if particle.fitness < personal_best[idx].fitness:
                    personal_best[idx] = particle.copy()
                if global_best is None or particle.fitness < global_best.fitness:
                    global_best = particle.copy()

            if global_best is None:
                break

            for idx, particle in enumerate(particles):
                best_personal = personal_best[idx]
                self._update_particle(particle, best_personal, global_best, dvfs_profiles, total_nodes, job_lookup)

        if global_best is None:
            if self.timeout is not None:
                super().timeout_policy()
            return self.events

        # Extract events from global best schedule (immediate dispatch only)
        available_nodes = [node for node in self.available]
        sleeping_nodes = [node for node in self.inactive]
        planned_events: List[dict] = []

        for job in candidates:
            nodes_needed = job['res']
            if len(available_nodes) < nodes_needed:
                deficit = nodes_needed - len(available_nodes)
                sleepers = [node for node in sleeping_nodes if node.get('state') == 'sleeping']
                if len(sleepers) >= deficit:
                    to_wake = [node['id'] for node in sleepers[:deficit]]
                    sleeping_nodes = sleeping_nodes[deficit:]
                    super().push_event(self.current_time, {
                        'type': 'switch_on',
                        'nodes': to_wake,
                    })
                continue

            selected_nodes = available_nodes[:nodes_needed]
            allocated_nodes = [node['id'] for node in selected_nodes]
            del available_nodes[:nodes_needed]

            event = {
                'job_id': job['job_id'],
                'subtime': job['subtime'],
                'runtime': job['runtime'],
                'reqtime': job['reqtime'],
                'res': nodes_needed,
                'type': 'execution_start',
                'nodes': allocated_nodes,
            }

            self.jobs_manager.add_job_to_scheduled_queue(
                event['job_id'], allocated_nodes, self.current_time
            )
            self.allocated.extend(
                [node for node in self.state if node['id'] in allocated_nodes]
            )
            planned_events.append(event)

        for event in planned_events:
            if self.timeout:
                super().remove_from_timeout_list(event['nodes'])
            super().push_event(self.current_time, event)

        if self.timeout is not None:
            super().timeout_policy()

        return self.events

    # ------------------------------------------------------------------ helpers

    def _get_dvfs_profiles(self) -> Dict[str, Dict[str, float]]:
        profiles = {}
        for node in self.state:
            dvfs = node.get('dvfs_profiles') or {}
            for mode, data in dvfs.items():
                if mode not in profiles:
                    power = float(data.get('power', 0.0) or 0.0)
                    speed = float(data.get('compute_speed', 1.0) or 1.0)
                    profiles[mode] = {'power': max(power, 1.0), 'speed': max(speed, 0.1)}
        if not profiles:
            profiles['base'] = {'power': 1.0, 'speed': 1.0}
        return profiles

    def _initialise_particles(
        self,
        jobs: List[dict],
        dvfs_profiles: Dict[str, Dict[str, float]],
        total_nodes: int,
    ) -> List[Particle]:
        particles = []
        dvfs_modes = list(dvfs_profiles.keys())
        for _ in range(self.NUM_PARTICLES):
            configs = {}
            velocity = {}
            for job in jobs:
                start_lb = max(self.current_time, job['subtime'])
                start_ub = start_lb + 600  # 10-minute lookahead
                start_time = random.uniform(start_lb, start_ub)
                min_nodes = min(job['res'], total_nodes)
                max_nodes = max(min_nodes, total_nodes)
                nodes = random.randint(min_nodes, max_nodes)
                dvfs = random.choice(dvfs_modes)
                configs[job['job_id']] = JobConfig(job['job_id'], start_time, nodes, dvfs)
                velocity[job['job_id']] = (
                    random.uniform(-60, 60),  # delta start
                    random.uniform(-2, 2),    # delta nodes
                )
            particles.append(Particle(configs, velocity))
        return particles

    def _evaluate_particle(
        self,
        particle: Particle,
        job_lookup: Dict[int, dict],
        dvfs_profiles: Dict[str, Dict[str, float]],
        total_nodes: int,
    ) -> float:
        configs = list(particle.configs.values())
        configs.sort(key=lambda c: c.start)

        event_timeline = []
        for cfg in configs:
            job = job_lookup.get(cfg.job_id)
            if job is None:
                continue
            start = max(self.current_time, job['subtime'], cfg.start)
            node_count = max(job['res'], min(cfg.nodes, total_nodes))
            dvfs = dvfs_profiles.get(cfg.dvfs, {'power': 1.0, 'speed': 1.0})
            speed = dvfs['speed']
            runtime_parallel = job['runtime'] / max(speed, 0.1)
            runtime_parallel = max(runtime_parallel * job['res'] / node_count, runtime_parallel)
            finish = start + runtime_parallel
            event_timeline.append((start, node_count))
            event_timeline.append((finish, -node_count))

        event_timeline.sort()
        current_nodes = 0
        penalty = 0.0
        for _, delta in event_timeline:
            current_nodes += delta
            if current_nodes > total_nodes:
                penalty += (current_nodes - total_nodes) * 1000

        total_energy = 0.0
        total_stretch = 0.0
        count = 0
        for job_id, job in job_lookup.items():
            cfg = particle.configs.get(job_id)
            if cfg is None:
                continue
            start = max(self.current_time, job['subtime'], cfg.start)
            nodes = max(job['res'], min(cfg.nodes, total_nodes))
            dvfs = dvfs_profiles.get(cfg.dvfs, {'power': 1.0, 'speed': 1.0})
            power = dvfs['power']
            speed = max(dvfs['speed'], 0.1)

            runtime_parallel = job['runtime'] * job['res'] / nodes
            runtime_parallel = runtime_parallel / speed
            wait = max(0.0, start - job['subtime'])
            stretch = (wait + runtime_parallel) / max(job['runtime'], 1e-3)
            total_stretch += stretch
            count += 1

            energy = power * runtime_parallel
            total_energy += energy

        if count == 0:
            particle.fitness = math.inf
        else:
            avg_stretch = total_stretch / count
            particle.fitness = (
                self.ALPHA * total_energy +
                self.BETA * avg_stretch +
                penalty
            )
        return particle.fitness

    def _update_particle(
        self,
        particle: Particle,
        personal_best: Particle,
        global_best: Particle,
        dvfs_profiles: Dict[str, Dict[str, float]],
        total_nodes: int,
        job_lookup: Dict[int, dict],
    ) -> None:
        dvfs_modes = list(dvfs_profiles.keys())
        for job_id, cfg in particle.configs.items():
            vel_start, vel_nodes = particle.velocity[job_id]

            personal_cfg = personal_best.configs[job_id]
            global_cfg = global_best.configs[job_id]

            r1 = random.random()
            r2 = random.random()

            vel_start = (
                self.INERTIA * vel_start
                + self.COGNITIVE * r1 * (personal_cfg.start - cfg.start)
                + self.SOCIAL * r2 * (global_cfg.start - cfg.start)
            )
            vel_nodes = (
                self.INERTIA * vel_nodes
                + self.COGNITIVE * r1 * (personal_cfg.nodes - cfg.nodes)
                + self.SOCIAL * r2 * (global_cfg.nodes - cfg.nodes)
            )

            cfg.start += vel_start
            cfg.start = max(self.current_time, min(cfg.start, self.current_time + 3600))

            cfg.nodes = int(round(cfg.nodes + vel_nodes))
            min_nodes = job_lookup.get(job_id, {}).get('res', 1)
            cfg.nodes = max(min_nodes, min(cfg.nodes, total_nodes))

            if random.random() < 0.2:
                cfg.dvfs = random.choice(dvfs_modes)
            else:
                cfg.dvfs = random.choice([cfg.dvfs, personal_cfg.dvfs, global_cfg.dvfs])

            particle.velocity[job_id] = (vel_start, vel_nodes)

        self._make_particle_feasible(particle, total_nodes, job_lookup, dvfs_profiles)

    def _make_particle_feasible(
        self,
        particle: Particle,
        total_nodes: int,
        job_lookup: Dict[int, dict],
        dvfs_profiles: Dict[str, Dict[str, float]],
    ) -> None:
        configs = list(particle.configs.values())
        configs.sort(key=lambda c: c.start)

        for cfg in configs:
            cfg.start = max(self.current_time, cfg.start)
            min_nodes = job_lookup.get(cfg.job_id, {}).get('res', 1)
            cfg.nodes = max(min_nodes, min(cfg.nodes, total_nodes))

        active = []
        for cfg in configs:
            job = job_lookup.get(cfg.job_id)
            if not job:
                continue
            dvfs = dvfs_profiles.get(cfg.dvfs, {'speed': 1.0})
            speed = max(dvfs['speed'], 0.1)
            runtime = (job['runtime'] * job['res'] / cfg.nodes) / speed
            start = cfg.start
            finish = start + runtime

            active = [item for item in active if item[1] > start]
            nodes_in_use = sum(item[0] for item in active)
            if nodes_in_use + cfg.nodes > total_nodes:
                latest_finish = max(item[1] for item in active) if active else start
                cfg.start = latest_finish + 1
                finish = cfg.start + runtime
            active.append((cfg.nodes, finish))

    def _allocate_nodes(
        self,
        nodes_needed: int,
        available_nodes: List[dict],
        sleeping_nodes: List[dict],
    ) -> Tuple[List[int], List[int]]:
        take_active = min(nodes_needed, len(available_nodes))
        taken_active_nodes = available_nodes[:take_active]
        remaining = nodes_needed - take_active

        if remaining > 0:
            return [], []

        del available_nodes[:take_active]

        allocated = [node['id'] for node in taken_active_nodes]
        return allocated, []
