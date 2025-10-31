# SPARS/Gym/translators/scalar_active_target.py
# NOTE: Logic unchanged; comments added only.
import logging
import torch as T

logger = logging.getLogger("runner")


def action_translator(logits, state, current_time):
    """
    Args:
      logits: [N,2] or [1,N,2] tensor/array/list
              left  -> switch_off score
              right -> switch_on  score
      state : list of per-node dicts with keys: 'state', 'job_id'
    Returns:
      events: [{'time':..., 'event': {'type': 'switch_off', 'nodes': [...]}},
               {'time':..., 'event': {'type': 'switch_on',  'nodes': [...]}}]
               (omitted if empty)
    """
    x = T.as_tensor(logits)

    # Drop batch dim if [1,N,2]
    if x.dim() == 3 and x.size(0) == 1:
        x = x[0]
    if x.dim() != 2 or x.size(1) != 2:
        raise ValueError(
            f"Expected logits shape [N,2] or [1,N,2], got {tuple(x.shape)}")

    # Align lengths if state/logits mismatch
    N = x.size(0)
    if len(state) != N:
        M = min(N, len(state))
        x = x[:M]
        state = state[:M]
        N = M

    # Raw decisions (ties ignored)
    left = x[:, 0].detach()
    right = x[:, 1].detach()
    switch_on = (right > left).nonzero(
        as_tuple=False).squeeze(1).cpu().tolist()
    switch_off = (left > right).nonzero(
        as_tuple=False).squeeze(1).cpu().tolist()

    # Build current sets
    current_idle = []                # active & job_id is None
    current_inactive = []            # state == 'inactive'
    # (active & job_id != None) OR switching_{on,off}
    unable_to_make_action = []

    for i, n in enumerate(state):
        st = str(n.get('state', '')).lower()
        jid = n.get('job_id', None)

        if st == 'active' and jid is None:
            current_idle.append(i)
        if st == 'inactive':
            current_inactive.append(i)
        if (st == 'active' and jid is not None) or (st in ('switching_on', 'switching_off')):
            unable_to_make_action.append(i)

    unable = set(unable_to_make_action)
    idle_set = set(current_idle)
    inactive_set = set(current_inactive)

    # Apply filters:
    # - ignore any node in 'unable'
    # - switch_on only if currently inactive
    # - switch_off only if currently idle
    switch_on = [i for i in switch_on if i not in unable and i in inactive_set]
    switch_off = [i for i in switch_off if i not in unable and i in idle_set]

    events = []
    if switch_off:
        events.append({'time': current_time, 'event': {
                      'type': 'switch_off', 'nodes': switch_off}})
    if switch_on:
        events.append({'time': current_time, 'event': {
                      'type': 'switch_on',  'nodes': switch_on}})

    return events
