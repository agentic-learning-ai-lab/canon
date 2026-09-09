"""Vendored from ReViWo common/utils.py; only ALL_ENVIRONMENTS is used by the collectors."""

ALL_ENVIRONMENTS = [
    "assembly-v2", "basketball-v2", "button-press-v2", "door-open-v2",
    "window-close-v2", "drawer-open-v2", "dial-turn-v2", "soccer-v2",
    "handle-pull-side-v2", "reach-v2",
]
ALL_ENVIRONMENTS += [
    'reach-v2', 'push-v2', 'pick-place-v2', 'button-press-v2', 'door-unlock-v2',
    'door-open-v2', 'window-open-v2', 'faucet-open-v2', 'coffee-push-v2', 'coffee-button-v2',
]
ALL_ENVIRONMENTS = sorted(list(set(ALL_ENVIRONMENTS)))
