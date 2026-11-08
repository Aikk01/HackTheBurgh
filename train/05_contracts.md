# 05 · Contracts

All model files are optional at run time: a missing file gives `None`, and the screen shows "No model loaded".

## Model files (`data/rl/`, committed; a few MB each)

| File | Produced by | Read by |
|---|---|---|
| `hand_range.npz` + `.json` | `scripts/train_hand_range.py` | `pokersim/hand_range.py`: grids, RL inputs |
| `next_move.npz` + `.json` | `scripts/train_next_move.py` | `pokersim/next_move.py`: Base rows, priors, pro/plur bots |
| `policy.npz` + `policy.json` | Copied from the chosen run's `export_heldout/` | `pokersim/rl_policy.py` (reads `use_ranges` from the json) |

## Functions

```python
# pokersim/hand_range.py
predict_ranges(obs, opponents) -> {index: np.ndarray[169]}        # hero's cards are dead cards, never features
grid(probs) -> 13x13 list; CLASS_NAMES[k] -> "AKs"; live_combos(dead) -> np.ndarray[169]

# pokersim/next_move.py
predict(obs, j, with_cards=False, pluribus=False) -> np.ndarray[7] | None
#   with_cards=True only for j == obs["hero"]["index"]; for other players the legal actions are built hypothetically
#   ("if they had to act now")

# pokersim/rl_policy.py
recommend(obs, styles) -> {"summary": {fold, call, raise}, "probs": [...], "ranges": [...], "value_bb", ...}

# pokersim/you_model.py  (to build)
predict(player_id, obs, j) -> {"probs": np.ndarray[7], "prior": "next_move" | "population"}
update(player_id, obs, j, action_index) -> None
score(player_id) -> {"right": int, "total": int, "baseline_right": int}
```

Action classes, in this order everywhere: `fold, call, raise_min, raise_half, raise_pot, raise_2pot, all_in`.
Groups: fold = 0, call = 1, raise = 2–6.

## `/api/game/advice` additions (existing fields stay)

```json
{
  "opponents": {
    "3": {"name": "p3", "colour": "#e07a5f",
          "base": {"next": {"action": "call", "p": 0.41}},
          "this_player": {"next": {"action": "fold", "p": 0.55}, "decisions_seen": 37},
          "hand": {"summary": {"pair_plus": 0.35, "draw": 0.20, "nothing": 0.45}, "top": [{"class": "KQo", "p": 0.05}],
                   "grid": [[...13]], "density": [[...13]]}}
  },
  "you": {
    "should": {"base": [{"action": "call", "p": 0.48}, {"action": "raise_half", "p": 0.30}],
               "rl":   [{"action": "raise_pot", "p": 0.55}, {"action": "raise_half", "p": 0.30}], "rl_value_bb": 6.2}
  }
}
```

"What you would do" is not in `/advice`, because advice is unavailable once it is no longer your turn. It is returned
by `/api/game/act` and kept in the game state as `you.last`:
`{"predicted": "call", "p": 0.58, "actual": "call", "right": true, "score": {...}}`.
