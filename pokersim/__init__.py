"""pokersim - an interactive, data-driven poker simulator.

* ``engine``     - a No-Limit Texas Hold'em state machine (also PLO, short-deck
                   and fixed-limit hold'em) with side pots, all-ins and showdowns.
* ``phh``        - a parser/writer for the Poker Hand History (PHH) format used by
                   the ``gabbler/PokerData`` Hugging Face dataset.
* ``replay``     - replays recorded hands through the engine, step by step.
* ``agents``     - the model interface plus built-in bots, an HTTP adapter and a
                   Claude adapter.
* ``evaluate``   - shadow-evaluates a model against recorded human decisions.
* ``simulate``   - self-play between models with PHH export.
* ``server``     - the web UI / JSON API.
"""

__version__ = "0.1.0"
