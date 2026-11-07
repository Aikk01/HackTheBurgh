# Sample hands

Excerpts of the [gabbler/PokerData](https://huggingface.co/datasets/gabbler/PokerData) dataset
(CC-BY-4.0) so the simulator works offline.

| File | Source in the dataset | Notes |
|------|-----------------------|-------|
| `dwan-ivey-2009.phh` | `dwan-ivey-2009.phh` | Full Tilt Million Dollar Cash Game, first televised $1M pot (NLHE, antes) |
| `antonius-blom-2009.phh` | `antonius-blom-2009.phh` | Largest online pot, pot-limit Omaha |
| `phua-xuan-2019.phh` | `phua-xuan-2019.phh` | Triton London 2019, short-deck hold'em |
| `pluribus-session-30.phhs` | `pluribus/30/*.phh` | 80 hands of Pluribus vs. professionals, all cards known, finishing stacks recorded |
| `wsop-2023-ppc-final-table-holdem.phhs` | `wsop/2023/43/5/*.phh` | The 25 hold'em-family hands (NT, PO, FT) of the 2023 $50k PPC final table |
| `handhq-pokerstars-200nl-sample.phhs` | `handhq/PS-.../2/ps NLH handhq_1-OBFUSCATED.phhs` | First 150 anonymised PokerStars 200NL cash-game hands (July 2009) |
| `handhq-ipoker-100nl-hero-cards.phhs` | `handhq/IPN-.../1/ipn NLH handhq_1-OBFUSCATED.phhs` | 120 iPoker 100NL hands in which the hero's hole cards are recorded |

Caveat for model evaluation: in the iPoker logs the hero's cards are only present in hands where
the hero never folded (196 of 985 hands in the source file, zero folds among them), so agreement
rates on that sample are biased towards continuing hands. Some iPoker records also stop right after
the river card (no river action, zero winnings); the replayer marks those "incomplete".
