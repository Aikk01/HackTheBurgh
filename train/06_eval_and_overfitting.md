# 06 · Evaluation and overfitting

| Risk | Guard |
|---|---|
| RL fits the training bots (an earlier run lost 894 bb/100 to an unseen rule bot) | Selection only on held-out tables (Pluribus clones + new rule bots), paired per hand on the same deals; the final report uses fresh deals |
| RL fits its own past copies (cycling) | Past copies capped at 25% of seats; a falling held-out score while in-distribution rises is flagged (08) |
| Base models memorise players | Online data split by player; a separate later shard range (34–36) for test; early stopping on validation; test read once |
| Hand model showdown bias | Quote the Pluribus-only figures as the honest ones (+0.31 nats) |
| Personal model fits 50 decisions | Dirichlet shrinkage towards the next-move prior; scored before each action; shown next to the usual-move baseline with its count |

## How the RL numbers are read

- A difference is real only if its **paired** 95% interval excludes 0.
- Every claim is against bots. The held-out Pluribus clones are copies of strong human play, not strong humans.
- The in-distribution tables include very weak bots (calling stations), so numbers like +3,687 bb/100 only say the
  policy found their weaknesses.
- "How much the policy relies on the hand inputs" comes from zeroing them. "Whether predicting hands helps" comes
  from the matched v3-ranges / v3-noranges comparison at equal updates.
