- **The unreached-surface ratchet caught pgw#1425's own fix within the hour, which is the
  instrument working.** `receipts.configured()` lost its only caller when `posture()` replaced it,
  so it is DELETED rather than baselined — two spellings of one state is how a caller ends up
  asking a two-answer question about a three-state gate, and "not configured" collapsing `local`
  into `unset` is exactly the fail-open pgw#1425 closed. pgw#1421's two new unreached callables
  are added as rows WITH AN OWNER AND AN EXPIRY (the tool's own prescribed remedy for a row that
  is not your commit) rather than guessed at.
