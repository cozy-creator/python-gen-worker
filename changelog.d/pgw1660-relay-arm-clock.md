- **A pgw#1660 relay arm read the wall clock per snapshot and failed about 1 run in 5.** The arms
  in `test_lifecycle_relay_pgw1660.py` compare projections for CHANGE; two snapshots built a
  millisecond apart differ in their intents' `since`/`updated` stamps without differing in
  anything the comparison is about, so `test_an_unchanged_projection_is_not_resent` reddened for a
  reason that says nothing about the relay. Pinned to a fixed clock — 20/20 green after, and the
  arm still goes red when the resend suppression is actually removed.
