# Concurrent research coordination

The b99d0f8f final request at 16:52:57Z did not contain the CatBoost series started
at 16:52:42Z. Its later admission happened after four trials had finished.

The dispatcher now publishes reasoning assignments, active searches with exact
spaces/fixed parameters and trial counts, and queued/running experiments. Publication
happens at dialogue dispatch/release, series authorization, queue/start and evaluation.
API threads read immutable mailbox copies; they never query SQLite. Pending records
carry no fabricated score. Every response traces the active series actually delivered.

Before authorizing a tune, compare its fixed pipeline/source/task/split/seed and axes
against other active searches in the same branch. When every matching domain overlaps
at least 80% of the smaller domain, defer an unacknowledged overlap. This is an explicit
heuristic, not a proof that the studies are equivalent. Different axes/source/fixed
parameters/branches do not trigger it. Intentional overlap is permitted only with a
reason and a series ID that the provider actually received. A deferred response gets
feedback and a fresh research opportunity; it is not silently converted into a fit.

Activity revisions are distinct from measurement generations. Starting/releasing
dialogues cannot repeatedly reset the no-new-information stall detector. Successful
measurements and terminal experiment failures advance that generation.

Events: research.admission_reconsidered and research.overlap_acknowledged. Default
graph rolling execution is the primary path; legacy graph-disabled scheduling still
uses batches. Synthetic admission/overlap tests are written, not run. No claim of a
measured RMSE or wall-time gain follows from the static checks.
