# s18 - s14 + partial ATLAS, Hungarian kept (submitted 2026-10-28 ~07:52 UTC, ref 57729649, NTL slot 4/5 of 10-28)

Trigger: s15 (s14 + soft-cap tau 0.5) scored **0.633**, below s14 (Hungarian) **0.643**. Soft-cap marginal decoding
hurts on the real stack (CPU proxy predicted +0.012..+0.037 - the proxy over-trusts the recovered test clusters).
s16/s17 are both built on s15 (soft-cap), so they likely inherit the -0.010.

Change: s17 notebook with `--post_mode softcap --post_tau 0.5 --post_min 2` replaced by `--post_mode hung`
(public dense-cluster Hungarian). Only diff vs s14 = partial ATLAS (no all-or-nothing skip on projected overrun).

Read: s18 - s14 = value of partial ATLAS on the Hungarian stack; compare with s17 - s15 for consistency.
Zero extra GPU: commit run is the short-circuit (~3 min), full pipeline only in the scoring rerun.
