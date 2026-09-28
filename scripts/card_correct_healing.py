"""Rewrite the healing bullet on an already-emitted card after the per-expert revert.

`s06_emit` writes the card before `s09_eval` runs and before anything is uploaded, so a card on
disk describes whatever healing was in force at emit time. The pass-2 cards claim per-expert
healing; that correction has since been measured worse and reverted, so the claim is now false
about the weights it ships with.

Rather than duplicate the wording, this calls `s06_emit._heal_note` - the same function the
emitter uses - so a card corrected here and a card emitted fresh cannot say different things.

It also appends a delimited Corrections section, because a reader who pulled this repo yesterday
deserves to know the weights changed under the same name, and because the negative result is more
useful to a passer-by than the correction itself.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

CBEGIN = "<!-- CORRECTIONS:BEGIN -->"
CEND = "<!-- CORRECTIONS:END -->"

CORRECTIONS_T = """{CBEGIN}
## Corrections to this repository

**2026-08-29 — the healing correction was replaced, and the weights changed.** If you downloaded
this checkpoint before this date you have the superseded version; re-pull to get the current one.

This checkpoint originally shipped a **per-expert** least-squares healing correction: one
coefficient per retained expert, fitted in closed form to reproduce the unpruned layer's output
under post-prune routing. It reduced held-out reconstruction residual in 41 of 42 layers, which is
why it shipped.

An end-to-end ablation then measured it against the **per-layer scalar** it had replaced — same
checkpoint, same 241,516 held-out tokens, same cached teacher, one variable moved:

| metric | per-expert (was shipped) | per-layer scalar (ships now) | Δ |
|---|---|---|---|
| top-1 agreement | 0.83693 | **0.84249** | **+0.00556** |
| ΔNLL vs teacher | 0.19396 | **0.17563** | −0.01833 |
| top-k KL | 0.69388 | **0.65030** | −0.04358 |

{parent}Better in every sufficiently-sampled domain. Both arms score the identical token set, so the
correct test is paired: McNemar over the 13,020 discordant tokens (5,839 the per-expert arm gets
right and the scalar does not, 7,181 the other way) gives **χ² = 138.1, z = 11.8**. Healing is a
multiply on the F32 block scales and is exactly invertible, so the fix was a rescale, not a
re-prune — no expert was re-selected and the mask is unchanged.

**Why it lost.** Under the measured near-orthogonality of expert outputs (off-diagonal mass 1.9%)
the coefficient reduces to `c_j = (gate mass expert j received before pruning) / (gate mass it
receives after)`. The intended reading was that an expert promoted into the top-8 by pruning is
doing work it never did before and should be shrunk. But it is doing that work *because the expert
that used to do it was deleted* — damping the substitute does not bring the original back, it
leaves a hole. 62% of the 5,760 coefficients landed below their layer's scalar; 45 experts were
suppressed more than 2×, the worst by 3.3×, and by construction those are exactly the experts the
pruned router depends on most.

**A second correction follows from the same measurement.** Pass 1 shipped scalar healing and
pass 2 shipped per-expert, so the two were never comparable. Holding healing fixed, the pass-2
mask is worth **+0.00545** over pass 1 (McNemar z = 9.3) — not the ≈0 previously reported here. The two
changes were real effects of opposite sign that had cancelled into a null.

Full analysis, including why a held-out reconstruction proxy improved while the model got worse:
`research/HEALING_ABLATION.md` in the pipeline repository.
{CEND}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    import stages.s06_emit as E
    adapters = json.loads((ROOT / "output" / "adapters" / "first_moment_gains.json").read_text())
    meta = {
        "heal_gain_median": st.median(adapters["gains"].values()),
        "heal_measured": True,
        "heal_per_expert_rejected": bool(adapters.get("reverted_per_expert")),
    }
    if adapters.get("per_expert_layers"):
        raise SystemExit("the adapter record still describes per-expert healing - revert first")
    note = E._heal_note(meta)
    # The healing correction lives in the FP8 parent; a derived quantisation inherits it. Say so,
    # or a reader takes the FP8 figures in the table below for this repository's own.
    parent = ("" if "nvfp4" not in a.name else
              "Those figures are measured on the **FP8 parent** this checkpoint is quantised "
              "from, which is where the healing correction lives. This repository's own measured "
              "numbers are in the evaluation section above; the NVFP4 build moved from 0.83294 "
              "to **0.83857** across the same correction.\n\n")
    corrections = CORRECTIONS_T.format(CBEGIN=CBEGIN, CEND=CEND, parent=parent)

    card = ROOT / "output" / a.name / "README.md"
    txt = card.read_text()

    lines = txt.splitlines(keepends=True)
    hits = [i for i, l in enumerate(lines) if l.startswith("- Healing is ")]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one healing bullet in {card}, found {len(hits)}")
    lines[hits[0]] = note + "\n"
    txt = "".join(lines)

    # Drop any interim notice: the corrected weights are what this card now describes.
    if "Correction in progress" in txt:
        # The notice is one contiguous blockquote; drop it line by line rather than by locating
        # its last sentence, which is the kind of match that silently eats the rest of a card.
        out, dropping = [], False
        for l in txt.splitlines(keepends=True):
            if l.startswith("> ### ") and "Correction in progress" in l:
                dropping = True
            elif dropping and not l.startswith(">") and l.strip():
                dropping = False
            if not dropping:
                out.append(l)
        txt = "".join(out)

    if CBEGIN in txt and CEND in txt:
        txt = txt.split(CBEGIN)[0] + corrections + txt.split(CEND, 1)[1]
    else:
        txt = txt.rstrip() + "\n\n" + corrections + "\n"
    card.write_text(txt)
    print(f"corrected healing claim on {card}")


if __name__ == "__main__":
    main()
