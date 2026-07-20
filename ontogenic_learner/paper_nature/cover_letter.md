# Cover letter — draft (for manual review)

*Target: Nature (flagship). Retargeting notes for Nature Machine Intelligence /
Nature Electronics at the bottom. Fill author/affiliation placeholders before use;
run the added neuroscience citations through the citation-verification workflow.*

---

**To the Editors of *Nature*,**

We submit our manuscript, *"One dynamics for perception and plasticity:
predictive coding that learns as it serves,"* for consideration as an Article.

Deployed neural networks do something brains never do: they separate acting from
learning into two processes, on different hardware, at different times — an
inference server that answers the stream and a training pipeline that periodically
revises the weights, coupled by storage, synchronization and considerable energy.
We ask whether that separation is necessary and answer, by construction, that it
is not.

On a tropical (max-plus) sequence model, credit assignment reduces to re-tracing
the winning computation the forward pass has already performed. A single process
can therefore serve every input and, whenever it is surprised, convert that same
forward pass into an exact learning step — with no second pass, no stored
activations, and no multiplication in the reverse sweep. We validate this fused
loop along a pre-registered ladder: it captures most of the achievable adaptation
benefit on shifting streams, sustains a ten-million-example lifetime in one
process at roughly 1.006× serving cost with bounded forgetting, and on
reconfigurable (FPGA) silicon the marginal power of learning-while-serving falls
below the instrument's floor.

We believe the advance is of broad interest for two reasons. First, it collapses
a deployment assumption — the train/deploy dichotomy — that pervades modern AI,
and shows that continual, auditable, on-device learning can be an occasional
property of serving rather than a second system. Second, and central to this
collaboration, **the loop is a concrete, hardware-realized instantiation of
predictive coding**: serving is inference, a learning update is the same
computation with the output clamped, and the architecture makes three falsifiable
predictions for cortex — that a useful "learn-now" surprise signal must
decorrelate from stimulus variance rather than track it; that exact credit
assignment can proceed without symmetric feedback weights; and that what a synapse
consolidates is set by the precision of its predictions, not by how much it has
recently changed. Our team spans machine learning, computer architecture, and
systems/perception neuroscience (B. He), and we have written the paper to speak to
all three.

We have been deliberate about scope. The learning rule we implement is provably
equal to backpropagation, so we claim a predictive-coding *model with testable
predictions*, not a novel biological rule; our results are at a controlled scale
(~1M parameters); the on-chip energy result is reported as a measured floor rather
than a board-level win; and two pre-registered negative results — including the
honest gap between our fully closed optimizer and Adam — are part of the record.
We think this candor strengthens, rather than weakens, the claim that the fused
loop holds up when its parts are measured honestly.

The manuscript has not been published elsewhere and is not under consideration by
another journal. We declare no competing interests. We would be glad to suggest
referees spanning machine learning, neuromorphic/hardware, and predictive-coding
neuroscience, and to exclude any the editors deem appropriate.

Thank you for your consideration.

Sincerely,
[Corresponding author], on behalf of all authors
[Affiliation(s)]

---

## Retargeting notes

- **Nature Machine Intelligence** — replace the "broad interest" paragraph's first
  half with an emphasis on *on-device continual learning, auditable/replayable
  adaptation, and the systems removal of the actor/learner split*; keep the
  predictive-coding paragraph; soften the flagship-level novelty framing.
- **Nature Electronics** — lead with the sub-1 W, multiply-free, serve-class-memory
  learning on reconfigurable silicon (the FPGA result) as the headline; the ML/PC
  results become supporting; add that board-level and ASIC instrumentation are the
  planned next step.
- **Title carries the predictive-coding advance; the multiply-free / serve-class /
  energy-floor claims are stated precisely in the abstract** (not the title),
  because the flagship life runs an AdamW optimizer while only the fully closed
  optimizer variant makes the whole loop multiply-free (Extended Data Table 2).
