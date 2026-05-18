# ECHO: Terminal Agents Learn World Models for Free

**Authors:** [Dimitris Papailiopoulos](https://x.com/DimitrisPapail) and Vaishnavh ([@VaishShrivas](https://x.com/VaishShrivas))

**Affiliation:** AI Frontiers, a boutique research lab inside Microsoft Research

**Source (blog/thread):** [https://x.com/DimitrisPapail/status/2056368948870811746](https://x.com/DimitrisPapail/status/2056368948870811746)

**Paper PDF:** [https://github.com/microsoft/echo-rl/blob/main/echo.pdf](https://github.com/microsoft/echo-rl/blob/main/echo.pdf)

**Code:** [https://github.com/microsoft/echo-rl](https://github.com/microsoft/echo-rl) (built on top of [SkyRL](https://github.com/NovaSky-AI/SkyRL))

**First cluster run:** March 29

---

## TL;DR

We taught CLI agents to predict terminal responses during RL, alongside the usual GRPO loss on actions. The change is tiny: same rollout and forward pass, but stop masking out terminal-output tokens. The effect is huge: all evals improve, and the resulting models measurably learn how the terminal behaves.

> **CLI agents can learn a terminal model for free — and use it to act better!**

This is **ECHO**: a hybrid objective that trains on both sides of the interaction — what the agent writes, and what the terminal writes back.

If you're too busy to read this whole post, here's what we found:

- **Standard agent RL throws away the environment's response.** GRPO trains on action tokens and masks out terminal responses, even though they are already in context, already pass through the model, and are ground truth signals on how the agent's actions affected the environment.
- **ECHO fixes this** by training on both sides of the interaction. It keeps the usual GRPO loss on action tokens, and adds a simple environment cross-entropy loss on terminal-output tokens. It's a few LoCs on top of any GRPO trainer. Same rollout and forward pass, just a different mask over the logits.
- **ECHO works, and it's free!** ECHO improves Qwen3-8B, OpenThinker-Agent-v1-SFT, and Qwen3-14B across every benchmark we tested. ECHO also trains up to **2.3× faster** to the same performance. **TerminalBench-2.0 pass@1 nearly doubles** at both 8B (2.7 → 5.2) and 14B (5.2 → 10.8).
- **ECHO teaches terminal dynamics!** On held-out trajectories, environment-token cross-entropy drops sharply with ECHO and barely moves with plain GRPO. Direct evidence that ECHO teaches the model how the terminal actually responds. The same checkpoints that better predict terminal outputs also solve more tasks.
- **ECHO can substitute for an expert teacher.** From a base Qwen3-8B with no expert demonstrations, ECHO nearly matches what GRPO after SFT on expert demonstrations achieves.
- **ECHO lets agents self-improve without verifier rewards!** Without any verifier rewards, ECHO (without any GRPO) allows the agent to further improve just by acting in the environment and predicting what happens.

This started as a simple question: *if every command produces a terminal response, why does RL train only on the command?*

Vaish did all the work to figure it out. Dimitris contributed a silly maze experiment, a strong opinion on the title, and saying "holy shit" when she showed him the first result. Thanks to **Ahmed Awadallah** for giving us space — and GPUs — to chase ideas like this, even when they start as just a research itch.

---

## How do you continually learn?

The first time this idea showed up it was motivated by a simple question on self-improvement and continual learning. How does an agent get better just from acting in the world?

Vaish and Dimitris had been talking since last Fall about self-improvement in CLI agents — i.e., what it would mean to get better from interacting with the environment (aka the terminal), especially without a verifier.

Verifier-free RL is a problem people have worked on for years, and most of the attempts at it face the same issue: *where does supervision come from, if there's no reward?*

Around the same time, a silly Twitter post led to a call with [@willccbb](https://x.com/willccbb) discussing continual learning again. During this chat, the silly idea was:

> Maybe continual learning is to train on whatever the environment gives you back as a response to your actions.

That should teach the model something, right?

**Turns out, yes!**

---

## The world is a loss function!

When an agent acts in an environment, the environment's response to that action is always true.

An example from the physical world: if you flip a light switch, the light comes on, or it does not. If it doesn't, that is a legitimate response: it tells you something about the bulb, or the wiring, or the breaker, etc. Either way, what comes back is a small piece of information of how the world changed because of your actions on it. You are not exposed to the full mechanism of how electricity, switches, and bulbs work, but you do see the outcome. *Did the light turn on?* And that is sufficient for you to start building a mental model of how flipping switches turns lights on.

The terminal works kind of in the same way.

The output after a bash command is a small summarization of how the state of the computer/container changed after the command was run. You see **stdout, stderr, exit codes, file listings**, etc. You don't see the kernel state or the process tree or anything super detailed.

> **Figure.** *An illustration of the agent ↔ terminal loop. The agent issues a command; the terminal returns a low-dimensional projection of how the underlying system state changed (stdout/stderr/exit code/listings). The agent uses this response to pick its next action.*

What you see back is a **low-dimensional projection** of what happened in the background — which is also what a CLI agent will use to choose a next action toward whatever task they are trying to achieve. And as with the light switch, that is enough signal to build a mental model — or if you will, a **world model** — of how the system behaves.

The best part is that the terminal output, which again is a reflection of how the system's state changed, is a supervision signal, **computed for you, on every single turn, for free.**

Cool!

The problem is that standard agent RL (e.g. GRPO in [SkyRL](https://github.com/NovaSky-AI/SkyRL)) pushes gradients only through action tokens and ignores terminal output tokens — in spite of the fact that the terminal output is already in the context. The model attends to it, the forward pass computes logits for it, yet the trainer masks it out of the loss.

What a waste of good tokens 😊

So, what if we didn't?

The model is already conditioned on those tokens. It already produces a probability distribution over them. Adding a cross-entropy loss costs essentially nothing.

And if we do… the model has reason to learn how the terminal actually behaves and can therefore build, inside itself, an implicit model of the system it is acting on. To predict what `ls` will return, the model has to track which files it just created, what lives where, etc.

As Ilya put it:

> Predicting the next token well means that you understand the underlying reality that led to the creation of that token.

In our setting, that would mean: an agent that is good at predicting terminal outputs has, in a small but real sense, built an implicit model of the terminal.

So how do we get the agent to predict terminal outputs?

---

## ECHO: Learn a model of the world without paying for it

A terminal-agent rollout already interleaves two streams of tokens: the agent's **action tokens** and the environment's **observation tokens**. Standard GRPO applies loss only on the action tokens.

This is especially wasteful because terminal rewards are sparse, delayed, and binary. In our Qwen3-8B setting, for many tasks **fewer than 15% of on-policy rollouts are successful**. But failed trajectories are not failed data: they still contain file listings, errors, logs, stack traces, grep outputs, and other consequences of the agent's commands.

Our method is the most embarrassingly simple way of learning from those consequences 😊

We add a length-normalized cross-entropy loss on environment-observation tokens, alongside the standard GRPO loss on action tokens. **ECHO** is the hybrid objective:

$$
\mathcal{L}_{\text{ECHO}} = \mathcal{L}_{\text{GRPO}}(\text{Actions}) \;+\; \lambda \cdot \mathcal{L}_{\text{env}}(\text{Observations})
$$

where **Actions** are the agent-action positions and **Observations** are the terminal-output positions.

### A few technical tidbits

- **ECHO learns on-policy.** Instead of training on a frozen set of terminal transcripts from the base model or a teacher, ECHO learns from terminal responses produced by the current model during RL. As the agent gets better, it explores new parts of the environment and gets fresh supervision from new action → observation transitions. Better policies induce better feedback; better feedback prediction gives the policy better action priors. *A loop! how fun?*
- **In the joint objective, $\lambda$ matters.** If it's very small, the environment loss does not shape the model much. If too large, the policy can optimize for predictable outputs instead of task progress. *Gotta balance it out!*
- **The target tokens matter.** We train on the actual terminal output, not harness warnings. Warnings are easy to memorize; the useful signal is the actual terminal response — filenames, stack traces, and error messages.

### So what does this cost?

An astute reader might ask:

> isn't the backward pass more expensive if you're back-propagating gradients at more token positions?

**Almost not.** The expensive part of backprop is the matmuls through attention and MLP layers, and those run over the same token sequence regardless of which output positions contribute to the loss. The logits at every response position are already computed for GRPO. The action mask and the observation mask just gather different subsets of them for different loss terms.

Pause here for a second: **We added a world modeling loss, and the cost is basically ZERO.** No extra rollouts, teacher model, and no extra forward pass.

---

## Does ECHO help train a better CLI Agent?

We ran the cleanest comparison possible on multi-turn terminal tasks: same models, same GRPO recipe, same tasks, same rollout and turn budget, same number of training steps. Reward = 1 if the agent passes the test cases after $n$ turns, 0 if it fails.

The only difference is whether terminal-output tokens also enter the loss.

> **Figure.** *Training curves on three held-out validation sets. Pink = ECHO, teal = GRPO. Across model sizes and evaluation slices, the pink curves separate from the teal early on and generally stay above.*

ECHO consistently improves performance across all three held-out validation sets — the pink curves separate from the teal early on and generally stay above.

ECHO also learns substantially faster: **ECHO matches GRPO's 500-step performance on Terminal-Bench Lite 280 steps faster!** A **2.3× speedup** and it keeps climbing 😊

These results substantiate our intuition behind ECHO. GRPO trains only with sparse, binary outcome rewards. For hard domains like terminal tasks where pass-rate is low for small models, this translates to little or no signal for many tasks.

ECHO makes training **much more sample efficient by turning failed actions into supervision**. Even when an action does not solve the task, the terminal response still teaches the model what that action caused! And predicting the consequences of failed actions can help the agent choose better ones.

If you'd rather see the numbers across evals, same story in table form:

> **Figure (table).** *Pass@1 across Qwen3-8B / OT-SFT / Qwen3-14B on TBLite, TB2, ITD, and val100, comparing GRPO vs ECHO at the final checkpoint. The last row in each block (ECHO) consistently improves over GRPO. TerminalBench-2.0 pass@1 nearly doubles at 8B (2.7 → 5.2) and 14B (5.2 → 10.8) scales.*

Look at the last row in each block: **ECHO**. TerminalBench-2.0 pass@1 nearly doubles at 8B (2.7 → 5.2) and 14B (5.2 → 10.8) scales. And importantly, this is **not from extra data, rollouts, a teacher model, or a different verifier**. The rollout already contained the terminal response. ECHO just learns from it.

> "Performance nearly doubles at no extra cost" is a line you very rarely read across your whole research career 😊.

ECHO substantially beats GRPO performance on all benchmarks and model sizes, is much more sample efficient, and costs basically nothing. You learn a world model as your policy improves, which helps it improve faster.

Skeptics though may push back: *do you really learn a world model?*

Let's see!

---

## Does ECHO actually learn terminal dynamics?

We're gonna hedge a little here because the world-modeling crowd can get a little intense.

We won't claim ECHO learns a world model in the strongest sense. But we **will** claim that ECHO trains a policy whose hidden states have absorbed something about how the terminal behaves, and whose ability to predict what the terminal will do has measurably improved.

If you reverse Ilya's quote, you get a more falsifiable version. For our setting it would be something like this:

> If the model has learned terminal dynamics it has to be good at predicting terminal output.

Because there is no other way to consistently assign high probability to the right tokens. A model that's a better predictor is, in information-theoretic terms, a better compressor of the system it's predicting.

So the question becomes empirical: *does ECHO actually make the model a better predictor of terminal output?*

**Yes. By a lot.**

To make this test clean, we use a stronger teacher model **Qwen3-32B** (not used in any of our training runs) to generate trajectories for each of our validation sets. Then we evaluated our starting policies, policies trained with GRPO, and policies trained with ECHO and measured how "surprised" each model was by the resulting terminal-output tokens.

The pattern is the same on every panel: GRPO barely changes environment-token cross-entropy relative to the starting policy. **ECHO sharply lowers it.**

> **Figure.** *Environment-token cross-entropy on held-out trajectories generated by Qwen3-32B. Across every panel, GRPO barely changes the cross-entropy relative to the starting policy; ECHO sharply lowers it.*

So we won't say *world model* with a capital W. But we will say this:

> ECHO produces policies that are measurably better at compressing terminal dynamics, on trajectories they didn't generate.

Which is the operational version of the claim the title is making, and the version that is fully defensible.

---

## Surprising finding 1: ECHO reduces dependence on expert SFT

A common recipe for agent RL is: first behavior-clone expert trajectories from a stronger model, then run RL. This is especially common for terminal agents, where the reward is sparse and the action space is huge.

In our setting, the expert-SFT baseline is **OpenThoughts-Agent-v1-SFT (OT-SFT)**: Qwen3-8B fine-tuned on terminal-agent demonstrations generated by a stronger **GLM-4.6** teacher.

So we asked: *how much of that expert-SFT benefit can ECHO recover without behavior-cloning the teacher?*

**Can ECHO let you skip expert SFT? In our setting, mostly yes!**

> **Figure.** *Three runs compared: plain GRPO on the base model, ECHO on the base model, and GRPO on the SFTed model (SFT + GRPO).*

Relative to the gap between GRPO and SFT+GRPO (i.e., the gain starting from SFT provides), ECHO recovers:

- **104%** of the gain on **ITD**
- **89%** on **Terminal Bench Lite (TBLite)**
- **50%** on **TerminalBench-2.0 (TB2)** pass@1

The result suggests that **a large part of expert SFT's value may come from teaching the model an interaction prior, not just an expert strategy prior**. Expert demonstrations show both how to behave like a terminal agent — inspect files, run tests, follow tracebacks, etc — **and** what an expert would do in specific states. ECHO does not imitate those expert choices. Instead, it trains the model to predict the terminal consequences of its own actions, so it learns which commands expose useful state, which errors are diagnostic, and which terminal output tokens signal progress. Better strategies can then emerge through interaction rather than imitation.

This also helps interpret the benchmark split. On ITD and TBLite, ECHO nearly matches expert SFT, suggesting much of the SFT advantage there comes from a better model of terminal interaction. On TB2, ECHO still recovers a substantial 50% of the gap without demonstrations. The remaining gap is consistent with TB2 being harder and distributionally farther away from the training set.

We would not treat this as a fixed ceiling: broader or longer training on TB2-like tasks should improve the agent further.

So the takeaway is **not** that expert SFT is obsolete, but that **much of what expert SFT buys may be a better model of terminal interaction, and that part can be learned directly from the environment**.

> **Bottom line: The terminal is the teacher!**

---

## Surprising finding 2: Sparks of self-improvement without rewards

So far ECHO has been GRPO with an auxiliary environment loss. The verifier still tells the agent whether it solved the task, and GRPO updates the model on action tokens. So standard RL setup, with a small extra term.

But if ECHO is genuinely teaching the policy something about how the terminal behaves, then maybe we don't need the verifier signal at all.

We ask: *What happens if we turn the verifier off?* No rewards to learn from, just this:

$$
\mathcal{L} = \mathcal{L}_{\text{env}}(\text{Observations})
$$

That is — the model acts, observes, updates only by predicting the terminal outputs as a consequence of its own actions.

This sounds like it should not improve task performance. There is no label saying which action was good. If the policy gets better, it has to be because learning to predict the terminal **indirectly reshapes the policy's action priors**.

So we tried it!

We took our strongest **Qwen3-8B + ECHO** checkpoint, removed the GRPO term entirely, and trained for **100 more steps on held-out tasks using only the environment cross-entropy loss**. The question was whether the model could improve on OOD tasks it had never seen before, purely by interacting with the environment and predicting what came back.

Did this insane idea work? **Kinda did!**

> **Figure.** *Env-only training improvements across three evaluation sets after 100 steps of training with only the environment-cross-entropy loss.*

- On **val100** (in-distribution): **+3.8 pp**
- On **ITD**: **+5.2 pp**
- On **PyTerm** (a held-out OOD set of Python-heavy terminal tasks): **+10.0 pp** *after filtering to clean tool-call trajectories*

Env-only training improves the policy when the terminal output is useful supervision. With no reward signal, the model trains only to predict the outputs caused by its own actions, so gains depend on whether those outputs expose useful dynamics.

- On **val100**, which is close to the training mix, the gain is real but small: +3.8 pp before saturation. The policy has already learned most of the local dynamics during ECHO training.
- On **ITD**, the weaker starting policy produces noisy trajectories — invalid commands, parse errors, dead-end loops. Filtering to clean rollouts denoises the signal and gives +5.2 pp.

Clean trajectories alone aren't enough, though. The same filtering did not consistently improve TBLite, while PyTerm started from a similar pass rate but improved under the same recipe — suggesting the bottleneck is not just policy strength. The key difference is **how informative the observations are**: Python tasks give dense action-linked feedback — code → traceback → fix — while broader terminal tasks reveal state more indirectly through files, configs, and multi-step setup.

We believe that verifier-free adaptation is possible: **once RL has produced a decent exploring model, the agent can sometimes keep improving from consequences alone — but only when its rollouts are clean and the terminal feedback is informative**. That IS the surprising part. Not that the agent self-improves perfectly, but that it self-improves at all, from nothing but acting and predicting what comes back.

---

## Where this leaves us

ECHO's core lesson is simple: **agent rollouts contain more supervision than just the final reward, and we should use it.**

Every command an agent runs produces a terminal response — stdout, errors, traces, files, logs, etc — and standard RL uses those tokens only as context for the next action. ECHO turns them into training targets. **No teacher model, extra rollouts, or separate world model needed.** We just stop throwing away the environment tokens that are already in the transcript.

That small change led to three surprising results:

1. **Stronger RL performance.**
2. **Much less dependence on expert SFT.**
3. **In some settings, verifier-free self-improvement from environment interaction alone.**

We do not think this means rewards or demonstrations are obsolete. Expert trajectories still teach strategy and verifiers provide the cleanest task-level signal. But ECHO suggests that **between "imitate the expert" and "wait for sparse reward," there is a dense, underused source of supervision: the consequences of the agent's own actions.**

The broader idea is a continuation of auxiliary prediction which has a long history in RL, and recent work has revived world-modeling objectives for LLM agents, e.g.:

- **Agent Learning via Early Experience** — uses action-consequence signal as a pre-RL stage.
- **VAGEN** — adds a world-modeling reward for VLM agents.
- **RWML** — pre-trains on next-state prediction.
- **CWM** — mid-trains a code model on observation-action trajectories.

**ECHO is the online, in-the-RL-loop, CLI-flavored version of the same idea.**

---

## How far can this idea go?

The next step is to make this environment signal more potent — and test how far it generalizes. ECHO uses raw terminal outputs because they are already in the rollout, but the best learning target may be a cleaner and more compact representation: summaries or task-relevant views of state. Also:

- Which observations should we train on?
- When should we filter trajectories?
- How should we weight environment prediction against policy optimization?
- Can the same idea work beyond terminals — for browser agents, multi-tool systems, long-horizon coding agents, or user-facing assistants where follow-ups, corrections, and preferences are another form of interaction feedback?

Our bet is that **anywhere an agent acts and the world responds in tokens, those response tokens — or better representations of them — should be part of the learning signal**. ECHO is the simplest version of that idea we could think of, and we suspect some form of environment-token prediction will be standard in agent RL trainers by the end of 2026.

Check out the [full paper](https://github.com/microsoft/echo-rl/blob/main/echo.pdf) and [code](https://github.com/microsoft/echo-rl) on top of SkyRL.

> *Try ECHO out and tell us how much faster your agent trained.*

---

## Footnote: training a maze world model on my laptop… kinda

Remember when Dimitris said he "contributed a silly maze experiment"? Here's the silly maze experiment.

The setup was a teeny tiny version of ECHO: a grid maze in a tiny terminal. The agent (a 10M-parameter transformer in a loop) issues a direction — up, down, left, right — and the terminal responds with where the agent is with respect to its "neighbors" (it's basically a path-finding-in-a-2D-grid problem), and the distance to the destination. So the rollout looks exactly (for small values of *exactly*) like a CLI agent rollout, just much simpler 😊:

> action → environment response → action → environment response → …

Two conditions were tested on a from-scratch 10M-parameter transformer:

1. Train only on the **action tokens**.
2. Train on the **action tokens and the terminal's response** (neighbors, distance, etc.).

All trained on fresh 6×6 / 7×7 / 8×8 mazes.

> **Figure.** *|ECHO − action-only| solve rate, by maze size. Positive means ECHO solves more mazes than the action-only baseline at the same compute. Basically this silly figure says that ECHO works on mazes.*

Is this maze thingy a Nature paper? No. But: there's a point that keeps generalizing.

> Almost every clean idea has a microcosm: a scaled-down version you can run on a laptop in an evening that tells you whether the idea is worth scaling up.

The maze didn't prove ECHO would work. It gave enough conviction to send Vaish a Teams message instead of forgetting about the idea. It turned out Vaish had been circling around the same idea independently, and when her first cluster run came back with results, Dimitris was thrilled and genuinely surprised. The ECHO maze had hinted that the direction was right, but it could not have predicted **doubling TerminalBench scores, recovering most of expert SFT, or self-improvement without rewards**. Those were Vaish's results.

"Kinda solving a 6×6 maze" and "doubling on TerminalBench" are very different epistemic states.

But the point of this addendum isn't that the laptop replaces the cluster experiment. The point is that **most ideas are wrong** and the laptop experiment (with the help of Claude Code and Codex) tells you which ones to drop before they cost anyone else time. Once in a while, an idea survives, and when it does it can perhaps earn the right to a collaborator's time and GPUs.

**ECHO is one of them.**

---

## Links

- **Paper PDF:** [https://github.com/microsoft/echo-rl/blob/main/echo.pdf](https://github.com/microsoft/echo-rl/blob/main/echo.pdf)
- **Code:** [https://github.com/microsoft/echo-rl](https://github.com/microsoft/echo-rl)
- **Blog/thread:** [https://x.com/DimitrisPapail/status/2056368948870811746](https://x.com/DimitrisPapail/status/2056368948870811746)
- **Built on:** [SkyRL](https://github.com/NovaSky-AI/SkyRL)

---

## Related work referenced

- **Agent Learning via Early Experience** — action-consequence signal as a pre-RL stage.
- **VAGEN** — world-modeling reward for VLM agents.
- **RWML** — pre-training on next-state prediction.
- **CWM** — mid-training a code model on observation-action trajectories.
- **SkyRL** — base GRPO trainer that ECHO is implemented on top of.
- **OpenThoughts-Agent-v1-SFT (OT-SFT)** — Qwen3-8B fine-tuned on terminal-agent demonstrations generated by GLM-4.6 (the expert-SFT baseline).
- **Qwen3-8B / Qwen3-14B / Qwen3-32B** — the model family used for the policy (8B, 14B) and as the held-out teacher for cross-entropy evaluation (32B).
- **GLM-4.6** — teacher behind the OT-SFT expert-SFT baseline.
- **TerminalBench Lite (TBLite) / TerminalBench-2.0 (TB2) / ITD / val100 / PyTerm** — the held-out evaluation sets.

---

## Citation

> Papailiopoulos, D. and Vaishnavh. *ECHO: Terminal Agents Learn World Models for Free.* AI Frontiers, Microsoft Research, 2026. [https://github.com/microsoft/echo-rl](https://github.com/microsoft/echo-rl)

*Conversion note: this file is a faithful Markdown rendering of the public ECHO write-up (Twitter/X thread linked above), reorganized for the `docs/papers/` archive. Figures referenced in the original post are summarized as blockquoted captions because the underlying images were not part of the source provided.*
