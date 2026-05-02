# Kiln launch post drafts

This directory stages per-channel launch announcement drafts for Phase 11
(public-announce). The Pages workflow only renders top-level
`docs/site/*.html` and `docs/site/assets/*` — this directory is **not** built
or deployed; it's a versioned drafting space.

**Companion artifacts:**

- The live launch blog post: [`docs/site/launch.html`](../launch.html)
  (deployed at https://ericflo.github.io/kiln/launch.html, shipped in PR #667)
- The demo asciicast: live at
  [`docs/site/demo/`](../demo/) (recording script in
  [`SCRIPT.md`](../demo/SCRIPT.md), standalone player in
  [`index.html`](../demo/index.html), canonical recording in
  [`kiln-60s.cast`](../demo/kiln-60s.cast), and the reference shell driver
  in [`demo.sh`](../demo/demo.sh)). Embedded at the standalone demo page
  and ready to be linked from `docs/site/launch.html`.

## Status

> **All drafts in this directory are DRAFT status.**
> **Demo has landed. Do not post any draft until Eric reviews it and the
> v0.2.13 pre-launch ops gate remains green.**

## Files

| File | Channel | Use for |
| --- | --- | --- |
| [`hackernews.md`](hackernews.md) | HN (Show HN) | The biggest single-shot traffic spike. Submit URL = `launch.html`. First comment posts immediately after submission. |
| [`twitter.md`](twitter.md) | X / Twitter | An 8-tweet thread with image/asciicast suggestions and a post-thread reply template for the "is this a wrapper?" question. |
| [`lobsters.md`](lobsters.md) | lobste.rs | Skeptical technical audience. Long-form comment that front-loads the honest tradeoffs. |
| [`reddit-localllama.md`](reddit-localllama.md) | /r/LocalLLaMA | Hobbyist + enthusiast audience. Emphasizes the consumer-GPU angle and includes the GRPO Python snippet inline. |
| [`discord-rust.md`](discord-rust.md) | Rust Discord `#showcase` | Rust-systems framing — vendored CUDA kernels, in-process training thread, build matrix. |
| [`anthropic-discord.md`](anthropic-discord.md) | Anthropic Discord | Claude / AI-product-builder framing — live online learning, GRPO reward loops, hot-swap LoRA, single-GPU economics. |

## Recommended posting order and timing

The order matters. Earlier posts should set up the discussion that later posts
join — and the HN thread is the highest-leverage single venue, so it goes
first while everything else is fresh. Run the launch within a single 24-hour
window:

1. **Hacker News (Show HN)** — Tuesday or Wednesday, **9:00–11:00 ET**.
   The single highest-traffic shot. Submit, then post the first comment
   within 60 seconds.
2. **X / Twitter thread** — same day, **within 30 minutes of the HN
   submission**. Pin tweet 1 to the profile for the launch week.
3. **lobste.rs** — same day, **9:00–11:00 ET**. Either right after the X
   thread or staggered ~1 hour after HN to avoid lobsters seeing it as a
   pure HN cross-post. Lobsters explicitly dislikes "saw it on HN, here it
   is" submissions.
4. **/r/LocalLLaMA** — same day, **9:00–11:00 ET** (catches both US morning
   and EU afternoon). Stagger ~30 minutes after lobsters.
5. **Rust Discord `#showcase`** — same day, **12:00–14:00 ET** (US
   lunchtime, EU late afternoon). Discord is conversational; stay in the
   channel for ~1 hour to answer questions.
6. **Anthropic Discord** — same day or next morning as a targeted
   AI-builder follow-up after Rust Discord. Lead with Claude-in-the-loop
   evaluation, GRPO reward loops, hot-swap LoRA, and the single-GPU product
   economics rather than Rust internals.

If HN goes hot (front page), pause channels 3–6 by an hour or two so they
don't compete with the HN thread for Eric's attention. If HN flops, still
ship 2–6 — they reach different audiences and recovery from a bad HN day is
boring not fatal.

## After-posting checklist

For each channel, after going live:

- [ ] Add the live post URL back to that channel's draft file as
      `posted_url:` in the frontmatter (so future agents can find it).
- [ ] Update `status:` from `draft` to `posted`.
- [ ] Watch for the first 30 minutes (Discord) / 90 minutes (HN, X) /
      2 hours (lobste.rs, Reddit). The early window is where the founder
      presence matters most.
- [ ] Reply to every substantive question. One-line replies are fine for
      "great work!" comments, but technical questions get full answers.
- [ ] If the same question shows up across channels, copy the best answer
      back into this README under "FAQ" so future agents can reuse it.

## Cross-channel boundaries

- **Don't post the same body across channels.** Each draft is rewritten for
  its audience — HN gets crisp tradeoffs, lobste.rs gets long-form honesty,
  /r/LocalLLaMA gets the consumer-GPU angle, Rust Discord gets the systems
  angle, Anthropic Discord gets the Claude / AI-product-builder online-learning
  angle, X gets one-line hooks. Reusing a body across channels reads as spam.
- **Don't link channels at each other.** "Saw this on HN, posting here too"
  triggers downvotes on lobste.rs and Reddit.
- **Don't autopost.** Every submission is manual, by Eric, after review.

## Pre-launch ops gate (must be green before any draft goes live)

- `gh release view kiln-v0.2.13 -R ericflo/kiln` — clean
- `gh attestation verify` against the latest release artifact — clean
- `gh api repos/ericflo/kiln` — public, README rendered, links work
- https://ericflo.github.io/kiln/ and https://ericflo.github.io/kiln/launch.html
  — render correctly on mobile and desktop
- The demo asciicast — embedded on `launch.html` and visible in the page

### 2026-05-02 v0.2.13 verification

- [ ] `gh release view kiln-v0.2.13 -R ericflo/kiln` reports
      `isDraft=false`, `isPrerelease=false`, and includes the expected release
      assets.
- [ ] `gh attestation verify kiln-0.2.13-x86_64-unknown-linux-gnu-cuda124.tar.gz -R ericflo/kiln`
      verifies SLSA provenance from `https://github.com/ericflo/kiln`.
- [ ] GHCR `kiln-server` package includes tags `0.2.13` and `latest`; verify
      with `gh api orgs/ericflo/packages/container/kiln-server/versions` or the
      package UI before posting.

### 2026-05-02 v0.2.12 verification

- [x] `gh release view kiln-v0.2.12 -R ericflo/kiln` reports
      `isDraft=false`, `isPrerelease=false`, published at
      `2026-05-02T08:30:14Z`, and 7 release assets.
- [x] `gh attestation verify kiln-0.2.12-x86_64-unknown-linux-gnu-cuda124.tar.gz -R ericflo/kiln`
      verifies SLSA provenance from `https://github.com/ericflo/kiln`.
- [x] `gh api repos/ericflo/kiln` reports `private=false` and public repo URL
      `https://github.com/ericflo/kiln`.
- [x] GHCR `kiln-server` package includes tags `0.2.12` and `latest`.
- [x] `https://ericflo.github.io/kiln/`,
      `https://ericflo.github.io/kiln/launch.html`, and
      `https://ericflo.github.io/kiln/demo/` return HTTP 200 and render the
      expected titles: `Kiln — Your model gets better every time you use it`,
      `Launching Kiln — Your model gets better every time you use it`, and
      `Kiln Demo — 60-second online-learning loop`.
- [x] `docs/site/launch.html` links to `demo/`, and
      `docs/site/demo/index.html` embeds `kiln-60s.cast`.

## Where this directory came from

Companion to PR #667 (live launch blog) and the landed demo asciicast. Staged
so Eric can review the channel-specific framing before any public post goes
live.
