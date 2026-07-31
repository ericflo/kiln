# Public site audit and copyediting plan

Status: living execution plan
Inventory snapshot: 2026-07-30
Scope: 69 public HTML routes, plus shared navigation, search, metadata, and
machine-readable discovery surfaces

This plan exists to prevent a page from being treated as complete merely
because it builds, looks polished, or contains accurate facts. Every public
page must help a specific reader reach a specific outcome, in language that is
clear, economical, current, and demonstrably true.

Copyediting is the center of this audit. Visual design, accessibility,
performance, and technical validation remain release gates, but no page can
pass without a structural edit, a sentence-level line edit, technical
verification, and a final proofread.

## 1. Outcomes

After the audit, a reader landing on any Kiln page should be able to answer:

1. What is this page about?
2. Is it meant for me?
3. What can I do or decide after reading it?
4. Which statements are current release behavior, verified source behavior,
   staged demonstration data, or historical evidence?
5. Where should I go next?

The site as a whole should:

- explain Kiln consistently without making the reader reconcile competing
  descriptions;
- lead with the answer and progressively disclose implementation detail;
- use concrete language instead of inflated product language;
- distinguish support, optimization, verification, and release status;
- make commands, examples, defaults, constraints, and failure modes
  unambiguous;
- keep reference pages exhaustive without forcing every visitor through the
  exhaustive material;
- provide a coherent path from orientation to first request, diagnosis,
  training, evaluation, and deeper reference;
- remain readable and navigable on a 390 px mobile viewport as well as a
  desktop viewport;
- give every claim a source, scope, and freshness boundary.

## 2. Canonical inventory and drift rule

The inventory has three sources:

- nine hand-authored product routes in `docs/site/`;
- the generated documentation home at `/docs/`;
- 59 generated document routes declared in
  `docs/site/docs-manifest.json`.

That produces **69 public HTML routes** in this snapshot. The generated
`sitemap.xml` is the final authority for what the assembled site publishes.
Supporting surfaces such as `llms.txt`, `search-index.json`, `robots.txt`, and
the sitemap are audited separately because they affect discovery but are not
HTML pages.

Before each campaign:

- [x] Build the assembled site from the current branch.
- [x] Count canonical HTML URLs in the generated sitemap.
- [x] Compare the static product routes, manifest documents, and sitemap.
- [x] Add every newly discovered route to the master inventory before editing
      begins.
- [ ] Remove a route only when its redirect, inbound links, canonical behavior,
      and search impact have been reviewed.
- [x] Record the new inventory date and route count at the top of this file.

Never hand-edit generated files in `_site/`. Edit `docs/site/*.html` for
hand-authored product pages, Markdown or contract sources for generated
documentation, and `docs/site/docs-manifest.json` for navigation titles and
descriptions.

## 3. Audit notation and definition of done

Each route has five independent gates:

| Gate | Meaning |
|---|---|
| **SE** | Structural edit: audience, reader job, promise, hierarchy, order, and progressive disclosure |
| **LE** | Line edit: every heading, paragraph, sentence, label, caption, table cell, note, and CTA |
| **TV** | Technical verification: claims, commands, defaults, links, examples, versions, and provenance |
| **RQ** | Rendered quality: desktop/mobile layout, navigation, accessibility, metadata, search, and performance |
| **FP** | Final proof: clean read in the deployed rendering by someone other than the primary editor |

A route is complete only when all five boxes in its inventory row are checked.
Passing automated tests does not check **SE**, **LE**, or **FP**.

### Severity

- **P0 — dangerous:** materially false, unsafe, destructive, security-relevant,
  or likely to cause data loss.
- **P1 — task-blocking:** prevents the intended reader from completing the
  page's job or substantially misstates product behavior.
- **P2 — comprehension:** confusing structure, undefined terms, stale context,
  buried answer, duplication, or weak transitions.
- **P3 — polish:** rhythm, concision, typography, metadata, minor consistency,
  or visual finish.

No page ships with an open P0 or P1. P2 findings must be fixed or explicitly
deferred with an owner and reason.

## 4. The page audit record

Create one record for each route during an audit campaign. A completed campaign
may be frozen as one dated report under `docs/audits/`; working screenshots and
temporary measurements do not belong in the repository.

```markdown
## <route> — <page title>

- Source:
- Generated from:
- Primary audience:
- Reader job:
- One-sentence page promise:
- Truth class: released | verified source | staged fixture | historical | mixed
- Primary CTA or next step:
- Last fact-checked:
- Editor:
- Final reviewer:

### Findings

| Severity | Location | Finding | Evidence | Resolution |
|---|---|---|---|---|

### Copy decisions

- Keep:
- Rewrite:
- Cut:
- Move:
- Define:
- Link:

### Verification

- [ ] SE
- [ ] LE
- [ ] TV
- [ ] RQ
- [ ] FP
```

The one-sentence promise is an editorial test, not necessarily page copy. If
the editor cannot write it without using “and” repeatedly, the page may be
serving too many reader jobs.

## 5. Copyediting protocol for every page

### Pass 1 — reader and message

- [ ] Name one primary audience. List secondary audiences only when they
      materially change the reading path.
- [ ] Name the single job the reader should complete.
- [ ] Confirm that the title and opening screen communicate that job.
- [ ] Put the answer, outcome, or working command before background and
      implementation history.
- [ ] Decide what the reader must know, may need, and can defer.
- [ ] Remove or relocate material that belongs to another page.
- [ ] Ensure the page delivers the promise made by its inbound link text,
      navigation label, title, and description.
- [ ] End with a useful next action rather than a generic list of links.

### Pass 2 — structure

- [ ] Give the page one unique H1.
- [ ] Make headings describe decisions, tasks, or content—not vague buckets
      such as “More,” “Details,” or “Other.”
- [ ] Make the heading outline understandable without reading the body.
- [ ] Keep prerequisites before actions and actions before interpretation.
- [ ] Keep warnings immediately before the action they constrain.
- [ ] Replace introductory throat-clearing with the conclusion or task.
- [ ] Convert dense comparison prose into a table only when readers need
      repeated, exact mappings.
- [ ] Convert procedures into ordered steps with observable outcomes.
- [ ] Move exhaustive fields, policy matrices, and historical forensics behind
      a clearly named deep-reference layer.
- [ ] Eliminate duplicated explanations or establish one canonical explanation
      and link to it.

### Pass 3 — sentence-level line edit

Read and edit every sentence. Do not rely on a spelling pass alone.

- [ ] Give each sentence one main idea.
- [ ] Prefer a concrete subject and active verb.
- [ ] Replace noun-heavy phrases with verbs: “perform validation of” becomes
      “validate.”
- [ ] Cut filler: “in order to,” “it should be noted that,” “the fact that,”
      “as mentioned above,” and similar scaffolding.
- [ ] Cut unearned adjectives and adverbs.
- [ ] Replace vague pronouns such as “this” and “it” when the antecedent is not
      immediate.
- [ ] Replace “supports” with the precise boundary: accepts, runs, builds,
      verifies, accelerates, or falls back.
- [ ] Define an acronym or specialist term at first use when the audience may
      not know it.
- [ ] Keep terminology stable; do not rotate synonyms merely for variety.
- [ ] Prefer positive instructions, while preserving explicit prohibitions
      where safety requires them.
- [ ] Use second person for reader actions and imperative verbs for procedures.
- [ ] Break sentences that hide conditions, exceptions, or consequences.
- [ ] Break paragraphs that mix multiple decisions or exceed a comfortable
      visual block.
- [ ] Make list items grammatically parallel.
- [ ] Give every table column a single, consistent semantic type.
- [ ] Make notes and callouts state why the information changes the reader's
      action.
- [ ] Read the edited page aloud to catch missing words, mechanical rhythm, and
      sentences that require a second pass to understand.

### Pass 4 — technical and provenance edit

- [ ] Run every command that is intended to work.
- [ ] Confirm flags, endpoints, field names, defaults, environment variables,
      filenames, response shapes, and status codes against source or generated
      contracts.
- [ ] Confirm that example output is possible and label truncated or
      illustrative output.
- [ ] Mark shell placeholders unmistakably and explain how to obtain them.
- [ ] State platform, backend, model, workload, and version boundaries where
      behavior varies.
- [ ] Separate “available in the latest release” from “verified on `main`.”
- [ ] Separate “works” from “optimized,” and “optimized” from “measured.”
- [ ] Keep benchmark values attached to metric definitions, workload,
      hardware, build identity, date, and reproduction evidence.
- [ ] Treat device names and driver versions as receipt metadata, never as
      universal product policy.
- [ ] Label staged UI fixtures and historical captures; never imply that a
      staged screenshot is a live performance run.
- [ ] Replace relative freshness claims such as “recently” or “currently”
      with a version, commit, or date when they can become misleading.
- [ ] Verify every internal, external, anchor, source, and download link.

### Pass 5 — UI copy and accessibility

- [ ] Make navigation labels predictable and consistent with destination H1s.
- [ ] Make link text meaningful without surrounding prose; avoid repeated
      “click here” or “learn more.”
- [ ] Make button labels describe the action and, when useful, the object.
- [ ] Give form labels, help text, validation errors, empty states, and success
      states the same editorial attention as body copy.
- [ ] Ensure error text states what happened, what remains safe, and what the
      reader can do next.
- [ ] Write alt text for the image's purpose in context, not a visual inventory.
- [ ] Do not repeat adjacent captions in alt text.
- [ ] Check that captions distinguish UI source, data source, capture date, and
      performance provenance.
- [ ] Confirm skip-link, menu, search, disclosure, and keyboard-control labels
      make sense when announced by a screen reader.

### Pass 6 — metadata and search copy

- [ ] Give the route a unique, accurate `<title>`.
- [ ] Make the meta description a useful summary, not a slogan or keyword list.
- [ ] Align title, H1, description, Open Graph, Twitter, navigation, sitemap,
      search-index, and `llms.txt` wording.
- [ ] Include the terms a reader is likely to search for without duplicating
      awkward synonyms.
- [ ] Ensure the canonical URL is correct and singular.
- [ ] Confirm headings and manifest descriptions expose the page's real
      subject to site search.

### Pass 7 — final proof

- [ ] Review the rendered page, not only its source.
- [ ] Proof at 390×844 and 1440×900.
- [ ] Inspect the top of the page before scrolling.
- [ ] Inspect every code block, table, callout, disclosure, caption, and footer.
- [ ] Check spelling, grammar, punctuation, capitalization, code font, and
      whitespace.
- [ ] Check orphan headings, widows, clipped text, horizontal overflow, and
      broken anchor offsets.
- [ ] Have a reviewer who did not perform the line edit read the page from
      beginning to end.
- [ ] Re-read adjacent pages to catch contradictions introduced by the edit.

## 6. Kiln editorial standard

### Voice

Kiln should sound direct, technically literate, calm, and candid. Prefer
evidence over swagger. Explain constraints without apology and advantages
without hype. Use conversational phrasing when it shortens the path to
understanding, not to manufacture excitement.

### Preferred language

- “Run,” “measure,” “verify,” “compare,” and “inspect” when those are the real
  actions.
- “Latest published release” for released behavior.
- “Verified source result at `<commit>`” for unreleased evidence.
- “Fallback” only when the fallback and its effect are named.
- “Decode tok/s,” “prefill,” “time to first token (TTFT),” and “mean
  inter-token latency (ITL)” with the workload's metric definition.
- “Vulkan-capable device” only when the stated minimum and optional capability
  behavior are accurate.

### Language to challenge

These terms are not automatically forbidden, but each use requires evidence or
a rewrite:

- best-in-class, blazing, seamless, effortless, instant, revolutionary;
- simply, just, obviously, easy;
- full support, native support, production-ready, universal;
- fast, faster, low-latency, efficient, lightweight;
- current, latest, new, recently, soon;
- all, every, never, always, guaranteed.

### Mechanics

- Use sentence-case headings.
- Use the serial comma.
- Use an em dash sparingly; prefer a full stop when it improves scanability.
- Keep UI labels exactly as rendered.
- Use backticks for literal commands, flags, fields, environment variables,
  paths, endpoints, and code identifiers.
- Use human-readable link labels and preserve the exact route in code where a
  reader must copy it.
- Use “Kiln” for the product and lowercase `kiln` only for binaries, packages,
  paths, or literal commands.
- Use “CUDA,” “ROCm,” “Metal,” and “Vulkan” consistently.
- Do not use a specific GPU or computer name as shorthand for a general
  capability class.

## 7. Page scoring and acceptance

Score each dimension from 0 to 3:

| Dimension | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Reader job | absent | inferred | stated but diffuse | immediate and singular |
| Information order | obstructive | answer buried | mostly useful | answer-first and progressive |
| Clarity | frequently opaque | repeated rereading | generally clear | precise on first read |
| Concision | substantially padded | recurring excess | minor excess | every section earns its place |
| Terminology | contradictory | inconsistent | mostly stable | canonical and defined |
| Technical truth | false/unsafe | unverified or stale | verified with small gaps | verified and bounded |
| Provenance | misleading | ambiguous | present | immediate and explicit |
| Actionability | no usable next step | incomplete | usable | observable and well linked |
| Accessibility copy | blocking | multiple weak labels | minor issues | purposeful throughout |
| Cross-site fit | contradicts journey | isolated | connected | advances a coherent journey |

Acceptance requires:

- at least 27/30 overall;
- a score of 3 for clarity, technical truth, and provenance;
- no dimension below 2;
- no open P0 or P1 finding;
- all five route gates checked.

The score is a forcing function for discussion, not a substitute for editorial
judgment.

## 8. Execution plan

### Phase 0 — baseline and ownership

- [x] Freeze the route inventory for the campaign.
- [ ] Assign an editor and independent final reviewer to every route.
- [x] Capture desktop and mobile screenshots of every route.
- [x] Record titles, descriptions, word counts, heading outlines, broken links,
      console errors, accessibility results, and performance baselines.
- [x] Build a cross-site claim ledger for releases, backend support, model
      scope, performance, and security boundaries.
- [x] Build a terminology ledger for product, training, evaluation, adapter,
      serving, backend, and metric terms.

### Phase 1 — entry and decision pages

Edit `/`, `/docs/`, `/quickstart.html`, `/demo/`, and
`/docs/benchmarks/` first. These pages establish the vocabulary, evidence
boundaries, and information architecture that downstream pages must follow.

- [ ] Complete all five gates for the five entry pages.
- [x] Lock the one-sentence product description.
- [x] Lock release/source/demo/historical provenance wording.
- [x] Lock the primary reader journeys and CTA labels.
- [x] Apply those decisions to the terminology and claim ledgers.

### Phase 2 — task pages

Edit GRPO, evals, API, CLI, architecture, and troubleshooting as complete
reader workflows.

- [x] Reconcile repeated explanations with one canonical owner page.
- [x] Run task instructions and verify examples.
- [x] Make prerequisites, success signals, recovery paths, and next actions
      explicit.
- [x] Remove reference material that interrupts the main task path.

### Phase 3 — core generated guides

Edit overview, quickstart reference, configuration, architecture, benchmarks,
security, thinking budgets, training guides, eval guide, and interoperability
guides at their Markdown sources.

- [x] Make each generated guide useful as a direct search landing page.
- [x] Reconcile product-guide and generated-guide duplication.
- [x] Keep exhaustive detail linked, not repeated.

### Phase 4 — contracts and deep reference

Edit generated schemas, OpenAPI descriptions, exhaustive references, evidence
protocols, integrity documents, and operations documents.

- [x] Edit descriptions at their schema or source-document owner.
- [x] Verify generated anchors and search descriptions.
- [x] Add orientation, examples, and links without weakening normative
      language.
- [x] Check that generated tables remain usable on mobile.

### Phase 5 — cross-site convergence

- [x] Run a single search for each canonical term and reconcile variants.
- [x] Run a single search for each claim-ledger item and reconcile scope.
- [x] Audit all navigation, related-link, previous/next, CTA, footer, and
      in-product help labels.
- [x] Audit title and description uniqueness.
- [x] Audit links, anchors, redirects, sitemap, search, `llms.txt`, and
      `robots.txt`.
- [x] Re-run desktop/mobile screenshots, accessibility checks, and performance
      measurements.
- [ ] Complete the independent proofread.
- [x] Publish a dated campaign report with remaining P2/P3 debt, owners, and
      re-audit dates.

### Phase 6 — prevent regression

- [x] Add CI inventory drift detection between static routes, the manifest,
      generated sitemap, and this checklist.
- [x] Add duplicate-title and duplicate-description checks.
- [x] Add link and anchor validation for the assembled site.
- [x] Add checks for unbounded release, support, and performance language where
      automation can identify it reliably.
- [x] Schedule a monthly freshness audit for entry/task pages.
- [x] Trigger targeted re-audits when a release, CLI/API contract, default,
      supported backend, benchmark, or UI screenshot changes.
- [x] Schedule a quarterly full-site proofread and journey audit.

## 9. Master route checklist

The “copy focus” is the first editorial question for the page, not the only
work required. Every row still receives the full seven-pass protocol.

### Product and entry pages

| Route and source | Primary copy focus | SE | LE | TV | RQ | FP |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `/` — `docs/site/index.html` | State what Kiln is, for whom, why it is distinct, and what evidence is released versus source-verified. | [x] | [x] | [x] | [x] | [ ] |
| `/quickstart.html` — `docs/site/quickstart.html` | Get a new user from install to a verified first response with no ambiguous branch or unexplained placeholder. | [x] | [x] | [x] | [x] | [ ] |
| `/grpo.html` — `docs/site/grpo.html` | Explain the training loop, prerequisites, inputs, observable progress, artifacts, and safe promotion path. | [x] | [x] | [x] | [x] | [ ] |
| `/evals.html` — `docs/site/evals.html` | Make grade, compare, replay, and training-feedback workflows distinct and executable. | [x] | [x] | [x] | [x] | [ ] |
| `/api.html` — `docs/site/api.html` | Keep the common request path immediate; make endpoint descriptions, errors, examples, and deep reference precise. | [x] | [x] | [x] | [x] | [ ] |
| `/cli.html` — `docs/site/cli.html` | Organize commands by reader task; verify flags, defaults, output, failure behavior, and API equivalents. | [x] | [x] | [x] | [x] | [ ] |
| `/architecture.html` — `docs/site/architecture.html` | Explain system flow before component detail and clearly separate capability, policy, fallback, and provenance. | [x] | [x] | [x] | [x] | [ ] |
| `/troubleshooting.html` — `docs/site/troubleshooting.html` | Start from symptoms; give diagnostic evidence, likely cause, safe action, and escalation path. | [x] | [x] | [x] | [x] | [ ] |
| `/demo/` — `docs/site/demo/index.html` | Tell one current product story and label UI source, fixture data, capture date, and performance evidence unmistakably. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/` — manifest + `scripts/docs-site/lib.mjs` | Let readers choose a workflow before exposing the full reference library; make search and category labels predictable. | [x] | [x] | [x] | [x] | [ ] |

### Start here

| Route and source | Primary copy focus | SE | LE | TV | RQ | FP |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `/docs/overview/` — `docs/public/OVERVIEW.md` | Define scope, audience, supported workflows, operational model, and a clear first path. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/quickstart-reference/` — `docs/public/QUICKSTART.md` | Reconcile with the product quickstart and keep commands, success signals, and next steps exact. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/configuration/` — `docs/public/CONFIGURATION.md` | Explain important defaults and precedence before linking to exhaustive fields; distinguish configuration from runtime policy. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/configuration-complete/` — `docs/CONFIGURATION.md` | Make an exhaustive reference searchable, non-duplicative, and explicit about canonical and retired names. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/configuration-schema/` — `contracts/kiln-config-v1.schema.json` | Make field descriptions, defaults, bounds, cross-field conditions, and migration guidance understandable in isolation. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/http-api/` — `contracts/kiln-http-api-v1.openapi.json` | Make every operation's purpose, auth, transport, request, response, errors, and ownership unambiguous. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/changelog/` — `docs/public/CHANGELOG.md` | Keep release entries scannable, user-impact first, historically accurate, and free of unreleased ambiguity. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/security/` — `SECURITY.md` | State supported versions, reporting path, response expectations, trust boundaries, and deployment responsibilities plainly. | [x] | [x] | [x] | [x] | [ ] |

### Serving

| Route and source | Primary copy focus | SE | LE | TV | RQ | FP |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `/docs/inference-schema/` — `contracts/kiln-inference-v1.schema.json` | Clarify request variants, streaming events, thinking and timing fields, batch behavior, and provenance. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/observability-schema/` — `contracts/kiln-observability-v1.schema.json` | Explain health versus readiness, diagnostic gating, model state, metrics, and cache statistics. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/architecture/` — `docs/public/ARCHITECTURE.md` | Give a concise end-to-end mental model and link each deeper implementation decision once. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/architecture-deep-dive/` — `ARCHITECTURE.md` | Explain runtime policy, request scheduling, accelerator ownership, capability-driven dispatch, learning workflows, and failure boundaries without an internal migration diary. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/benchmarks/` — `docs/public/BENCHMARKS.md` | Lead with the current comparable result; bind every number to workload, build, hardware, date, metric, and evidence. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/hf-next-token-request-schema/` — `qualification/schema/hf-next-token-request-v1.schema.json` | Explain the independent attribution request and the identity of prompt, tokenizer, model, prefix, and candidates. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/rocm-hf-next-token-result-schema/` — `qualification/schema/rocm-hf-next-token-oracle-v2.schema.json` | Make oracle provenance, logits, process evidence, artifacts, and candidate attribution explicit. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/rocm-hf-path-attribution-result-schema/` — `qualification/schema/rocm-hf-path-attribution-v2.schema.json` | Explain which numerical path is compared, how evidence is bound, and what a result can establish. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/rocm-hf-layer-attribution-result-schema/` — `qualification/schema/rocm-hf-layer-attribution-v2.schema.json` | Clarify sequential layer comparison, error growth, identities, artifacts, and interpretation limits. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/serving-benchmark-launch-schema/` — `qualification/schema/serving-benchmark-server-launch-v1.schema.json` | Make server ownership, readiness, logging, shutdown, timeout, and accepted exits operationally exact. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/serving-benchmark-protocol/` — `docs/SERVING_BENCHMARK_PROTOCOL.md` | Turn the protocol into an executable sequence and distinguish hard requirements from rationale. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/serving-profiles/` — `docs/SERVING_PROFILES.md` | Define stable, experimental, and maintenance profiles by guarantees, tradeoffs, and intended use. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/latency-observability/` — `docs/LATENCY_OBSERVABILITY.md` | Define timing metrics once and guide readers from symptom to request-local, dashboard, and Prometheus evidence. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/thinking-budgets/` — `docs/THINKING_BUDGET_CONTRACT.md` | Explain token and wall-clock limits, resolution order, streaming behavior, and client-visible outcomes. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/thinking-budget-schema/` — `contracts/thinking-budget-v1.schema.json` | Make request, resolved budget, streaming, and provenance fields precise and consistent with the narrative contract. | [x] | [x] | [x] | [x] | [ ] |

### Training and evals

| Route and source | Primary copy focus | SE | LE | TV | RQ | FP |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `/docs/native-sft-profile/` — `docs/NATIVE_SFT_PROFILE.md` | Define the fixed update, backend-owned loss route, memory admission, receipts, and checkpoint identity. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/sft-ingestion/` — `docs/sft-ingestion.md` | Explain row admission, invalid input, content identity, deduplication, and receipt consequences. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/sft-tokenization/` — `docs/sft-tokenization.md` | Make chat rendering, assistant-only labels, masking, truncation, and parity fixtures concrete. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/training-checkpoints/` — `docs/training-checkpoints.md` | Explain checkpoint contents, identity, compatibility, resume behavior, promotion, and failure recovery. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/grpo/` — `docs/GRPO_GUIDE.md` | Make group construction, scoring, updates, observability, artifacts, and promotion a coherent workflow. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/openenv/` — `docs/OPENENV_GUIDE.md` | Make discovery, seed-matched episode collection, environment rewards, direct training, identity, limits, and failure recovery one coherent workflow. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/openenv-replay/` — `docs/OPENENV_REPLAY_REFERENCE.md` | Define content-bound verification, exact live replay, protocol recovery, capacity acquisition, drift, and the implementation-neutral conformance boundary. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/openenv-schema/` — `contracts/kiln-openenv-v1.schema.json` | Define discovery identity, episode outcomes, rollout statistics, dataset hashes, and the content-addressed summary receipt. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/echo/` — `docs/ECHO_GUIDE.md` | Explain ECHO's purpose, data requirements, objective, workflow, evidence, and limitations without research shorthand. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/dataset-splits/` — `docs/DATASET_SPLITS.md` | Define split ownership, leakage prevention, train/eval separation, and promotion consequences. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/evals/` — `docs/EVAL_GUIDE.md` | Guide readers through suites, judgments, comparisons, replay, artifacts, and feedback to training. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/eval-api-schema/` — `contracts/kiln-evals-v1.schema.json` | Clarify eval requests, judgments, comparisons, statuses, artifacts, and error behavior. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/control-plane-api-schema/` — `contracts/kiln-control-plane-v1.schema.json` | Clarify training and agent lifecycle operations, state transitions, cancellation, receipts, and errors. | [x] | [x] | [x] | [x] | [ ] |

### Interoperability

| Route and source | Primary copy focus | SE | LE | TV | RQ | FP |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `/docs/hf-trl-interoperability/` — `docs/HF_TRL_INTEROP.md` | State exactly what crosses the Kiln/Hugging Face/TRL boundary, what does not, and how identity is preserved. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/artifact-api-schema/` — `contracts/kiln-artifacts-v1.schema.json` | Make artifact creation, listing, activation, deletion, identity, safety, and errors explicit. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/opd-teacher-jsonl/` — `docs/OPD_TEACHER_JSONL.md` | Define each record, ordering and tokenizer assumptions, validation, examples, and malformed-row behavior. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/vllm-teacher-identity/` — `docs/VLLM_TEACHER_IDENTITY.md` | Explain immutable teacher identity, launch provenance, architecture boundaries, and mismatch handling. | [x] | [x] | [x] | [x] | [ ] |

### Integrity and artifacts

| Route and source | Primary copy focus | SE | LE | TV | RQ | FP |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `/docs/adapter-manifest/` — `docs/ADAPTER_MANIFEST.md` | Define required identity, compatibility, provenance, validation, activation, and rejection fields. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/base-weight-provenance/` — `docs/BASE_WEIGHT_PROVENANCE.md` | Explain how base weights are identified, verified, recorded, compared, and rejected. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/execution-provenance/` — `docs/EXECUTION_PROVENANCE.md` | Explain which runtime, device, process, build, and policy facts are captured and what they prove. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/request-lineage-integrity/` — `docs/REPLAY_INTEGRITY.md` | Make lineage, replay authority, mutation boundaries, hashes, and failure semantics understandable. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/training-receipt-schema/` — `docs/TRAIN_RECEIPT_SCHEMA.md` | Define receipt fields, lifecycle, identities, evidence, validation, and interpretation limits. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/qualification-workload-schema/` — `qualification/schema/workload-v1.schema.json` | Make workload identity, inputs, constraints, expected evidence, and comparison rules exact. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/qualification-case-result-schema/` — `qualification/schema/case-result-v1.schema.json` | Explain case status, measurements, evidence, failures, environment identity, and comparability. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/qualification-receipt-schema/` — `qualification/schema/receipt-v1.schema.json` | Explain receipt aggregation, authority, signatures or hashes, cases, environment, and verdict. | [x] | [x] | [x] | [x] | [ ] |

### Operations and development

| Route and source | Primary copy focus | SE | LE | TV | RQ | FP |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `/docs/hardware-qualification/` — `docs/qualification.md` | Provide a reproducible qualification sequence, expected artifacts, pass criteria, and failure triage. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/backend-latency-evidence/` — `docs/backend-latency-result-schema.md` | Define latency evidence, metric boundaries, workload identity, environment, and valid comparisons. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/runtime-environment-inventory/` — `docs/RUNTIME_ENVIRONMENT_INVENTORY.md` | Separate recognized, direct-read, passthrough, unsafe, retired, and provenance-only environment variables. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/artifact-retention/` — `docs/ARTIFACT_RETENTION.md` | State what is retained, where, for how long, why, and how deletion or archival works. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/verification-test-inventory/` — `docs/VERIFICATION_TEST_INVENTORY.md` | Map each guarantee to tests and explain scope, platform gaps, evidence, and update responsibility. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/contributing/` — `CONTRIBUTING.md` | Give a new contributor a reliable setup, change, test, documentation, and review path. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/ci-policy/` — `docs/ci-policy.md` | Explain which checks run where, why jobs skip, what local qualification proves, and how to respond to failure. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/verification-policy/` — `docs/VERIFICATION_POLICY.md` | Define evidence classes, required gates, ownership, exceptions, and the boundary between test and claim. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/release-version-policy/` — `docs/release-version-policy.md` | Define version ownership, allowed references, release drift checks, historical exceptions, and update procedure. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/tensor-substrate-quickstart/` — `docs/SUBSTRATE_QUICKSTART.md` | Get a developer from tensor creation to a verified operation while explaining device, dtype, shape, and errors. | [x] | [x] | [x] | [x] | [ ] |
| `/docs/backend-capabilities/` — `docs/backend-capability-report.md` | Present the capability matrix with definitions, constraints, source ownership, generation date, and interpretation limits. | [x] | [x] | [x] | [x] | [ ] |

## 10. Shared and non-page surfaces

These checks do not replace any route row:

- [x] Header and mobile menu labels match route titles and reader vocabulary.
- [x] Footer links are complete, useful, and consistently ordered.
- [x] Documentation sidebar categories and labels match reader intent.
- [x] “On this page” labels match actual headings and remain concise.
- [x] Previous/next and related-link copy describes the destination.
- [x] Search placeholder, empty state, result title, description, and no-result
      guidance are edited.
- [x] Search results include all 56 manifest documents and no unpublished
      source.
- [x] `sitemap.xml` contains every canonical HTML route exactly once.
- [x] `llms.txt` and any full-text companion use current, bounded product copy
      and canonical source links.
- [x] `robots.txt` allows the intended public surface and points to the correct
      sitemap.
- [x] Open Graph and social preview copy match the current positioning.
- [x] Favicon, logo, screenshots, and social assets have current surrounding
      copy and provenance.
- [x] GitHub Pages' not-found behavior gives a stranded reader a useful path
      back to Kiln, even if a custom 404 must be added.

## 11. Validation commands

Use a fresh output directory so stale generated pages cannot hide omissions:

```bash
npm ci --prefix scripts/docs-site
npm test --prefix scripts/docs-site

audit_site_out="$(mktemp -d)"
node scripts/docs-site/build.mjs --out "$audit_site_out"
KILN_DOCS_SITE_ROOT="$audit_site_out" \
  KILN_DOCS_REQUIRE_GENERATED=true \
  node scripts/check_docs_site_smoke.mjs

git diff --check
```

Also run the repository's release-version, configuration, API-contract, and
documentation-link checks used by the Pages workflow. For the rendered pass:

- capture every route at 390×844 and 1440×900;
- run keyboard and screen-reader smoke checks;
- run automated accessibility checks, treating zero automated findings as a
  starting point rather than proof;
- run Lighthouse or equivalent performance, accessibility, best-practice, and
  SEO audits on all templates and every high-traffic route;
- fetch the deployed URLs after Pages completes and confirm that the public
  content, canonical URLs, assets, and cache headers match the audited build.

## 12. Re-audit triggers and cadence

| Trigger | Required scope |
|---|---|
| Any release | Home, quickstarts, changelog, configuration, API, CLI, benchmarks, download/install copy, metadata |
| API or schema change | Owning product page, generated contract, examples, search description, related troubleshooting |
| CLI or default change | Quickstarts, CLI, configuration, troubleshooting, examples |
| Backend/capability change | Home claims, architecture, configuration, troubleshooting, capabilities, benchmarks |
| New performance receipt | Home evidence, benchmarks, architecture claims, demo disclaimers where relevant |
| Dashboard change | Demo tour, screenshots, captions, alt text, product descriptions |
| Training/eval behavior change | GRPO, evals, API, schemas, relevant generated guides and receipts |
| Monthly | Entry pages, task pages, links, search, release/source freshness |
| Quarterly | All 66 routes, shared surfaces, end-to-end journeys, independent proofread |

The route inventory must grow with the site. A new page is not complete until
it has a source owner, audience, reader job, copy focus, five-gate checklist
row, and re-audit trigger.
