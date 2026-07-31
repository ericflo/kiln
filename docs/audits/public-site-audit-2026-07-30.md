# Public site audit — 2026-07-30

Campaign plan:
[`docs/plans/public-site-audit-and-copyediting-plan.md`](../plans/public-site-audit-and-copyediting-plan.md)

## Campaign status

The primary editorial and rendered-quality passes are complete for the 66
canonical HTML routes in the 2026-07-30 inventory:

| Gate | Routes complete | Status |
|---|---:|---|
| Structural edit (SE) | 66/66 | complete |
| Sentence-level line edit (LE) | 66/66 | complete |
| Technical verification (TV) | 66/66 | complete |
| Rendered quality (RQ) | 66/66 | complete |
| Independent final proof (FP) | 0/66 | pending |

Primary editor: Codex, working from the repository source and rendered site.
The repository maintainer must assign an independent reviewer before checking
FP. The primary editor must not clear that gate.

The canonical inventory comprises nine hand-authored product routes, the
documentation home, and 56 manifest-driven document routes. The custom
`404.html` is audited as a noncanonical recovery surface, not counted as a
67th canonical route.

## Findings resolved

| Severity | Surface | Finding | Resolution |
|---|---|---|---|
| P1 | `/docs/benchmarks/` | Different throughput metrics and historical results were presented without a usable narrative, obscuring a severe Vulkan regression. | Rebuilt the page around the current measured position, a dated regression timeline, metric definitions, the exact reproduction command, and bounded evidence links. It now states that the tracked decode workload fell from the historical teens to 0.142 tok/s—about an 87× regression—and recovered to 13.46 tok/s on the verified source correction. |
| P1 | Vulkan support copy | Machine provenance could be mistaken for runtime policy, and the v0.5.1 global route table assumed one fast route set was legal everywhere. | Made the support boundary explicit throughout the site: route legality derives from reported Vulkan capabilities and request constraints. Device names, vendor/device IDs, PCI identity, and driver names are evidence metadata only. |
| P1 | `/demo/` | The tour mixed an old terminal cast with current product claims and did not identify current, staged, or archived evidence. | Replaced the primary tour with a current 2026-07-30 capture and source-backed fixture boundary. Retained the older cast only as a clearly labeled historical artifact. |
| P1 | `/docs/backend-capabilities/` | A dense generated inventory read like runtime verification and overstated host-fallback coverage. | Reworked the generator and page into an answer-first static source inventory. Added explicit status definitions, open coverage work, collapsed deep tables, and a regression test that forbids the previous overclaim. |
| P2 | Generated references | Large schema tables were difficult to use on mobile. | Added responsive table wrappers and labeled row-card rendering while preserving exact field values and constraints. |
| P2 | Documentation navigation | The closed mobile sidebar remained in the keyboard focus order while off canvas. | The sidebar is now inert whenever the mobile menu is closed, becomes interactive when opened, and returns to inert state on Escape or breakpoint changes. Browser tests lock the state transitions and focus restoration. |
| P2 | Site-wide accessibility | Skip links on hand-authored pages were entirely outside the viewport even when keyboard-focused. Several small labels and controls also missed comfortable mobile sizing. | Normalized visibly focused skip links, raised the smallest text to 12 px, enlarged documentation table-of-contents and demo-footer targets, and added an accessibility-tree shell audit across all 66 routes plus the 404 page. |
| P2 | Navigation and recovery | Footers differed by page, benchmark links diverged, and GitHub Pages had no useful custom recovery page. | Standardized the 15-link product footer, routed benchmark links to the public guide, and added a responsive, `noindex` custom 404 with task-oriented destinations. |
| P2 | Search and discovery | Search empty-state guidance was weak, and inventory drift or duplicate metadata could go unnoticed. | Copyedited the no-result message and added exact search/sitemap counts, route-checklist parity, unique title/description, robots, canonical, link, and anchor checks. |
| P3 | Assets and page weight | Nine superseded screenshots and social images remained in the public asset directory. | Removed the unreferenced tracked assets and added a smoke assertion that prevents them from returning. The assembled site is approximately 7.7 MB; the public asset directory is approximately 2.0 MB. |

No open P0 or P1 documentation finding remains in the edited source. Product
coverage gaps that the source itself reports remain visible rather than being
converted into support claims.

## Claim ledger

| Claim area | Approved boundary | Canonical evidence owner | Re-audit trigger |
|---|---|---|---|
| Product | Kiln combines OpenAI-compatible local inference, adapter training, evaluation, and promotion in one server. Individual backend, model, and workflow limits remain explicit. | `README.md`, `docs/public/OVERVIEW.md`, and the HTTP contract | Product-scope or API change |
| Release status | Use “latest published release” only for a published tag. Label newer repository behavior “unreleased on main” or “verified source result” with a revision. | `Cargo.toml`, GitHub releases, `docs/public/CHANGELOG.md`, and `scripts/check_release_versions.py` | Release or version change |
| Backend availability | A declared Cargo feature means a backend can be selected at build time; it does not prove every operation, device, or workload. | `docs/backend-capability-report.md` generated from backend sources | Backend feature, predicate, or fallback change |
| Vulkan dispatch | Select routes from reported API, workgroup, shared-memory, descriptor, push-constant, subgroup, and memory-topology capabilities. Never select from machine or vendor identity. | `crates/kiln-vulkan-kernel/src/policy.rs`, `crates/kiln-vulkan-kernel/src/device.rs`, and backend contract tests | Vulkan policy or device-probe change |
| Vulkan performance | The tracked short single-stream diagnostic measured 13.46 decode tok/s at source revision `f3ae29e4a`. It is neither a cross-device result nor a published-release guarantee. | `docs/public/BENCHMARKS.md` and linked receipts/logs | Benchmark receipt, workload, source, driver, or route change |
| Demo evidence | Current captures show a dated UI state. Seeded examples demonstrate flow, not production measurements. Archived casts remain historical. | `docs/site/demo/index.html` and its linked capture/source artifacts | Dashboard or demo-fixture change |
| Security | Security copy states boundaries and reporting paths; it does not claim that local operation eliminates application, model, dependency, or operator risk. | `SECURITY.md` | Authentication, network, storage, dependency, or reporting-policy change |

## Terminology ledger

| Term | Site-wide usage |
|---|---|
| Kiln | Product name. Use lowercase `kiln` only for a binary, package, command, or path. |
| backend | A runtime implementation such as CPU, CUDA, ROCm, Metal, or Vulkan. A build feature is not an unbounded support guarantee. |
| capability | A reported or typed condition used to decide whether a route is legal. Do not substitute a device family or qualification-machine name. |
| native route | An operation executed by the selected accelerator implementation. Name constraints and fallbacks when they affect behavior. |
| fallback | A named alternate path with its correctness and performance effect. Avoid the word without stating where execution moves. |
| adapter | A finalized LoRA artifact with identity and provenance. Use “active adapter” for the artifact currently serving. |
| promote | Make an evaluated adapter the selected serving candidate through the documented control path. Do not use as a synonym for merely saving a checkpoint. |
| evaluation | A declared dataset, grader, candidate set, and result artifact. Distinguish grade, compare, and replay workflows. |
| receipt | Machine-readable evidence for a bounded run. A receipt proves only the workload, environment, source, and metrics it records. |
| decode tok/s | `1000 / mean inter-token latency in milliseconds` for a defined request path. Do not compare it directly with request-window throughput. |
| prefill | Prompt processing before decode. Report time or throughput with prompt length and workload identity. |
| TTFT | Time to first token: request arrival through the first emitted token. |
| current | Use only with a date, release, source revision, or another freshness boundary. |

## Verification evidence

The following checks passed against the edited source:

- documentation build tests: 11 passed;
- assembled-site smoke and browser checks, including all 66 canonical routes
  plus `404.html`;
- exact sitemap inventory: 66 canonical URLs, each present once;
- exact search inventory: 64 searchable entries—eight product pages and 56
  manifest documents;
- configuration, HTTP API, runtime-environment, runtime-default, release,
  thinking-budget, artifact, eval, control-plane, and observability contract
  checks;
- repository artifact and production-file-budget checks;
- `cargo fmt --all -- --check`;
- backend capability contract tests: 22 passed;
- documentation quickstart tests: 2 passed;
- training-receipt rejection test: 1 passed;
- `git diff --check`.

Every canonical route was inspected at 390×844 and 1440×900. The permanent
mobile browser pass checks document language, one H1, main and skip landmarks,
visible heading order, duplicate IDs, image alternatives, horizontal overflow,
closed-sidebar interactivity, skip-link keyboard activation, and accessible
names for interactive roles.

Lighthouse 12.8.2 was run against the five highest-risk entry routes after the
accessibility fixes:

| Route | Mobile: performance / accessibility / best practices / SEO | Desktop |
|---|---|---|
| `/` | 92 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| `/docs/` | 95 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| `/quickstart.html` | 90 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| `/docs/benchmarks/` | 93 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| `/demo/` | 98 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |

The measurements used a local static server and are comparative template
checks, not production latency measurements.

## Remaining gates and scheduled follow-up

| Item | Severity | Owner | Due or trigger |
|---|---|---|---|
| Read every deployed route from beginning to end and clear FP only where no issue remains. | release gate | Independent reviewer, not the primary editor | Before this campaign is declared complete |
| Fetch and inspect the production Pages deployment, including canonical URLs, assets, navigation, search, 404 behavior, and representative cache headers. | release gate | Primary editor | After merge and Pages deployment |
| Re-run entry/task freshness, links, search, contract, and rendered-shell checks. | recurring | Documentation owner | 2026-08-30, then monthly |
| Re-run all 66 route journeys and independent final proof. | recurring | Documentation owner plus independent reviewer | 2026-10-30, then quarterly |
| Import broader device and workload receipts before making cross-device Vulkan performance claims. | product evidence, not a documentation defect | Backend/performance owner | New qualification evidence or performance claim |

The Pages workflow now builds and checks the full assembled site on every pull
request, on relevant source changes to `main`, and on a monthly schedule.
Deployment is skipped for pull-request runs.
