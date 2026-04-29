# Security Policy

## Reporting a vulnerability

**Please do not open a public issue or pull request for security bugs.** Public reports tip off attackers before a fix is available.

Use one of these channels instead:

- **Preferred:** GitHub's private vulnerability reporting on [`ericflo/kiln`](https://github.com/ericflo/kiln) — open the **Security** tab, then **Report a vulnerability**. This keeps the discussion private and gives us a structured place to coordinate a fix.
- **Fallback:** email <floguy@gmail.com> with a subject prefixed `[kiln security]`. Plain text is fine; PGP is not currently set up.

## What to include

A useful report has:

- The affected version or commit SHA (Kiln is pre-1.0; `main` moves fast, so a SHA is more useful than a tag).
- A minimal reproduction — the exact command line, request body, training payload, or config that triggers the issue.
- An impact assessment — what an attacker can do, who is exposed, and under which deployment shape.
- A suggested mitigation if you have one. Not required, but it speeds things up.

## Response expectations

Kiln is maintained by one person right now. Realistic SLAs:

- **Initial acknowledgment:** within 7 days.
- **Triage and status update:** within 30 days of the acknowledgment.
- **Fix or coordinated disclosure plan:** depends on severity and complexity; we will keep you in the loop rather than go silent.

If you do not hear back within 7 days, send a follow-up — it almost certainly means the report got lost, not ignored.

## Supported versions

Kiln is pre-1.0. **Only the latest tagged release on `main` is supported** — fixes land on `main` and ship in the next release; there are no backports to older minor versions. See [`CHANGELOG.md`](CHANGELOG.md) for the current release.

Once a v1.0 release exists, this policy will be revisited.

## In scope

Security issues we want to hear about:

- Remote code execution via training inputs, prompt content, adapter files, or config.
- Prompt-injection attacks that escape any sandboxing the server claims to provide.
- Authentication or authorization bypass on management endpoints (adapters, training, admin).
- Exfiltration of model weights, LoRA adapters, or training data across tenancy or trust boundaries that the server is supposed to enforce.
- Denial of service that crashes the server or wedges it into an unrecoverable state from a single malformed request.
- Supply-chain compromise — a malicious dependency, a tampered release artifact, or a compromised CI workflow.

## Out of scope

Some things look like vulnerabilities but are documented design choices. Please do not file these as security issues:

- **No HTTP authentication.** Kiln's HTTP API currently has no built-in auth. The deployment model assumes a trusted network (loopback, VPN, or a reverse proxy that adds auth). "I can call `/v1/train/sft` without a token" is the documented behavior.
- **Training data and adapters at rest are unencrypted.** Disk encryption is delegated to the host filesystem.
- **Model outputs are not safety-filtered.** Kiln runs the model you load and returns whatever it generates.
- **Self-DoS via expensive-but-legitimate requests** — long contexts, large training batches, or many concurrent generations can saturate a single GPU. That is a capacity-planning concern, not a security bug.

If you think one of the above *should* be in scope, open a normal issue or discussion to argue the design — that is the right channel for it.

## Supply-chain provenance

Every `kiln-v*` release ships with [build provenance attestations](https://docs.github.com/actions/security-guides/using-artifact-attestations) so you can verify that an artifact was produced by this repository's GitHub Actions workflow and not tampered with after the fact. The attestation is a Sigstore-signed in-toto statement linking the artifact's SHA-256 digest to the workflow run, commit, and source repository.

To verify a downloaded binary tarball or zip:

```sh
gh attestation verify kiln-<version>-<target>.tar.gz --repo ericflo/kiln
```

To verify the published Docker image:

```sh
gh attestation verify oci://ghcr.io/ericflo/kiln-server:<tag> --repo ericflo/kiln
```

Both commands require [`gh`](https://cli.github.com/) 2.49 or newer. They exit non-zero if the artifact's digest does not match a recorded attestation, so they are safe to drop into a deployment script before unpacking the tarball or pulling the image.

## Disclosure policy

We prefer coordinated disclosure. The default window is **90 days from initial acknowledgment** before public disclosure, extendable by mutual agreement if a fix is genuinely in flight.

When a fix lands, we will credit the reporter in the release notes by name or handle of their choosing. If you prefer to remain anonymous, say so in the report and we will honor that.

## Safe harbor

Good-faith security research on Kiln is welcomed. If you follow this policy — private report, no data exfiltration beyond what is needed to demonstrate the issue, no disruption to other users — we will not pursue legal action against you, and we will treat your report as a contribution to the project.
