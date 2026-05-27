#!/usr/bin/env python3
"""Fetch and convert the RLM research corpus.

The generated corpus intentionally distinguishes between sources that can be
mirrored verbatim and sources that should only be indexed with metadata.
"""

from __future__ import annotations

import html
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md


ROOT = Path(__file__).resolve().parents[1]
ARXIV_DIR = ROOT / "sources" / "arxiv"
GITHUB_DIR = ROOT / "sources" / "github"
WEB_DIR = ROOT / "sources" / "web"


CC_BY_4 = "http://creativecommons.org/licenses/by/4.0/"
CC0_1 = "http://creativecommons.org/publicdomain/zero/1.0/"
MIRRORABLE_ARXIV_LICENSES = {CC_BY_4, CC0_1}
CA_BUNDLE = Path("/etc/ssl/certs/ca-certificates.crt")


@dataclass(frozen=True)
class ArxivSource:
    arxiv_id: str
    slug: str
    category: str
    relevance: str
    related: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GitHubReadme:
    slug: str
    raw_url: str
    source_url: str
    license_name: str
    license_url: str
    relevance: str


@dataclass(frozen=True)
class WebSource:
    slug: str
    title: str
    url: str
    date: str
    relevance: str
    reuse_note: str


ARXIV_SOURCES: list[ArxivSource] = [
    ArxivSource(
        "2512.24601",
        "2512.24601-recursive-language-models",
        "core",
        "Canonical RLM paper: prompt-as-environment, Python REPL interaction, recursive subcalls, long-context inference-time scaling, and RLM-Qwen3-8B post-training.",
        [
            "https://github.com/alexzhang13/rlm",
            "https://github.com/alexzhang13/rlm-minimal",
            "https://alexzhang13.github.io/blog/2025/rlm/",
        ],
    ),
    ArxivSource(
        "2602.16520",
        "2602.16520-rlm-jailbreak-detection",
        "direct-application",
        "RLM-JB applies bounded recursive analysis, chunk screening, normalization, and evidence aggregation to jailbreak detection for tool-augmented agents.",
    ),
    ArxivSource(
        "2603.02615",
        "2603.02615-reproducing-rlms",
        "direct-reproduction",
        "Reproduction and recursion-depth ablation: depth-1 can help complex OOLONG-style reasoning, while depth-2 can overthink, hurt accuracy, and inflate latency/cost.",
        ["https://github.com/drbillwang/rlm-reproduction"],
    ),
    ArxivSource(
        "2603.15653",
        "2603.15653-self-reflective-program-search",
        "direct-extension",
        "SRLM critiques RLM program selection and uses self-consistency, reasoning length, and verbalized confidence to select context-interaction programs.",
    ),
    ArxivSource(
        "2603.20105",
        "2603.20105-lambda-rlm",
        "direct-extension",
        "lambda-RLM replaces free-form REPL code with typed lambda-calculus combinators to obtain termination, cost bounds, and predictable recursive control.",
        ["https://github.com/lambda-calculus-LLM/lambda-RLM"],
    ),
    ArxivSource(
        "2605.05632",
        "2605.05632-rag-poisoning-architectures",
        "direct-evaluation",
        "Evaluates Recursive Language Models as a RAG architecture under knowledge-base poisoning and reports substantially lower attack success than vanilla RAG in that setup.",
        ["https://zenodo.org/records/19582218"],
    ),
    ArxivSource(
        "2605.09968",
        "2605.09968-opmech-adaptive-learning",
        "direct-theory",
        "Introduces order-gap control signals and develops a Recursive Language Model application as a replacement for heuristic stopping rules and fixed recursion budgets.",
    ),
    ArxivSource(
        "2605.04050",
        "2605.04050-lossless-context-management",
        "direct-adjacent",
        "LCM explicitly positions itself as a deterministic extension of RLM-style recursive context management, with recursive compression and task partitioning.",
        ["https://www.losslesscontext.ai/"],
    ),
    ArxivSource(
        "2603.20432",
        "2603.20432-coding-agents-long-context",
        "adjacent-coding-agents",
        "Shows that coding agents can externalize long-context processing through filesystems and native tools, a close comparison point for RLM's prompt-as-environment interface.",
    ),
    ArxivSource(
        "2405.17402",
        "2405.17402-thread-recursive-spawning",
        "adjacent-recursive-agents",
        "THREAD is an early recursive-spawning framework cited by the RLM paper; it dynamically creates child threads to decompose work and return only useful tokens to parents.",
    ),
    ArxivSource(
        "2510.11967",
        "2510.11967-context-folding",
        "adjacent-context-folding",
        "Context-Folding for long-horizon agents: branch/return sub-trajectories and FoldGRPO for learnable context management.",
    ),
    ArxivSource(
        "2510.24699",
        "2510.24699-agentfold",
        "adjacent-context-folding",
        "AgentFold treats context as a proactively managed workspace for long-horizon web agents.",
    ),
    ArxivSource(
        "2601.18285",
        "2601.18285-u-fold",
        "adjacent-context-folding",
        "U-Fold performs dynamic intent-aware context folding for user-centric multi-turn agents.",
    ),
    ArxivSource(
        "2510.00615",
        "2510.00615-acon",
        "adjacent-context-compression",
        "ACON optimizes context-compression guidelines for long-horizon LLM agents and is a strong foil for RLM's avoid-lossy-compaction framing.",
    ),
    ArxivSource(
        "2509.13313",
        "2509.13313-resum",
        "adjacent-context-compression",
        "ReSum is a context-summarization baseline and training recipe for long-horizon web agents; it frames the main compaction alternative to RLM-style source-preserving recursion.",
    ),
    ArxivSource(
        "2512.16970",
        "2512.16970-paace",
        "adjacent-context-engineering",
        "PAACE uses plan-aware context engineering and distilled compressors for agent workflows.",
    ),
    ArxivSource(
        "2601.07190",
        "2601.07190-active-context-compression",
        "adjacent-context-compression",
        "Active Context Compression studies autonomous agent-controlled pruning and compression in long-horizon SWE-style tasks.",
    ),
    ArxivSource(
        "2510.04618",
        "2510.04618-agentic-context-engineering",
        "adjacent-context-engineering",
        "Agentic Context Engineering frames context as an evolving playbook updated through generation, reflection, and curation.",
    ),
    ArxivSource(
        "2601.21557",
        "2601.21557-meta-context-engineering",
        "adjacent-context-engineering",
        "Meta Context Engineering studies meta-level optimization of context artifacts and context-engineering skills.",
    ),
    ArxivSource(
        "2601.01885",
        "2601.01885-agentic-memory",
        "adjacent-memory",
        "Agentic Memory exposes memory operations as tool actions and trains unified short-term/long-term memory management with GRPO.",
    ),
    ArxivSource(
        "2506.15841",
        "2506.15841-mem1",
        "adjacent-memory",
        "MEM1 studies a constant-memory long-horizon agent state and is useful for contrasting externalized RLM control with persistent memory state.",
    ),
    ArxivSource(
        "2510.18866",
        "2510.18866-lightmem",
        "adjacent-memory",
        "LightMem is a staged sensory/short-term/long-term memory system with large token and API reductions.",
    ),
    ArxivSource(
        "2504.19413",
        "2504.19413-mem0",
        "adjacent-memory",
        "Mem0 is a production-oriented memory extraction, consolidation, and retrieval baseline for agents.",
    ),
    ArxivSource(
        "2402.09727",
        "2402.09727-gist-memory-reading-agent",
        "adjacent-memory",
        "Human-inspired reading agent with gist memory and lookup over very long contexts; an earlier agentic long-context reading baseline.",
    ),
    ArxivSource(
        "2404.07143",
        "2404.07143-infini-attention",
        "adjacent-architecture",
        "Infini-attention is an architectural compressive-memory approach for efficient infinite-context transformers.",
    ),
    ArxivSource(
        "2405.06067",
        "2405.06067-hierarchical-memory-transformer",
        "adjacent-architecture",
        "HMT uses hierarchical segment memory and recall for efficient long-context language processing.",
    ),
    ArxivSource(
        "2501.00663",
        "2501.00663-titans",
        "adjacent-architecture",
        "Titans learns neural long-term memory at test time, an architecture-side cousin to RLM's external memory/control loop.",
    ),
    ArxivSource(
        "2505.23735",
        "2505.23735-atlas",
        "adjacent-architecture",
        "ATLAS learns to optimally memorize context at test time and reports very-long-context memory gains.",
    ),
    ArxivSource(
        "2502.06049",
        "2502.06049-large-memory-models",
        "adjacent-architecture",
        "Large Memory Models add explicit auxiliary memory to decoder-only Transformers, an architecture-side alternative to recursive scaffolds.",
    ),
    ArxivSource(
        "2404.06654",
        "2404.06654-ruler",
        "evaluation",
        "RULER grounds the long-context degradation and needle/retrieval evaluation setting used by many RLM discussions.",
    ),
    ArxivSource(
        "2410.10813",
        "2410.10813-longmemeval",
        "evaluation",
        "LongMemEval is a long-term interactive memory benchmark; useful for comparing RLM source-addressability with indexed/retrieved chat memory systems.",
        ["https://github.com/xiaowu0162/LongMemEval"],
    ),
    ArxivSource(
        "2511.02817",
        "2511.02817-oolong",
        "evaluation",
        "OOLONG evaluates long-context reasoning and aggregation; it is one of the main benchmark families in RLM and reproductions.",
    ),
    ArxivSource(
        "2604.14140",
        "2604.14140-longcot",
        "evaluation",
        "LongCoT benchmarks long-horizon chain-of-thought reasoning over problems requiring tens to hundreds of thousands of reasoning tokens.",
    ),
]


GITHUB_READMES: list[GitHubReadme] = [
    GitHubReadme(
        "hampton-io-rlm-readme",
        "https://raw.githubusercontent.com/hampton-io/RLM/main/README.md",
        "https://github.com/hampton-io/RLM",
        "MIT",
        "https://raw.githubusercontent.com/hampton-io/RLM/main/LICENSE",
        "Node.js/TypeScript RLM implementation with JavaScript REPL, sandboxing, streaming, tracing, CLI, and multi-provider support.",
    ),
    GitHubReadme(
        "hmbown-aleph-readme",
        "https://raw.githubusercontent.com/Hmbown/aleph/main/README.md",
        "https://github.com/Hmbown/aleph",
        "MIT",
        "https://raw.githubusercontent.com/Hmbown/aleph/main/LICENSE",
        "MCP server and skill for RLM-style external working state, search indexes, code execution, evidence, and recursion.",
    ),
    GitHubReadme(
        "q00-rlm-forge-readme",
        "https://raw.githubusercontent.com/Q00/rlm-forge/main/README.md",
        "https://github.com/Q00/rlm-forge",
        "MIT",
        "https://raw.githubusercontent.com/Q00/rlm-forge/main/LICENSE",
        "Small runtime-lifted RLM experiment with bounded inner calls, evidence manifests, and deterministic TraceGuard checks.",
    ),
    GitHubReadme(
        "alexzhang13-rlm-readme",
        "https://raw.githubusercontent.com/alexzhang13/rlm/main/README.md",
        "https://github.com/alexzhang13/rlm",
        "MIT",
        "https://raw.githubusercontent.com/alexzhang13/rlm/main/LICENSE",
        "Official RLM implementation and documentation entry point.",
    ),
    GitHubReadme(
        "alexzhang13-rlm-minimal-readme",
        "https://raw.githubusercontent.com/alexzhang13/rlm-minimal/main/README.md",
        "https://github.com/alexzhang13/rlm-minimal",
        "MIT",
        "https://raw.githubusercontent.com/alexzhang13/rlm-minimal/main/LICENSE",
        "Minimal implementation that exposes the core REPL plus recursive-call idea without the full framework.",
    ),
    GitHubReadme(
        "lambda-calculus-llm-lambda-rlm-readme",
        "https://raw.githubusercontent.com/lambda-calculus-LLM/lambda-RLM/main/README.md",
        "https://github.com/lambda-calculus-LLM/lambda-RLM",
        "MIT",
        "https://raw.githubusercontent.com/lambda-calculus-LLM/lambda-RLM/main/LICENSE",
        "Typed functional lambda-RLM implementation and benchmark harness.",
    ),
    GitHubReadme(
        "primeintellect-verifiers-readme",
        "https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/README.md",
        "https://github.com/PrimeIntellect-ai/verifiers",
        "MIT",
        "https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/LICENSE",
        "Prime Intellect environment framework that includes RLMEnv, composable harnesses, and RLM SWE examples.",
    ),
    GitHubReadme(
        "primeintellect-dspy-rlm-env-readme",
        "https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/environments/dspy_rlm/README.md",
        "https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/dspy_rlm",
        "MIT",
        "https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/LICENSE",
        "DSPy RLM environment example for GSM8K math problems.",
    ),
    GitHubReadme(
        "primeintellect-hello-rlm-v1-env-readme",
        "https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/environments/hello_rlm_v1/README.md",
        "https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/hello_rlm_v1",
        "MIT",
        "https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/LICENSE",
        "Compact v1 RLM-style CLI harness migration example.",
    ),
    GitHubReadme(
        "primeintellect-rlm-swe-v1-env-readme",
        "https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/environments/rlm_swe_v1/README.md",
        "https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/rlm_swe_v1",
        "MIT",
        "https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/LICENSE",
        "RLM coding environment using the R2E-Gym SWE taskset and packaged RLM harness.",
    ),
]


WEB_SOURCES: list[WebSource] = [
    WebSource(
        "raw-works-rlms-new-reasoning-models",
        "RLMs are the New Reasoning Models",
        "https://raw.works/rlms-are-the-new-reasoning-models/",
        "2026-02",
        "Commentary connecting RLMs, coding agents as long-context processors, benchmark contamination concerns, and RAW Works' recursive coding-agent experiments.",
        "No explicit reuse license found during corpus generation; indexed with metadata only.",
    ),
    WebSource(
        "raw-works-rlms-memory-systems",
        "Recursive Language Models as Memory Systems",
        "https://raw.works/recursive-language-models-as-memory-systems/",
        "2026",
        "Frames RLMs as memory systems rather than just long-context workarounds; useful conceptual bridge to LongMemEval and memory-agent work.",
        "No explicit reuse license found during corpus generation; indexed with metadata only.",
    ),
    WebSource(
        "daytona-rlm-guides",
        "Recursive Language Model Guides",
        "https://www.daytona.io/docs/en/guides/rlm/",
        "2026",
        "Daytona implementation guides for recursive RLM agents and DSPy RLMs with isolated Daytona-backed interpreters.",
        "No explicit reuse license found during corpus generation; indexed with metadata only.",
    ),
    WebSource(
        "daytona-deep-rlm-guide",
        "Build deep Recursive Language Models",
        "https://www.daytona.io/docs/ja/recursive-language-models/",
        "2026",
        "Detailed sandboxed coding-agent RLM guide that extends single-level recursion to sub-agents in isolated Daytona sandboxes.",
        "No explicit reuse license found during corpus generation; indexed with metadata only.",
    ),
    WebSource(
        "dbreunig-potential-of-rlms",
        "The Potential of RLMs",
        "https://www.dbreunig.com/",
        "2026",
        "Secondary analysis from Drew Breunig's site; included as a pointer because an RLM post surfaced during source discovery but was not mirrored.",
        "No explicit reuse license found during corpus generation; indexed with metadata only.",
    ),
    WebSource(
        "tenobrus-claude-rlm",
        "RLM: Recursive Language Models for Claude Code",
        "https://github.com/Tenobrus/claude-rlm",
        "2026",
        "Claude Code skill design for filesystem and bash-based recursive delegation; metadata only because no license file was found in the quick source check.",
        "No explicit reuse license found during corpus generation; indexed with metadata only.",
    ),
    WebSource(
        "alex-zhang-rlm-blog",
        "Recursive Language Models",
        "https://alexzhang13.github.io/blog/2025/rlm/",
        "2025-10",
        "Original explanatory blog post and conceptual introduction for the RLM idea before the expanded arXiv preprint.",
        "No explicit reuse license found during corpus generation; indexed with metadata only.",
    ),
    WebSource(
        "prime-intellect-rlm-blog",
        "Recursive Language Models: the paradigm of 2026",
        "https://www.primeintellect.ai/blog/rlm",
        "2026-01-01",
        "Prime Intellect's applied framing of RLMs, RLMEnv, context folding, evaluation ablations, and RL training direction.",
        "No explicit reuse license found during corpus generation; indexed with metadata only.",
    ),
    WebSource(
        "alphaxiv-reinforcing-rlms",
        "Reinforcing Recursive Language Models",
        "https://www.alphaxiv.org/blog/reinforcement-learning-for-rlms",
        "2026-05-13",
        "Blog-style RL follow-up describing training smaller models to behave as native RLMs over scientific documents.",
        "No explicit reuse license found during corpus generation; indexed with metadata only.",
    ),
]


def fetch(url: str) -> str:
    verify: str | bool = str(CA_BUNDLE) if CA_BUNDLE.exists() else True
    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "kiln-rlm-docs/1.0"},
        verify=verify,
    )
    response.raise_for_status()
    return response.text


def normalize_markdown(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip() + "\n"


def frontmatter(items: dict[str, str | Iterable[str]]) -> str:
    lines = ["---"]
    for key, value in items.items():
        if isinstance(value, str):
            safe = value.replace('"', '\\"')
            lines.append(f'{key}: "{safe}"')
        else:
            lines.append(f"{key}:")
            for entry in value:
                safe = str(entry).replace('"', '\\"')
                lines.append(f'  - "{safe}"')
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def parse_arxiv_abs(source: ArxivSource) -> dict[str, object]:
    soup = BeautifulSoup(fetch(f"https://arxiv.org/abs/{source.arxiv_id}"), "html.parser")

    def meta(name: str) -> list[str]:
        return [tag.get("content", "").strip() for tag in soup.find_all("meta", attrs={"name": name})]

    def one(name: str) -> str:
        values = meta(name)
        return values[0] if values else ""

    license_a = soup.select_one(".abs-license a")
    license_url = license_a.get("href", "").strip() if license_a else ""
    canonical = soup.find("link", rel="canonical")
    og_url = soup.find("meta", property="og:url")
    page_url = ""
    if og_url and og_url.get("content"):
        page_url = og_url["content"]
    elif canonical and canonical.get("href"):
        page_url = canonical["href"]
    else:
        page_url = f"https://arxiv.org/abs/{source.arxiv_id}"
    return {
        "id": source.arxiv_id,
        "title": one("citation_title"),
        "authors": meta("citation_author"),
        "date": one("citation_date"),
        "online_date": one("citation_online_date"),
        "pdf_url": one("citation_pdf_url"),
        "page_url": page_url,
        "license_url": license_url,
        "mirrorable": license_url in MIRRORABLE_ARXIV_LICENSES,
        "category": source.category,
        "relevance": source.relevance,
        "related": source.related,
    }


def convert_arxiv_fulltext(source: ArxivSource, metadata: dict[str, object]) -> str | None:
    try:
        soup = BeautifulSoup(fetch(f"https://arxiv.org/html/{source.arxiv_id}"), "html.parser")
    except requests.HTTPError:
        return None
    article = soup.select_one("article.ltx_document")
    if article is None:
        return None

    for node in article.select("svg, script, style"):
        node.decompose()

    body = md(str(article), heading_style="ATX", bullets="-")
    body = normalize_markdown(body)
    related = "\n".join(f"- {url}" for url in metadata["related"]) or "- None recorded"
    authors = ", ".join(metadata["authors"])
    header = frontmatter(
        {
            "source_type": "arxiv-fulltext",
            "arxiv_id": str(metadata["id"]),
            "title": str(metadata["title"]),
            "authors": [str(a) for a in metadata["authors"]],
            "date": str(metadata["date"]),
            "online_date": str(metadata["online_date"]),
            "license_url": str(metadata["license_url"]),
            "source_url": str(metadata["page_url"]),
            "pdf_url": str(metadata["pdf_url"]),
            "category": str(metadata["category"]),
            "generated_utc": generated_at(),
        }
    )
    return (
        header
        + f"# {metadata['title']}\n\n"
        + "## Corpus Note\n\n"
        + "This file is a mechanical markdown conversion of arXiv HTML for a paper with a mirrorable reuse license. "
        + "Use the source links below as canonical when citing or checking equations, tables, or figures.\n\n"
        + f"- Source: {metadata['page_url']}\n"
        + f"- PDF: {metadata['pdf_url']}\n"
        + f"- License: {metadata['license_url']}\n"
        + f"- Authors: {authors}\n"
        + f"- RLM relevance: {metadata['relevance']}\n"
        + f"- Related links:\n{related}\n\n"
        + "## Converted Full Text\n\n"
        + body
    )


def write_arxiv_metadata(
    source: ArxivSource,
    metadata: dict[str, object],
    *,
    fulltext_unavailable: bool = False,
) -> str:
    authors = ", ".join(metadata["authors"])
    related = "\n".join(f"- {url}" for url in metadata["related"]) or "- None recorded"
    license_url = str(metadata["license_url"] or "unknown")
    if fulltext_unavailable:
        mirror_status = (
            "Metadata only. The arXiv record advertises a mirrorable reuse license, but arXiv HTML was unavailable "
            "during generation, so no mechanical full-text markdown was mirrored."
        )
    elif metadata["mirrorable"]:
        mirror_status = "Full markdown mirrored in this corpus because the arXiv record has a mirrorable reuse license."
    else:
        mirror_status = "Metadata only. The arXiv record did not advertise a mirrorable reuse license, so the full paper is not mirrored verbatim here."
    return normalize_markdown(
        frontmatter(
            {
                "source_type": "arxiv-metadata",
                "arxiv_id": str(metadata["id"]),
                "title": str(metadata["title"]),
                "authors": [str(a) for a in metadata["authors"]],
                "date": str(metadata["date"]),
                "online_date": str(metadata["online_date"]),
                "license_url": license_url,
                "source_url": str(metadata["page_url"]),
                "pdf_url": str(metadata["pdf_url"]),
                "category": str(metadata["category"]),
                "generated_utc": generated_at(),
            }
        )
        + f"# {metadata['title']}\n\n"
        + "## Metadata\n\n"
        + f"- arXiv: {metadata['id']}\n"
        + f"- Authors: {authors}\n"
        + f"- Date: {metadata['date']}\n"
        + f"- Latest online date: {metadata['online_date']}\n"
        + f"- Source: {metadata['page_url']}\n"
        + f"- PDF: {metadata['pdf_url']}\n"
        + f"- License URL: {license_url}\n"
        + f"- Corpus status: {mirror_status}\n\n"
        + "## RLM Relevance\n\n"
        + textwrap.fill(str(metadata["relevance"]), width=100)
        + "\n\n"
        + "## Related Links\n\n"
        + related
        + "\n"
    )


def generated_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalize_markdown(content), encoding="utf-8")


def write_github_readme(source: GitHubReadme) -> dict[str, str]:
    readme = fetch(source.raw_url)
    try:
        license_text = fetch(source.license_url)
    except requests.HTTPError:
        license_text = ""
    content = (
        frontmatter(
            {
                "source_type": "github-readme",
                "title": source.slug,
                "source_url": source.source_url,
                "raw_url": source.raw_url,
                "license": source.license_name,
                "license_url": source.license_url,
                "generated_utc": generated_at(),
            }
        )
        + f"# {source.slug}\n\n"
        + "## Corpus Note\n\n"
        + f"- Source repository/page: {source.source_url}\n"
        + f"- Raw README: {source.raw_url}\n"
        + f"- License: {source.license_name} ({source.license_url})\n"
        + f"- RLM relevance: {source.relevance}\n\n"
        + "## Verbatim README\n\n"
        + readme
    )
    if license_text:
        content += "\n\n## Verbatim License\n\n```text\n" + license_text.strip() + "\n```\n"
    write_text(GITHUB_DIR / f"{source.slug}.md", content)
    return {
        "kind": "github",
        "title": source.slug,
        "path": f"sources/github/{source.slug}.md",
        "url": source.source_url,
        "status": f"verbatim README ({source.license_name})",
        "category": "implementation",
    }


def write_web_source(source: WebSource) -> dict[str, str]:
    content = (
        frontmatter(
            {
                "source_type": "web-metadata",
                "title": source.title,
                "source_url": source.url,
                "date": source.date,
                "generated_utc": generated_at(),
            }
        )
        + f"# {source.title}\n\n"
        + "## Metadata\n\n"
        + f"- Source: {source.url}\n"
        + f"- Date: {source.date}\n"
        + f"- Corpus status: metadata only. {source.reuse_note}\n\n"
        + "## RLM Relevance\n\n"
        + textwrap.fill(source.relevance, width=100)
        + "\n"
    )
    write_text(WEB_DIR / f"{source.slug}.md", content)
    return {
        "kind": "web",
        "title": source.title,
        "path": f"sources/web/{source.slug}.md",
        "url": source.url,
        "status": "metadata only",
        "category": "blog",
    }


def write_manifest(entries: list[dict[str, str]]) -> None:
    rows = [
        "| Category | Source | Status | Local file | URL |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in sorted(entries, key=lambda e: (e["category"], e["title"])):
        rows.append(
            f"| {entry['category']} | {entry['title']} | {entry['status']} | "
            f"[{entry['path']}]({entry['path']}) | {entry['url']} |"
        )
    content = (
        "# RLM Source Manifest\n\n"
        f"Generated: {generated_at()}\n\n"
        "This manifest records which sources were mirrored verbatim and which were indexed as metadata only. "
        "Full-text mirroring is limited to sources with a clear reuse license, primarily CC BY 4.0/CC0 arXiv HTML and MIT-licensed GitHub READMEs.\n\n"
        + "\n".join(rows)
        + "\n"
    )
    write_text(ROOT / "SOURCE_MANIFEST.md", content)


def main() -> None:
    entries: list[dict[str, str]] = []
    for directory in (ARXIV_DIR, GITHUB_DIR, WEB_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    for source in ARXIV_SOURCES:
        metadata = parse_arxiv_abs(source)
        status = "metadata only"
        output_path = ARXIV_DIR / f"{source.slug}.md"
        if metadata["mirrorable"]:
            converted = convert_arxiv_fulltext(source, metadata)
            if converted:
                write_text(output_path, converted)
                status = "verbatim arXiv HTML conversion (mirrorable license)"
            else:
                write_text(output_path, write_arxiv_metadata(source, metadata, fulltext_unavailable=True))
                status = "metadata only; arXiv HTML unavailable"
        else:
            write_text(output_path, write_arxiv_metadata(source, metadata))

        entries.append(
            {
                "kind": "arxiv",
                "title": str(metadata["title"]),
                "path": f"sources/arxiv/{source.slug}.md",
                "url": f"https://arxiv.org/abs/{source.arxiv_id}",
                "status": status,
                "category": source.category,
            }
        )

    for source in GITHUB_READMES:
        entries.append(write_github_readme(source))

    for source in WEB_SOURCES:
        entries.append(write_web_source(source))

    write_manifest(entries)


if __name__ == "__main__":
    main()
