#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CURRENT_DRIVER_VERSIONS=("26" "27")
LEGACY_VALIDATOR_COMMIT="9371035bfeb3908ebf47bd107eb53f98c529be3e"
LEGACY_BENCHMARK_ROOT="benchmarks/receipts/rocm/strix-halo"
LEGACY_ORACLE_ROOT="qualification/oracle-results/rocm/strix-halo"

cd "$ROOT"

mapfile -d '' benchmark_receipts < <(
    find benchmarks/receipts -type f -name '*.json' -print0 | sort -z
)
current_benchmark_receipts=()
legacy_benchmark_receipts=()
for receipt in "${benchmark_receipts[@]}"; do
    driver_version="$(
        python3 -c \
            'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("driver_version", ""))' \
            "$receipt"
    )"
    if [[ " ${CURRENT_DRIVER_VERSIONS[*]} " == *" $driver_version "* ]]; then
        current_benchmark_receipts+=("$receipt")
    else
        legacy_benchmark_receipts+=("$receipt")
    fi
done

if ((${#current_benchmark_receipts[@]})); then
    python3 scripts/bench-concurrent-batch.py \
        --validate-receipt "${current_benchmark_receipts[@]}"
fi

mapfile -d '' oracle_results < <(
    find qualification/oracle-results -type f -name '*.json' -print0 | sort -z
)
current_oracle_results=()
legacy_oracle_results=()
for result in "${oracle_results[@]}"; do
    schema="$(
        python3 -c \
            'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("schema", ""))' \
            "$result"
    )"
    case "$schema" in
        kiln.rocm-hf-next-token-oracle.v2 | \
        kiln.rocm-hf-layer-attribution-result.v2 | \
        kiln.rocm-hf-path-attribution-result.v2)
            current_oracle_results+=("$result")
            ;;
        *)
            legacy_oracle_results+=("$result")
            ;;
    esac
done

if ((${#current_oracle_results[@]})); then
    python3 scripts/qualification/check_oracle_results.py \
        "${current_oracle_results[@]}"
fi

if ((
    ${#legacy_benchmark_receipts[@]} == 0
    && ${#legacy_oracle_results[@]} == 0
)); then
    exit 0
fi

if ! git cat-file -e "${LEGACY_VALIDATOR_COMMIT}^{commit}"; then
    echo "error: legacy evidence validator commit is unavailable" >&2
    echo "fetch it with:" >&2
    echo "  git fetch --no-tags --depth=1 origin $LEGACY_VALIDATOR_COMMIT" >&2
    exit 2
fi

for receipt in "${legacy_benchmark_receipts[@]}"; do
    case "$receipt" in
        "$LEGACY_BENCHMARK_ROOT"/*.json) ;;
        *)
            echo "error: unsupported legacy benchmark receipt: $receipt" >&2
            exit 2
            ;;
    esac
done

for result in "${legacy_oracle_results[@]}"; do
    case "$result" in
        "$LEGACY_ORACLE_ROOT"/*.json) ;;
        *)
            echo "error: unsupported legacy oracle result: $result" >&2
            exit 2
            ;;
    esac
done

if ! git diff --quiet "$LEGACY_VALIDATOR_COMMIT" -- "$LEGACY_BENCHMARK_ROOT"; then
    echo "error: retained legacy benchmark receipts differ from their validator checkpoint" >&2
    git diff --name-status "$LEGACY_VALIDATOR_COMMIT" -- "$LEGACY_BENCHMARK_ROOT" >&2
    exit 2
fi

if ! git diff --quiet "$LEGACY_VALIDATOR_COMMIT" -- "$LEGACY_ORACLE_ROOT"; then
    echo "error: retained legacy oracle results differ from their validator checkpoint" >&2
    git diff --name-status "$LEGACY_VALIDATOR_COMMIT" -- "$LEGACY_ORACLE_ROOT" >&2
    exit 2
fi

temporary_root="$(mktemp -d)"
legacy_tree="$temporary_root/validator"
cleanup() {
    git worktree remove --force "$legacy_tree" >/dev/null 2>&1 || true
    rmdir "$temporary_root" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

git worktree add --quiet --detach "$legacy_tree" "$LEGACY_VALIDATOR_COMMIT"
mapfile -d '' checkpoint_benchmark_receipts < <(
    find "$legacy_tree/$LEGACY_BENCHMARK_ROOT" \
        -type f -name '*.json' -print0 | sort -z
)
if ((
    ${#checkpoint_benchmark_receipts[@]}
    != ${#legacy_benchmark_receipts[@]}
)); then
    echo "error: legacy benchmark receipt inventory disagrees with validator checkpoint" >&2
    exit 2
fi

if ((${#checkpoint_benchmark_receipts[@]})); then
    python3 "$legacy_tree/scripts/bench-concurrent-batch.py" \
        --validate-receipt "${checkpoint_benchmark_receipts[@]}"
fi

mapfile -d '' checkpoint_oracle_results < <(
    find "$legacy_tree/$LEGACY_ORACLE_ROOT" \
        -type f -name '*.json' -print0 | sort -z
)
if ((${#checkpoint_oracle_results[@]} != ${#legacy_oracle_results[@]})); then
    echo "error: legacy oracle result inventory disagrees with validator checkpoint" >&2
    exit 2
fi

if ((${#checkpoint_oracle_results[@]})); then
    python3 "$legacy_tree/scripts/qualification/check_oracle_results.py" \
        "${checkpoint_oracle_results[@]}"
fi
