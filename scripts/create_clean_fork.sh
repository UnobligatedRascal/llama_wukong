#!/bin/bash
# create_clean_fork.sh - Create a clean llama_wukong fork with proper ancestry
#
# This script creates a new branch with:
# 1. ggml-org/master as base
# 2. llama_lazarus commits cherry-picked on top
# 3. wukong-specific commits cherry-picked on top
#
# WARNING: This is a complex operation that will create conflicts.
# Run only after backing up your current work!
#
# Usage: ./scripts/create_clean_fork.sh
#
# UnobligatedRascal

set -e

REPO_DIR="/home/whistler/llama_wukong"
cd "$REPO_DIR"

echo "=== Creating clean llama_wukong fork ==="
echo ""

# Ensure we have latest from all remotes
echo "Fetching latest from all remotes..."
git fetch ggml-org 2>/dev/null || git fetch https://github.com/ggml-org/llama.cpp.git master:refs/remotes/ggml-org/master
git fetch upstream
git fetch origin

# Create a backup of current state
echo "Creating backup branch..."
git branch -D clean-fork-backup 2>/dev/null || true
git branch clean-fork-backup HEAD

# Start from ggml-org/master
echo "Creating new branch from ggml-org/master..."
git checkout --orphan clean-fork
git reset --hard ggml-org/master

# Cherry-pick llama_lazarus commits (5 commits since merge-base)
echo ""
echo "Cherry-picking llama_lazarus commits..."
LAZARUS_COMMITS=$(git log --oneline upstream/master $(git merge-base ggml-org/master upstream/master)^..upstream/master | awk '{print $1}')
for commit in $LAZARUS_COMMITS; do
    echo "  Cherry-picking: $commit"
    git cherry-pick "$commit" || {
        echo "  CONFLICT in $commit - manual resolution needed"
        git cherry-pick --abort
    }
done

# Wukong-specific commits (identified from analysis)
# These are commits that add wukong-specific features:
# - sm_37/Kepler support
# - TurboQuant
# - TriAttention
# - NUMA replication
# - RoPE LUT
# - Tensor-split fixes for turbo types
# - Documentation specific to wukong
echo ""
echo "Cherry-picking wukong-specific commits..."

# Note: These commits must be cherry-picked in chronological order (oldest first)
# The list below is from main-backup history, ordered oldest to newest
WUKONG_COMMITS=(
    "5af423369"    # Kepler sm_37 support: cuBLAS fixes for Tesla K80/K40
    "be145adb1"    # Initial: project documentation, research notes, and build script
    "931aa32d2"    # feat: add NUMA replication support (P0 phase 1)
    "86463c8ff"    # fix(ggml-cpu-numa-replicate): use correct numa_node_to_cpus bitmask API
    "9a33f2973"    # NUMA replication: implement offset-based weight replication, benchmark results
    "0028a097c"    # docs: mark NUMA replication Task 1 complete
    "74b1eb488"    # docs: add ARCHITECTURE_REFERENCE.md from recovered pitch
    "799d12292"    # feat(sm_37): RoPE sin/cos lookup table for fast transcendental approximation
    "127ef2ee3"    # feat: TurboQuant + TriAttention sm_37 integration (Task 2.1 complete)
    "e8538a0f9"    # feat: TurboQuant backend integration complete (Task 2.2)
    "52c801fb9"    # security: purge credentials from repo history and redact all sensitive values
    "329510125"    # docs: update project vision to reflect NOUGHT hardware optimization focus
    "1c5237a95"    # docs: update project documentation to reflect current integration status
    "a63281cd6"    # feat: complete TriAttention implementation (Task 3)
    "f0a44002a"    # fix(sm_37): use __device__ instead of __constant__ memory for RoPE LUT
    "f02d5acda"    # Document code_mapper tool in llama_wukong.md
    "5970fe28a"    # fix: make FA output contiguous before reshape for tensor-split + turbo3_0 V cache
    "39b90a800"    # docs: document FA + turbo3_0 V cache + tensor-split fix
    "3709ed8ba"    # Fix turbo3_0/turbo4_0 KV cache with tensor-split: meta backend split state bugs
    "372f66602"    # Update PHASE1_TODO.md: document turbo3_0 tensor-split meta backend fix
    "1219e9e4f"    # docs: consolidate all documentation into PROJECT.md; remove redundant files
    "7d7944e3b"    # fix: use KV cache tensor's own block size for split granularity
    "5177c1716"    # docs: add ROTA for quantized KV cache + tensor-split fix
    "220da23a7"    # docs: consolidate TODO, add KV cache audit methodology, clean README
    "70481fa68"    # fix(meta): correct stride scaling for quantized split tensors
    "95df2228b"    # docs: update PROJECT.md with turbo2_0/turbo3_0 meta backend fix status
)

for commit in "${WUKONG_COMMITS[@]}"; do
    echo "  Cherry-picking: $commit"
    git cherry-pick "$commit" || {
        echo "  CONFLICT in $commit - manual resolution needed"
        echo "  Run 'git cherry-pick --continue' or 'git cherry-pick --abort' manually"
        exit 1
    }
done

echo ""
echo "=== Clean fork created on branch 'clean-fork' ==="
echo ""
echo "Next steps:"
echo "1. Build and test: cd build && cmake .. && make -j36 llama-server"
echo "2. Verify it works with turbo KV cache + tensor-split"
echo "3. If satisfied, force-push to origin/main:"
echo "   git push origin clean-fork:main --force"
echo ""
echo "WARNING: Force-pushing will rewrite history and may break any open PRs!"
echo "Current main is preserved as 'clean-fork-backup' branch."
echo ""
echo "UnobligatedRascal - Making old hardware sing."
