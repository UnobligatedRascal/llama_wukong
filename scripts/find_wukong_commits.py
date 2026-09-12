#!/usr/bin/env python3
"""
Identify wukong-specific commits that should be cherry-picked for clean fork.

Wukong-specific commits are those that:
1. Don't reference a ggml-org PR number (e.g., #27991)
2. Are related to wukong-specific features (sm_37, turbo, triattention, NUMA, etc.)
"""

import subprocess
import re

def run_git(args):
    result = subprocess.run(['git'] + args, capture_output=True, text=True, cwd='/home/whistler/llama_wukong')
    return result.stdout.strip()

# Get all commits from main-backup
commits = run_git(['log', '--oneline', 'main-backup']).split('\n')

# Wukong-specific keywords
wukong_keywords = [
    'sm_37', 'turbo', 'triattention', 'NUMA', 'kepler', 'k80',
    'wukong', 'lazarus', 'rope.lut', 'async.pipeline',
    'ggml-cpu-numa-replicate', 'turbo-quant', 'triattention-score',
    'rope-lut', 'nccl-stagger', 'numa-gpu-bind',
]

# PR number pattern (ggml-org PRs have numbers like #27991)
pr_pattern = re.compile(r'#\d{4,5}')

wukong_commits = []
upstream_commits = []

for line in commits:
    if not line.strip():
        continue
    
    sha, msg = line.split(' ', 1)
    
    # Check if it's a ggml-org PR commit
    has_pr = bool(pr_pattern.search(msg))
    
    # Check if it's wukong-specific
    is_wukong = any(kw in msg.lower() for kw in wukong_keywords)
    
    # Also check for wukong-specific file changes
    files = run_git(['diff-tree', '--no-commit-id', '--name-only', '-r', sha]).split('\n')
    wukong_files = [
        'ggml-cpu-numa-replicate.c', 'turbo-quant.cuh', 'turbo-innerq.cuh', 'turbo-wht.cuh',
        'triattention-score.cu', 'rope-lut.cuh', 'nccl-stagger.cuh', 'numa-gpu-bind.cuh',
        'PROJECT.md', 'llama_wukong.md', 'PHASE1_TODO.md', 'ARCHITECTURE_REFERENCE.md',
        'code_mapper.py', 'ASYNC_PIPELINE_ATTEMPTED_CHANGES.md',
        'test_cublas_kepler.cu', 'test_cublas_kepler2.cu', 'test_cublas_kepler3.cu',
    ]
    has_wukong_files = any(f in files for f in wukong_files)
    
    if has_wukong_files or (is_wukong and not has_pr):
        wukong_commits.append((sha, msg))
    elif has_pr:
        upstream_commits.append((sha, msg))
    else:
        # Ambiguous - check if it's a docs/security commit specific to wukong
        if 'security' in msg.lower() and ('credential' in msg.lower() or 'redact' in msg.lower()):
            wukong_commits.append((sha, msg))
        else:
            upstream_commits.append((sha, msg))

print(f"Wukong-specific commits: {len(wukong_commits)}")
print("=" * 80)
for sha, msg in wukong_commits:
    print(f"{sha} {msg}")

print(f"\n\nUpstream/ggml-org commits: {len(upstream_commits)}")
print("(First 20 shown)")
for sha, msg in upstream_commits[:20]:
    print(f"{sha} {msg}")
