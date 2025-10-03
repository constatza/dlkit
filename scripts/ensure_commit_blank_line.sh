#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: $0 <start-commit> <end-commit>

Rewrites the commit messages between <start-commit> (exclusive) and <end-commit> (inclusive)
so that any commit message with a body has a blank line separating the title from the body.
The script rewrites history: it creates a backup branch, rebuilds the commit range without
merges, and updates the current branch in place.
USAGE
}

if [ $# -ne 2 ]; then
    usage >&2
    exit 1
fi

start_commit="$1"
end_commit="$2"

if ! git rev-parse --verify "${start_commit}^{commit}" >/dev/null 2>&1; then
    echo "Start commit '${start_commit}' is not valid." >&2
    exit 1
fi

if ! git rev-parse --verify "${end_commit}^{commit}" >/dev/null 2>&1; then
    echo "End commit '${end_commit}' is not valid." >&2
    exit 1
fi

current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" = "HEAD" ]; then
    echo "Please run the script from a branch (not in detached HEAD)." >&2
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "Working tree is not clean. Commit or stash changes first." >&2
    exit 1
fi

start_oid=$(git rev-parse "${start_commit}^{commit}")
end_oid=$(git rev-parse "${end_commit}^{commit}")
head_oid=$(git rev-parse HEAD)

if ! git merge-base --is-ancestor "$start_oid" "$end_oid"; then
    echo "Start commit must be an ancestor of end commit." >&2
    exit 1
fi

if ! git merge-base --is-ancestor "$end_oid" "$head_oid"; then
    echo "End commit must be on the current branch history." >&2
    exit 1
fi

if [ "$(git rev-parse "$current_branch")" != "$head_oid" ]; then
    echo "Current branch HEAD changed during execution; aborting." >&2
    exit 1
fi

merge_count=$(git rev-list --count --merges "${start_oid}..${head_oid}")
if [ "$merge_count" -ne 0 ]; then
    echo "Commit range contains merge commits; cherry-pick rewrite is unsupported." >&2
    exit 1
fi

mapfile -t range_commits < <(git rev-list --reverse "${start_oid}..${end_oid}")
if [ ${#range_commits[@]} -eq 0 ]; then
    echo "No commits to process between ${start_commit} and ${end_commit}." >&2
    exit 1
fi

mapfile -t tail_commits < <(git rev-list --reverse "${end_oid}..${head_oid}")

backup_branch="${current_branch}-message-backup-$(date +%Y%m%d%H%M%S)"
temp_branch="message-rewrite-$(date +%s)"

echo "Creating backup branch '${backup_branch}'."
git branch "$backup_branch" "$current_branch"

echo "Checking out temporary branch '${temp_branch}' at '${start_oid}'."
git checkout -b "$temp_branch" "$start_oid"

for commit in "${range_commits[@]}"; do
    git cherry-pick "$commit"
    current_msg=$(git log --format=%B -n 1 HEAD)
    title=$(printf "%s" "$current_msg" | head -n 1)
    body=$(printf "%s" "$current_msg" | tail -n +2)
    first_body_line=$(printf "%s" "$body" | head -n 1 || true)

    if [ -z "$body" ]; then
        continue
    fi

    trimmed_first=$(printf "%s" "$first_body_line" | tr -d '[:space:]')
    if [ -z "$trimmed_first" ]; then
        continue
    fi

    formatted_msg="${title}

${body}"

    if [ "$formatted_msg" != "$current_msg" ]; then
        printf "%s\n" "$formatted_msg" | git commit --amend --no-verify --cleanup=verbatim --quiet -F -
    fi
done

for commit in "${tail_commits[@]}"; do
    git cherry-pick "$commit"
done

git checkout "$current_branch"
git reset --hard "$temp_branch"

git branch -D "$temp_branch"

echo "History rewritten. Original state saved on '${backup_branch}'."
echo "Review the rewritten commits and delete the backup branch when satisfied."
