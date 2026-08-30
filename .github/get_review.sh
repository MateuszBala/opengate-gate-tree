#!/usr/bin/env bash

set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: required tool 'gh' is not installed." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: required tool 'jq' is not installed." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Error: you are not signed in to gh. Run: gh auth login" >&2
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
REPO_FULL="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
OWNER="${REPO_FULL%/*}"
REPO="${REPO_FULL#*/}"

OUTPUT_DIR=".tmp/code-reviews"
mkdir -p "$OUTPUT_DIR"

BRANCH_SAFE="$(echo "$BRANCH" | sed -E 's/[^A-Za-z0-9._-]+/-/g')"
if [[ -z "$BRANCH_SAFE" ]]; then
  BRANCH_SAFE="unknown-branch"
fi

FETCH_DATE="$(date '+%Y%m%d-%H%M%S')"
FETCH_DAY="${FETCH_DATE%%-*}"
EXISTING_COUNT="$({
  find "$OUTPUT_DIR" -maxdepth 1 -type f \
    -name "pr-code-review-${BRANCH_SAFE}-${FETCH_DAY}-*.md"
} | wc -l | tr -d '[:space:]')"
ORDER_NUMBER="$(printf '%02d' "$((EXISTING_COUNT + 1))")"
OUTPUT_FILE="${OUTPUT_DIR}/pr-code-review-${BRANCH_SAFE}-${FETCH_DATE}-${ORDER_NUMBER}.md"

PRS_JSON="$(gh pr list --state open --head "$BRANCH" --json number,title,url,headRefName,author 2>/dev/null || true)"

if [[ -z "$PRS_JSON" || "$PRS_JSON" == "[]" ]]; then
  {
    echo "# Review tasks"
    echo
    echo "No open PR found for branch: $BRANCH."
  } >"$OUTPUT_FILE"
  echo "Saved: $OUTPUT_FILE"
  exit 0
fi

PR_NUMBERS="$(echo "$PRS_JSON" | jq -r '.[].number')"
FIRST_PR_NUMBER="$(echo "$PR_NUMBERS" | head -n1)"

GRAPHQL_QUERY='query($owner:String!, $repo:String!, $number:Int!) {
  repository(owner:$owner, name:$repo) {
    pullRequest(number:$number) {
      number
      title
      url
      headRefName
      reviewThreads(first: 100) {
        nodes {
          isResolved
          isOutdated
          path
          line
          startLine
          comments(first: 100) {
            nodes {
              author { login }
              body
              url
              createdAt
            }
          }
        }
      }
    }
  }
}'

PR_JSON="$(gh api graphql -f query="$GRAPHQL_QUERY" -F owner="$OWNER" -F repo="$REPO" -F number="$FIRST_PR_NUMBER")"

UNRESOLVED_COUNT="$(echo "$PR_JSON" | jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)] | length')"

{
  echo "# Review tasks"
  echo
  echo "Repository: \`$REPO_FULL\`  "
  echo "Branch: \`$BRANCH\`  "
  echo "PR: [#${FIRST_PR_NUMBER}]($(echo "$PR_JSON" | jq -r '.data.repository.pullRequest.url')) - $(echo "$PR_JSON" | jq -r '.data.repository.pullRequest.title')"
  echo
  echo "## Unresolved threads"
  echo

  if [[ "$UNRESOLVED_COUNT" -eq 0 ]]; then
    echo "No unresolved threads ✅"
  else
    echo "$PR_JSON" | jq -r '
      .data.repository.pullRequest.reviewThreads.nodes
      | map(select(.isResolved == false))
      | to_entries[]
      | .value as $t
      | $t.comments.nodes[0] as $c
      | "- [ ] "
        + "[" + ($t.path // "(brak pliku)")
        + (if ($t.line // $t.startLine) then ":L" + (($t.line // $t.startLine)|tostring) else "" end)
        + "](" + ($c.url // "") + ")"
        + " — @" + ($c.author.login // "unknown")
        + (if $t.isOutdated then " *(outdated)*" else "" end)
        + "\n  > " + (($c.body // "") | gsub("\n"; " ") | gsub("\r"; " ") | .[0:50000])
    '
  fi
} >"$OUTPUT_FILE"

echo "Saved: $OUTPUT_FILE"