---
description: Runs pre-commit checks, handles branch synchronization, commits, and orchestrates remote pushes and Pull Requests.
---

# Advanced Git Orchestration

Usage: /commit

## Step 1: Pre-Commit Checks & Testing

Do not stage any files yet. First, ensure the codebase is pristine:

1. `uv run ruff check --fix .`
2. `uv run ruff format .`
3. `uv run mypy app`
4. `uv run pytest`

If any of these fail, STOP. Output the error and ask the user to fix it before proceeding.

## Step 2: Branch Awareness & Synchronization

Before committing, you must understand the repository state using your terminal tool:

1. Run `git branch --show-current` to identify the active branch.
2. Run `git fetch origin` to update remote tracking data.
3. Run `git status` to check if the current branch is behind its remote counterpart or the main integration branch (`main` or `develop`).

**Conditional Action:** If the current branch is NOT `main` or `develop`, AND the integration branch (`origin/main` or `origin/develop`) is ahead of the current branch, STOP and ask the user: 

> *"The main/develop branch is ahead of your current branch. Would you like me to merge it into this branch to prevent future conflicts?"*

If the user approves, run `git merge origin/main` (or develop). Pause if there are merge conflicts.

## Step 3: Git Staging & Messaging

Once the branch is synced:

1. Run `git diff` and `git status` to understand the local changes.
2. Generate a Conventional Commit message (e.g., `feat:`, `fix:`, `refactor:`) that accurately describes the business logic or architectural changes.
3. Output the proposed commit message as an Artifact for the user to approve.

## Step 4: Execution & Remote Push

Once the user approves the commit message:

1. `git add .`
2. `git commit -m "[Approved Message]"`
3. STOP and ask the user: *"Would you like me to push these changes to the remote repository?"* If the user approves, run `git push -u origin [current_branch]`.

## Step 5: Pull Request Prompt

If the current branch is NOT `main` or `develop` and changes were just pushed to the remote, ask the user: 

> *"Would you like to open a Pull Request for this branch?"*

If the user approves, check if the GitHub CLI is installed by running `gh --version`. 

- If installed, use `gh pr create --web` or generate it directly via CLI. 
- If not installed, output the exact GitHub URL for the user to click to open the PR in their browser.
