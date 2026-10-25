---
name: github-workflow
description: Automate GitHub operations including PRs, issues, and CI/CD
author: jarvis
version: 1.0.0
tags: [github, git, ci-cd, automation]
metadata:
  jarvis:
    requires:
      bins: [gh, git]
      env: [GITHUB_TOKEN]
      config: []
    primaryEnv: GITHUB_TOKEN
    user-invocable: true
    emoji: octocat
---

# GitHub Workflow Skill

Automate GitHub operations using the `gh` CLI and git commands.

## Supported Operations

### Repository Management
- Create/clone repositories
- Fork repositories
- Manage branches

### Pull Requests
- Create PRs with auto-generated descriptions
- Review PRs (approve, request changes, comment)
- Merge PRs with appropriate strategies
- Handle merge conflicts

### Issues
- Create issues with templates
- Label and triage issues
- Close issues with resolution comments

### CI/CD
- Trigger workflows
- Monitor workflow status
- Download artifacts

## Workflow Patterns

### Create Feature PR
```bash
# Create branch
gh repo clone owner/repo && cd repo
git checkout -b feature/my-feature

# Make changes...
git add . && git commit -m "Add feature"

# Push and create PR
git push -u origin feature/my-feature
gh pr create --title "Add feature" --body "Description"
```

### Review and Merge
```bash
# Review PR
gh pr view 123
gh pr diff 123

# Approve
gh pr review 123 --approve -b "LGTM!"

# Merge
gh pr merge 123 --squash --delete-branch
```

### Issue Management
```bash
# Create issue
gh issue create --title "Bug: Something broken" --body "Details"

# Label and assign
gh issue edit 456 --add-label "bug,high-priority" --add-assignee @me
```

## Automation Opportunities

When you detect repetitive GitHub tasks:

1. **PR Triage**: Automatically label PRs based on files changed
2. **Issue Routing**: Assign issues to teams based on area
3. **Release Automation**: Create release PRs, tag releases
4. **Dependency Updates**: Handle dependabot PRs

## Best Practices

- Always use `--dry-run` first for destructive operations
- Include co-authors for collaborative commits
- Reference issues in commits (Fixes #123)
- Squash merge for clean history