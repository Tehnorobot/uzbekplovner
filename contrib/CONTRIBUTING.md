# Fundamental Rules for Working with the Repository

Before starting, please familiarize yourself with the following documents:

- [Onboarding Process](ONBOARDING.md) — environment setup, launching services, and running tests.
- [Development and Testing Guide](DEVELOPMENT_GUIDE.md) — coding, testing, and security standards.
  
## Quick Start

1. Set up the local environment according to `ONBOARDING.md`.
2. Follow the coding and testing standards described in `DEVELOPMENT_GUIDE.md`.
3. Submit all changes via PR in accordance with `CONTRIBUTING.md`.

## General Contribution Principles

- All changes must be submitted through a Pull Request (PR).
- Direct push to the main branches (`main` or `dev`) is prohibited.
- Each PR must be atomic, logically consistent, and free of irrelevant modifications.

A PR must include:
- a description of the task and the essence of the changes;
- references to issues/tasks;
- a list of key modifications;
- testing instructions.

Before submitting a PR:
- run all tests;
- perform static analysis (Ruff);
- verify formatting (Ruff);
- ensure there are no changes unrelated to the task.

Code Review:
- Reviewer comments must be addressed or justified.
- A PR may not be merged before CI and all checks have passed.

## Branch Management (based on Gitflow principles)

Description of [Gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).

### Primary Branches

* **main** — stable *release* versions. Updated only through verified changes from `dev`.
* **dev** — integration branch. Contains *stable builds* that have passed initial verification.

### Task Branching

* **feat/*** — functional enhancements.
* **fix/*** — defect corrections.

Both categories are always created **from the `dev` branch** and, upon completion, are merged back into `dev` via a pull request. This maintains a structure analogous to Gitflow: short-lived task branches and centralized integration.

### Branching Rules

| Branch Source          | Purpose / When It Is Created           | Target / Where It Is Merged                                                      |
| ---------------------- | --------------------------------------- | -------------------------------------------------------------------------------- |
| **dev**                | Base branch for development             | `main` (release), `docs` (when needed), `exp` (optional)                         |
| **feat/*** (from `dev`) | Implementation of new functionality     | `dev`                                                                             |
| **fix/*** (from `dev`)  | Bug fixes                               | `dev`                                                                             |
| **main**               | Storage of stable releases              | Not merged anywhere directly; updated only from `dev`                             |
