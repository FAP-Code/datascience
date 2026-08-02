# Data Science Projects — Working Conventions

Read this before starting or touching a project.

## Purpose

This repository is the central home for two kinds of data science work,
developed project-by-project from prompts:

1. **Replication projects** — reproduce and validate an existing study,
   paper, tutorial, or analysis using the original or a similar dataset.
2. **Original analyses** — new work on a topic, research question, dataset,
   or idea supplied via prompt: our own collection, cleaning, exploration,
   visualization, modelling, and evaluation.

## Per-project responsibilities

For every project, unless told otherwise:

- Help refine the topic and research question before writing code.
- Identify or recommend suitable datasets, and note why they fit.
- Perform data collection, cleaning, exploration, visualization, modelling,
  and evaluation, as applicable to the project.
- Write clean, modular, well-documented Python or R code.
- Explain methods, results, limitations, and conclusions in the project
  README, not just in code comments.
- Ask for clarification when a major requirement is genuinely unclear —
  don't guess at research questions, dataset choice, or modelling approach.

## Structure

- One project per folder under `projects/<project-slug>/`. Use
  `templates/project-template/` as the starting point.
- Each project README covers: objective, dataset (source, license, access
  method), methodology, installation, and steps to reproduce the results.
- Each project has its own dependency file — `requirements.txt`,
  `environment.yml`, or the equivalent for the language used (e.g. R's
  `renv.lock`). Don't share dependency files across projects.
- Projects are self-contained: no project imports code from another
  project's `src/`.

## Reproducibility

- Document data sources (URL, version/date accessed, license) in the
  project README.
- Set and record random seeds used in any stochastic step (splits,
  initialization, sampling).
- Record software/package versions in the dependency file, not just in
  prose.
- Note modelling decisions (why this model, why this metric, what was
  tried and rejected) in the README's methodology or limitations section.

## Editing existing projects

- Do not alter or delete an existing project's data, code, or results
  unless specifically asked to.
- New work on a topic close to an existing project still gets its own
  folder unless asked to extend the existing one in place.
