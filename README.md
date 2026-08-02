# Data Science Projects

Central repository for data science work: reproductions of existing studies
and original analyses, developed collaboratively through prompts to Claude.

## What lives here

Two kinds of projects, both organized the same way:

1. **Replication projects** — reproducing and validating existing studies,
   papers, tutorials, or analyses, using the original (or a similar) dataset.
2. **Original analyses** — new work built from a topic, research question,
   dataset, or idea supplied in a prompt: our own data collection, cleaning,
   exploration, visualization, modelling, and evaluation.

## Repository layout

```
projects/
  <project-slug>/
    README.md          # objective, dataset, methodology, how to reproduce
    requirements.txt    # or environment.yml / renv.lock, as appropriate
    data/                # raw/processed data, or scripts to fetch it
    notebooks/           # exploratory notebooks, if any
    src/                 # reusable, modular code
    results/             # figures, tables, model artifacts
templates/
  project-template/      # starting point for a new project folder
```

Each project is self-contained under `projects/<project-slug>/` and does not
depend on code from other projects.

## Adding a new project

1. Copy `templates/project-template/` to `projects/<project-slug>/`.
2. Fill in the project README: objective, research question, dataset source,
   methodology, results, limitations, conclusions, and reproduction steps.
3. Add a dependency file (`requirements.txt`, `environment.yml`, or the
   language-appropriate equivalent) that pins working versions.
4. Record data sources, random seeds, and modelling decisions needed to
   reproduce the results.

## Conventions

See `CLAUDE.md` for the full set of working conventions this repository
follows.
