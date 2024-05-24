#!/usr/bin/env just --justfile

# Load from '.env' file
set dotenv-load

# List available commands
help:
    @just --justfile {{justfile()}} --list --unsorted

# PyLint
lint:
  # PyLint lint from ./src and ./tests
  ./scripts/pylint_lint.sh


# Execute PyTests under '/tests'
test:
    poetry run pytest

# Start Jupyter Lab with the Poetry Virtual Environment
jupy:
  poetry run jupyter lab