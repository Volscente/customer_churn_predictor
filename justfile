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

# Start Churn Predictor local REST API
start_local:
    poetry run uvicorn src.model_deployment.churn_predictor_rest_api:app --reload

# Build docker image
build_docker tag:
    docker image build -f ./docker/Dockerfile -t volscente/churn_predictor_image:{{tag}} .

# Start Churn Predictor REST API on Docker
start_docker:
    docker container run -p 8000:8000 --name churn_predictor_container volscente/churn_predictor_image:1.0.4
