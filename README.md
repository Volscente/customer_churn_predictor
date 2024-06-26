# Customer Churn Predictor
Develop an end-to-end solution for predicting customer churn for a subscription-based service.

# Setup
## Update PYTHONPATH
Add the current directory to the `PYTHONPATH` environment variables.
``` bash
export PYTHONPATH="$PYTHONPATH:/<absolute_path>/MediBioticsAI"
```

## Justfile
> `just` is a handy way to save and run project-specific commands
> 
> The main benefit it to keep all configuration and scripts in one place.
> 
> It uses the `.env` file for ingesting variables.

You can install it by following the [Documentation](https://just.systems/man/en/chapter_4.html).
Afterwards, you can execute existing commands located in the `justfile`.

Type `just` to list all available commands.

# Usage
1. Build the docker image through the just command `just build_docker 1.0.4`.

2. Start the docker container through the just command `just start_docker`

3. Navigate to http://0.0.0.0:8000/docs

4. Try a customer ID from `data/prediction/sample.csv`

## Poetry

> Python packaging and dependency management made easy

### Installation

[Reference Documentation](https://python-poetry.org/)

Run the following command from the terminal:
``` bash
curl -sSL https://install.python-poetry.org | python3 -
```

For **MacOS** with ZSH add the `.local/bin` to the `PATH` environment variable. Modify the `.zshrc` file with the following command:

``` bash
export PATH="$HOME/.local/bin:$PATH"
```

### Add Dependency
``` bash
# NOTE: Use '--group dev' to install in the 'dev' dependencies list
poetry add <library_name>

poetry add <library> --group dev

poetry add <libarry> --group <group_name>
```

### Install Dependencies
``` bash
# Install the dependencies listed in pyproject.toml [tool.poetry.dependencies]
poetry install

# Use the option '--without test,docs,dev' if you want to esclude the specified group from install
poetry install --without test,docs,dev
```

## MLflow
### Usage
```bash
cd notebooks
poetry run mlflow ui
```