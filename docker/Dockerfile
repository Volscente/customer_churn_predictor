# Base Image
FROM python:3.12

# Install Poetry
RUN pip install poetry

# Change working directory
WORKDIR /app

# Copy code
COPY . /app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Install dependencies
RUN poetry install

CMD ["poetry", "run", "uvicorn", "src.model_deployment.churn_predictor_rest_api:app", "--reload"]