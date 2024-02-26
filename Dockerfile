FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Install OS dependencies
RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install git

# Install poetry
RUN pip install 'poetry'

# Disable virtualenv creation
RUN poetry config virtualenvs.create false

# Copy Source
WORKDIR /work
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
COPY main.py main.py

# Install application dependencies
RUN poetry install

ENTRYPOINT ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]