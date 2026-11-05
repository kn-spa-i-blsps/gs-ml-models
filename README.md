# gs-ml-models

This is a repository for the machine learning models used in our project.
It servers as a unified way of dealing with our datasets and models, allowing for easy experimentation and consistent
deployment by different members of the team.

## Local setup

### Virtual environment

To setup the environment locally, it is recommended to use a virtual environment with [uv](https://docs.astral.sh/uv/).

If you have uv installed, setup the virtual environment by running:

```bash
uv venv
uv pip install -e .
```

To have access to the package in your current shell (for example in the python REPL) run the following command

```bash
source .venv/bin/activate
```

This imports all the package's scripts into the shell, allowing for convenient usage. In general though, this is not necessary if you run python scripts through `uv run <script>`.

### Data

The data loading functions assume that our data is stored in the `data` directory, with `classification_data` containing synthetic images, and `splatdata.csv` containing the labels. All the data loading functions allow for custom paths, but using the default path will help if the paths change later. The exact data storage convention is subject to change, so keep that in mind.
