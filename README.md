# Powerpoint Speaking Notes Agent
Agent for creating speaking notes and possible follow up questions for a Powerpoint presentation rendered as a PDF file in handout format.

Agent currently uses GPT-4o, but could be adapted to use other models with multimodel capabilities.

## Pre-requisites

Install `poppler` on your system for converting PDF files to images.:

```bash
brew install poppler
```

## Dependency installation using Poetry

Install `poetry` if you haven't already:

```bash
brew install poetry
```

Navigate to the root directory of the project and install the dependencies using:

```bash
poetry install
```

## Dependency installation using `pip` or `conda`

If you prefer to use `venv` or `conda`, you can install the dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

```bash
conda install --file requirements.txt
```

## Environment Variables

Store your environment variables in a `.env` file in the root directory of the project. See the `.env.example` file for the expected variables.

## Important Notes

See `test_speaking_notes.ipynb` for an example of how to use the agent.  The main function is `speaking_notes()` which takes the following parameters and returns a csv file with the speaking notes and follow-up questions:

*  `file`: a path to a pdf file, which should be a print of a set of powerpoint slides in handout format (6 to a page works fine)
*  `audience`: a description of the intended audience for the presentation.
*  `context`: context or background information relevant to the presentation.
*  `output_file`: a path to a CSV file where the speaking notes and follow-up questions will be saved.
*  `model`: optional - the OpenAI model to use for analysis (default is "gpt-4o").