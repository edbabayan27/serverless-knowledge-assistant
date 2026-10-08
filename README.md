# serverless-knowledge-assistant
Proof of concept: ask natural-language questions about the AWS Serverless Applications Lens. Serverless RAG pipeline built with Python Lambda, Amazon Bedrock and S3 Vectors, deployed end-to-end with Terraform.

## Project layout

```
src/knowledge_assistant/   Application code, packaged into the Lambda functions
tests/                     Unit tests (run locally, no AWS account needed)
pyproject.toml             Dependencies and tool settings (managed with uv)
uv.lock                    Exact dependency versions, also used for the Lambda package
```

## Local development

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.13 (the same version as the
Lambda runtime) automatically if you don't have it.

```bash
uv sync                  # create .venv with runtime and dev dependencies
uv run pytest            # run the unit tests
uv run ruff check .      # lint
uv run ruff format .     # format
```
