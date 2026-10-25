# AI Testing Project — RAG Application Evaluation

A hands-on AI testing project focused on evaluating LLM response quality and retrieval effectiveness using Python, pytest, and DeepEval.

## Project Overview

This project demonstrates automated testing of a Retrieval-Augmented Generation (RAG) application that answers questions using a sample subscription-policy knowledge base.

The tests evaluate whether AI-generated answers are supported by retrieved context, whether responses are relevant to user questions, and whether relevant policy documents are retrieved.

**Project type:** Personal learning and portfolio project.

## Technology Stack

- Python
- pytest
- DeepEval
- OpenAI API
- RAG evaluation

## Testing Scope

### Faithfulness
Evaluates whether AI-generated claims are supported by the retrieved context.

A negative test supplies an incorrect answer guaranteeing a full refund, contradicting the policy. DeepEval returned a faithfulness score of `0.0` for this answer.

### Answer Relevancy
Evaluates whether the generated answer addresses the user's question.

### Contextual Precision
Evaluates whether relevant information is ranked appropriately in the retrieved context.

### Contextual Recall
Evaluates whether the retrieved context contains the information needed to answer the question.

## Test Scenarios

- Verify relevant policy retrieval for different question phrasings.
- Check whether expected policy documents appear in the top retrieved results.
- Evaluate subscription, cancellation, and refund answers.
- Detect unsupported claims and contradictions with retrieved context.
- Check how the application responds when policy information is insufficient.
- Evaluate a small dataset using multiple DeepEval metrics.

## Project Structure

```text
SubscriptionRAGAgent/
├── app/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites
- Python 3.13 or a compatible version
- Git
- An OpenAI API key with access to the configured models

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SubscriptionRAGAgent
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Use your own API key. Never commit `.env` or expose credentials.

### 5. Run the tests

```bash
python -m pytest tests/test_retrieval.py -v -s
python -m pytest tests/test_deepeval_rag.py -v -s
python tests/test_rag_dataset.py
python -m pytest tests/test_eval_valuator.py -v -s
```

Run all pytest-discoverable tests:

```bash
python -m pytest tests/ -v -s
```

Some tests make external API calls and may incur charges.

## Initial Evaluation Results

The initial dataset evaluation used three test cases.

| Metric | Score | Passed |
|---|---:|---:|
| Contextual Precision | 1.00 | 3/3 |
| Contextual Recall | 1.00 | 3/3 |
| Answer Relevancy | 1.00 | 3/3 |
| Faithfulness | 1.00 | 3/3 |

A separate negative test returned a faithfulness score of `0.0` for a deliberately contradictory refund answer.

These are initial results from a small dataset, not a comprehensive benchmark or a guarantee of production readiness.

## Future Improvements

- Add prompt-injection and adversarial test scenarios.
- Test missing, incomplete, and conflicting retrieval context.
- Expand the evaluation dataset.
- Track evaluation results across prompt and retrieval changes.
- Add continuous integration for automated test execution.

## Disclaimer

This is an independent portfolio project using sample subscription policies. It is not a production customer-support system.