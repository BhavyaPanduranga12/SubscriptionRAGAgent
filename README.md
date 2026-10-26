# SubscriptionRAGAgent — AI Testing Project

A hands-on project focused on testing a Retrieval-Augmented Generation (RAG) application using Python, pytest, and DeepEval.

## Overview

I built an automated evaluation suite for a subscription-policy RAG application. The tests check answer quality, retrieval effectiveness, and how the application handles questions when policy information is missing or insufficient.

## Tech Stack

- Python and pytest
- DeepEval
- LangChain
- OpenAI API
- RAG evaluation

## What I Tested

- **Faithfulness:** Checks whether answers are supported by retrieved context.
- **Answer Relevancy:** Checks whether answers address the user's questions.
- **Contextual Relevancy:** Checks whether retrieved information is relevant to the question.
- **Contextual Precision:** Evaluates the ranking of relevant and irrelevant retrieved information.
- **Contextual Recall:** Checks whether the retrieved context contains the information needed to support the expected answer.

I also used parameterized test data, reusable metric factories, and pytest fixtures to organize the test suite.

## Initial Results

- Contextual Precision: 3/3 tests passed (scores: 1.00, 1.00, 0.83).
- Contextual Recall: 3/3 tests passed (all scores: 1.00).
- A deliberately contradictory refund answer received a Faithfulness score of 0.0.

These results are from a small test dataset and are intended for learning and experimentation.

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Create a `.env` file with the required OpenAI API key and model configuration.
5. Run the tests:

   ```bash
   python -m pytest tests/ -v
   ```

Some tests call external APIs and may incur charges.

## Disclaimer

An independent learning and portfolio project using sample subscription policies. Not intended for production use.
