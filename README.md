# Multi-Model AI E-Commerce Recommender System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet)](https://mlflow.org/)

A production-grade recommendation engine leveraging **Hybrid Reinforcement Learning** and **Transformer Architectures** to optimize user engagement and long-term value (LTV).

---

## System Architecture

This project implements a **Multi-Stage Recommendation Pipeline** designed to handle the full user journey, from cold-start to loyal engagement.

| Component | Model Architecture | Role |
| :--- | :--- | :--- |
| **1. Discovery Engine** | **Collaborative Filtering (VAECF)** | Finds latent user preferences from implicit feedback (Clicks/Views). |
| **2. Similarity Engine** | **Content-Based (BERT)** | Recommends "More like this" using semantic embeddings of product metadata. |
| **3. Sequence Engine** | **Transformer (SASRec-style)** | Predicts *next-item* intent based on immediate user history. |
| **4. Decision Engine** | **Hybrid RL Agent (Transformer-DQN)** | **The Core Innovation.** Optimizes the final ranking to maximize long-term reward (Purchase vs. Click). |

---

## Key Performance Metrics

### 1. Sequential Model Performance (Transformer)
Our Transformer-based sequence model significantly outperforms standard baselines, achieving **100% Catalog Coverage** through a tiered inference strategy.

| Metric | Score | Business Impact |
| :--- | :--- | :--- |
| **Recall@10** | **8.72%** | **1.04x** lift vs. Popularity Baseline. Users see more relevant items. |
| **Coverage@50** | **99.40%** | The model recommends niche items, preventing "Popularity Bias." |
| **Recall@50** | **13.00%** | Strong candidate retrieval for the RL ranking stage. |

> *Data Source: MLflow Evaluation Logs*

### 2. Reinforcement Learning Convergence (Sim-to-Real)
The Hybrid RL Agent was trained using a **Sim-to-Real** workflow in a custom Gymnasium environment.
* **Stable Convergence:** The agent successfully transitioned from exploration to exploitation, with epsilon decaying to 0.05.
* **Reward Optimization:** Cumulative reward stabilized, indicating the agent learned to prioritize high-value actions (Purchases) over low-value ones (Views).

<img width="301" height="312" alt="Image" src="https://github.com/user-attachments/assets/f2d485fa-d51e-4347-9d75-515a6151dd94" />
*(Epsilon Decay vs. Cumulative Reward over 400 Episodes)*

---

## Deployment & MLOps

### Tech Stack
* **Training:** TensorFlow/Keras, Cornac (for VAE), Gymnasium (RL Env).
* **Tracking:** MLflow & DagsHub (Experiment tracking).
* **Serving:** FastAPI (Asynchronous inference).
* **Edge Optimization:** TensorFlow Lite (TFLite) for <10ms inference latency.
* **Containerization:** Docker for reproducible deployment.

### How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/okefemi12/ecommerce-recommender-system.git](https://github.com/okefemi12/ecommerce-recommender-system.git)
    cd ecommerce-recommender
    ```

2.  **Start the API (Docker)**
    ```bash
    docker build -t recommender-api .
    docker run -p 8000:8000 recommender-api
    ```

3.  **Test the Endpoint**
    ```bash
    curl -X POST "http://localhost:8000/recommend" \
         -H "Content-Type: application/json" \
         -d '{"history": [101, 204, 305], "user_id": "user_123"}'
    ```

---

## 📂 Project Structure

* `src/`: Production FastAPI code and inference logic.
* `notebooks/`: Detailed Data Science experiments (EDA, Training pipelines).
* `models/`: Serialized model artifacts (TFLite, Keras).
* `tests/`: Unit and integration tests.

---
