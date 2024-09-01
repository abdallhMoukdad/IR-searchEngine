Project Title: Information Retrieval System for Large Document Corpus

Description: This repository contains the implementation of an advanced Information Retrieval (IR) system designed to efficiently index and retrieve relevant documents from a large corpus of approximately 400,000 documents. The system uses TF-IDF vectorization and cosine similarity to process and evaluate queries. It is built with scalability in mind, offering features like batch processing, checkpointing, and graceful shutdown to handle large datasets effectively.

Features:

    Efficient Indexing: Multiple indexing strategies to accommodate different performance and resource constraints.
    Batch Processing: Processes documents in batches to optimize memory usage and performance.
    Checkpointing: Allows the indexing process to be paused and resumed without data loss, ensuring robustness.
    API Endpoints: Provides RESTful API endpoints using Flask for easy integration and interaction.
    Performance Evaluation: Includes modules for evaluating the system's performance using precision and recall metrics.
    Scalability: Designed to handle large document corpora with efficient memory management and processing strategies.

Technologies Used:

    Python
    Flask
    Scikit-learn
    NLTK
