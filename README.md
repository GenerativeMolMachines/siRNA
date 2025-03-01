# Machine learning on autoencoder- and LLM-derived embeddings for the design of highly effective chemically modified siRNAs for gene knockdown

## Abstract

Six small interference RNAs (siRNAs) have been approved as therapeutics making them promising nanosystems due to selective gene knockdown activity. siRNA design is complex due to various factors, where the chemical modifications are crucial to improve its half-life and stability. Machine learning (ML) enabled more efficient analysis of siRNA data, moreover predicting efficacy and off-target effects. This work proposes a novel pipeline for predicting gene knockdown activity of chemically modified siRNAs across the whole range of activities leveraging both descriptors of siRNA chemical composition-aware property matrices and large language model (LLM) embeddings for target gene encoding. Several general-purpose and domain-specific fine tuned LLMs were benchmarked on the target task, where the Mistral 7B general-purpose model outperformed even the models pre-trained on genomic data. Proposed model based on meta-learning mechanism successfully mitigates data imbalance towards moderate-to-high active constructs and achieves state-of-the-art (SOTA) quality with R2 = 0.84 and a RMSE = 12.27% on unseen data, where the probabilistic outputs of classifiers trained with F-scores up to 0.92 were used as additional descriptors. By filling the gap in the field of advanced chemical composition-aware siRNA design, our model aims to improve the efficacy of developed siRNA-based therapies.

## Key Features

- Advanced prediction of siRNA efficacy and specificity.
- Utilization of chemically aware descriptors for siRNA design.
- Application of ML techniques to enhance the accuracy of predictions.

## Project Structure

The repository contains the following directories:

- **Figures**: Contains code for visualizing figures in the article (excluding the regression and classification plots, which is located at the end of the regression and binary classification files .ipynb).
  
- **Dataset**: Includes various versions of the dataset obtained during cleaning and unification.

- **Regressor**: Contains files aimed at developing the final model, comparing descriptors and models.

- **Unified_data**: Includes notebooks and other files that facilitate obtaining the final version of the dataset.

- **Classifier**: Contains code for two types of classifiers used to build and optimize our model.
 
