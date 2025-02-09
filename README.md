# Machine learning on autoencoder- and LLM-derived embeddings for the design of highly effective chemically modified siRNAs for gene knockdown

## Abstract

Small interference RNAs (siRNAs) have emerged as pivotal molecular systems in the fields of therapy and diagnostics due to their regulatory roles in gene expression. Their design is complex due to various factors influencing their activity, including chain length, mismatches, and chemical modifications. These factors exert complex effects on the activity of siRNA therapies, which can complicate their design and application. The experimental design of siRNAs presents a substantial combinatorial challenge, as current methodologies lack versatile design rules. Machine learning (ML) has significantly advanced siRNA research, enabling more efficient and accurate analysis of siRNA data, particularly in predicting efficacy and off-target effects. Existing studies primarily rely on statistical representations of sequences that consider only nucleotide context, often failing to account for the chemical properties of natural and modified nucleotides. Consequently, models developed on such bases require extensive datasets to converge towards sufficient predictive accuracy. 

This work proposes a novel method for predicting gene knockout activity of chemically modified siRNAs leveraging both descriptors of siRNA chemical composition obtained with convolutional autoencoder (CAE) architecture and Mistral language model embeddings for target gene encoding. Model based on Light Gradient Boosting Machine (LGBM) achieves state-of-the-art (SOTA) quality with R2 = 0.87 and a root mean squared error (RMSE) as small as 10.89% in activity evaluation on unseen data, demonstrating its strong predictive capabilities, where binary and multiclass classification algorithms were trained with F-scores up to 0.92 for iterative filtration of poorly active siRNAs to account for data imbalance towards highly active constructs outperforming all existing classifiers. By filling the gap in the field of advanced chemical composition-aware siRNA design, our model aims to improve the efficacy of siRNA-based therapies.

## Key Features

- Advanced prediction of siRNA efficacy and specificity.
- Utilization of chemically aware descriptors for siRNA design.
- Application of ML techniques to enhance the accuracy of predictions.

## Project Structure

The repository contains the following directories:

- **Figures**: Contains code for visualizing figures in the article (excluding the regression plot, which is located at the end of the regression file .py).
  
- **Dataset**: Includes various versions of the dataset obtained during cleaning and unification.

- **Regressor**: Contains files aimed at developing the final model, comparing descriptors and models.

- **Unified_data**: Includes notebooks and other files that facilitate obtaining the final version of the dataset.

- **Classifier**: Contains code for two types of classifiers used to build and optimize our model.
 
