# Machine learning enables the design of highly effective natural and modified small interfering RNA

## Abstract

Small interference RNAs (siRNAs) have emerged as pivotal molecular systems in therapy and diagnostics due to their regulatory roles in gene expression. The design of siRNAs is complex, influenced by various factors such as chain length, mismatches, and chemical modifications. These factors complicate their design and application.

This project proposes a novel method for predicting gene knockout activity of chemically modified siRNAs using machine learning (ML), leveraging descriptors of siRNA chemical composition with a convolutional autoencoder (CAE) architecture. Our model, based on Light Gradient Boosting Machine (LGBM), achieves a Root Mean Squared Error (RMSE) as small as 13.8% in activity evaluation on unseen systems, demonstrating its strong predictive capabilities.

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
 
