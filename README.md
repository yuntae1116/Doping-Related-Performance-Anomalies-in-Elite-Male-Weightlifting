# Supplementary Material – Artificial Intelligence-Based Modeling for Detecting Doping-Related Performance Anomalies in Elite Male Weightlifting

This repository provides supplementary materials associated with the manuscript:  
**“Artificial Intelligence-Based Modeling for Detecting Doping-Related Performance Anomalies in Elite Male Weightlifting.”**

## Contents
- `new_male_test.csv` – Synthetic dataset (100 records) generated to mimic the structure of the real competition data.  
- `male_weightlift_script.py` – Python script implementing the preprocessing, clustering, and regression analyses described in the paper.

## Requirements
- Python 3.11
- pandas >= 2.2
- numpy >= 1.26
- scikit-learn >= 1.5
- seaborn >= 0.13
- matplotlib >= 3.9
- yellowbrick (for clustering visualization)

## Usage
Run the analysis script with:
```bash
python male_weightlift_script.py
```
This will load `new_male_test.csv` and demonstrate the full workflow:  
data preprocessing, K-Means clustering, regression modeling (Decision Tree, Random Forest, K-Neighbors), and evaluation.

Note: Results will **not exactly reproduce** the manuscript because the dataset here is synthetic.

## Data Availability
The file `new_male_test.csv` contains only **synthetic data** (100 randomly generated records) for demonstration and reproducibility.  

The **real dataset** is **available from the corresponding author upon reasonable request**, as stated in the manuscript.

## License
These supplementary materials are provided for reproducibility and educational purposes only.
