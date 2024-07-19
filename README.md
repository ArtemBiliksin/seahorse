# Repository to reproduce the metrics for question Q4 from the article "SEAHORSE: A Multilingual, Multifaceted Dataset for Summarization Evaluation" using google/seahorse-large-q4

# How to run

### 1. Cloning a repository to a local computer
```
git clone https://github.com/ArtemBiliksin/seahorse.git
```

### 2. Setting up the environment
```
pip install -r requirements.txt
```

### 3. Downloading the SEAHORSE dataset and recovering the original articles from the GEM benchmark on which the summary was made
```
bash scripts/download_seahorse.sh
```

### 4. Computing google/seahorse-large-q4 model predictions
```
bash scripts/get_google_seahorse_ratings.sh
```

### 5. Computation of the Pearson correlation and roc-auc metric
```
bash scripts/get_google_seahorse_metrics.sh
```
