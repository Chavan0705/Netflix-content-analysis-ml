# 🎬 Netflix Content Analysis & Classification Model

An end-to-end data analysis, visualization, and machine learning project using the Netflix titles dataset (`netflix_titles.csv`). This project cleans and analyzes the catalog, builds binary classification models to predict whether a title is a **Movie** or a **TV Show**, generates detailed business insights, and implements a content-based recommendation system.

---

## 📂 Project Structure

*   **[Netflix_Analysis.ipynb](Netflix_Analysis.ipynb)**: The core Jupyter notebook containing all tasks, detailed step-by-step code, and inline plots.
*   **[netflix_titles.csv](netflix_titles.csv)**: The raw dataset containing Netflix catalog metadata.
*   **[netflix_cleaned.csv](netflix_cleaned.csv)**: Cleaned and preprocessed version of the dataset, exported for downstream tasks.
*   **[netflix_logistic_regression_model.pkl](netflix_logistic_regression_model.pkl)**: The trained best-performing classification model saved using `joblib`.
*   **[save_charts.py](save_charts.py)**: Python script used to programmatically generate and save all plots.
*   **[charts/](charts/)**: A folder containing 16 high-resolution visualization charts exported as `.png` files.

---

## 📊 Dataset Insights & Distribution

*   **Total Titles:** 8,807 rows
*   **Content Types:**
    *   **Movies:** 6,131 (69.62%)
    *   **TV Shows:** 2,676 (30.38%)
*   **Dominant Rating:** `TV-MA` (Mature Audience) is the most frequent rating, reflecting Netflix's focus on adult demographics.
*   **Top Contributing Country:** The United States leads by a wide margin, followed by India and the United Kingdom.

---

## 🤖 Machine Learning Performance

Three classifiers were trained on scaled numerical features (`release_year`, `duration_minutes`, `seasons`) and one-hot encoded categorical columns (`rating`, `primary_country`, `primary_genre`) to classify titles:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.9994** | **0.9992** | **1.0000** | **0.9996** | **1.0000** |
| **Gradient Boosting** | 0.9989 | 0.9984 | 1.0000 | 0.9992 | 1.0000 |
| **Random Forest** | 0.9949 | 0.9959 | 0.9967 | 0.9963 | 0.9997 |

*Note: Logistic Regression was selected as the best performing model due to its high F1-Score and generalizability on the encoded feature set.*

---

## 📈 Visualizations Catalog (`charts/`)

The exported charts in the `charts/` folder cover two main categories:

### 1. Exploratory Data Analysis (EDA)
*   `eda_content_by_type.png`: Bar chart showing Movie vs TV Show frequency.
*   `eda_content_added_over_time.png`: Stacked bar chart showing titles added per year since 2008.
*   `eda_top10_countries.png`: Horizontal bar chart of the top 10 content-producing countries.
*   `eda_top10_directors.png`: Top 10 directors by catalog count.
*   `eda_top15_genres.png`: Top 15 genres in the Netflix catalog.
*   `eda_rating_distribution.png`: Bar chart of age rating frequencies.
*   `eda_release_year_trend.png`: Line chart highlighting release trends since 1990.
*   `eda_movie_duration_distribution.png`: Histogram displaying movie runtimes in minutes.
*   `eda_tv_seasons_distribution.png`: Distribution of TV Show season lengths.

### 2. Model Evaluation
*   `chart1_content_type_distribution.png`: Side-by-side count bar chart and percentage pie chart of content types.
*   `chart2_top10_countries.png` & `chart3_top12_genres.png`: Styled horizontal bars for countries and genres.
*   `chart4_movie_duration_distribution.png`: Histogram detailing runtime averages.
*   `chart5_confusion_matrix.png`: Heatmap showing actual vs predicted labels for the best model.
*   `chart6_roc_curve.png`: ROC Curves comparing all three models.
*   `chart7_feature_importances.png`: Visualizing the features that influenced predictions the most.

---

## 🛠️ Execution & Setup

### Requirements
Ensure you have the following packages installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib plotly
```

### Run Project Tasks (Local)
To run the notebook and view the interactive visualizations, start Jupyter notebook:
```bash
jupyter notebook Netflix_Analysis.ipynb
```

To regenerate and save the charts into the `charts/` folder, run the utility script:
```bash
python save_charts.py
```

### Run in Google Colab (Cloud)
To run the analysis directly in the cloud:
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload [Netflix_Analysis.ipynb](Netflix_Analysis.ipynb).
3. Upload the dataset [netflix_titles.csv](netflix_titles.csv) to the Colab runtime session storage (using the files pane on the left).
4. Run all cells (`Ctrl + F9`).
5. (Optional) Download any created files (e.g. `netflix_cleaned.csv`, `netflix_logistic_regression_model.pkl`) directly from the session storage.

