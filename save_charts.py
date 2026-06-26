import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid GUI window opening
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)

# Create charts/ directory if it doesn't exist
os.makedirs('charts', exist_ok=True)
print("Created 'charts/' directory.")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('netflix_titles.csv')

# Data Cleaning & Preprocessing
print("Cleaning data...")
df_clean = df.copy()
df_clean['director'].fillna('Unknown', inplace=True)
df_clean['cast'].fillna('Unknown', inplace=True)
df_clean['country'].fillna('Unknown', inplace=True)
df_clean['date_added'].fillna('January 1, 2000', inplace=True)
df_clean['rating'].fillna(df_clean['rating'].mode()[0], inplace=True)
df_clean['duration'].fillna('0 min', inplace=True)
df_clean.drop_duplicates(inplace=True)

df_clean['date_added'] = pd.to_datetime(df_clean['date_added'].str.strip(), format='mixed', errors='coerce')
df_clean['year_added']  = df_clean['date_added'].dt.year
df_clean['month_added'] = df_clean['date_added'].dt.month

df_clean['duration_minutes'] = df_clean.apply(
    lambda row: int(row['duration'].replace(' min', '').strip())
    if row['type'] == 'Movie' and 'min' in str(row['duration'])
    else 0, axis=1
)

df_clean['seasons'] = df_clean.apply(
    lambda row: int(row['duration'].replace(' Seasons', '').replace(' Season', '').strip())
    if row['type'] == 'TV Show' and 'Season' in str(row['duration'])
    else 0, axis=1
)

df_clean['primary_country'] = df_clean['country'].str.split(',').str[0].str.strip()
df_clean['primary_genre']   = df_clean['listed_in'].str.split(',').str[0].str.strip()

top_countries = df_clean['primary_country'].value_counts().nlargest(15).index
top_genres    = df_clean['primary_genre'].value_counts().nlargest(15).index

df_clean['primary_country'] = df_clean['primary_country'].where(
    df_clean['primary_country'].isin(top_countries), other='Other')
df_clean['primary_genre'] = df_clean['primary_genre'].where(
    df_clean['primary_genre'].isin(top_genres), other='Other')

df_encoded = pd.get_dummies(df_clean, columns=['rating', 'primary_country', 'primary_genre'], drop_first=False)
df_encoded['type_encoded'] = (df_encoded['type'] == 'Movie').astype(int)

scaler = StandardScaler()
numeric_features = ['release_year', 'duration_minutes', 'seasons']
df_encoded[['release_year_scaled', 'duration_minutes_scaled', 'seasons_scaled']] = \
    scaler.fit_transform(df_encoded[numeric_features])

# Train models
print("Training models...")
ohe_cols     = [c for c in df_encoded.columns if c.startswith(('rating_', 'primary_country_', 'primary_genre_'))]
feature_cols = ['release_year_scaled', 'duration_minutes_scaled', 'seasons_scaled'] + ohe_cols

X = df_encoded[feature_cols].fillna(0)
y = df_encoded['type_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting'  : GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results        = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'F1 Score' : f1_score(y_test, y_pred),
        'ROC-AUC'  : roc_auc_score(y_test, y_prob)
    }
    trained_models[name] = (model, y_pred, y_prob)

results_df = pd.DataFrame(results).T.sort_values('F1 Score', ascending=False)
best_model_name              = results_df['F1 Score'].idxmax()
best_model, best_y_pred, best_y_prob = trained_models[best_model_name]

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
else:
    coefs       = best_model.coef_[0]
    feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': np.abs(coefs)})

feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False).head(10).reset_index(drop=True)

# Generate and save plots
print("Generating and saving plots...")

sns.set_theme(style='darkgrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

# --- Task 3 Plots ---

# 3.1 Content by Type
fig, ax = plt.subplots(figsize=(6, 4))
type_counts = df_clean['type'].value_counts()
colors = ['#E50914', '#564d4d']
bars = ax.bar(type_counts.index, type_counts.values, color=colors, edgecolor='white', linewidth=0.8)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{bar.get_height():,}', ha='center', va='bottom', fontweight='bold')
ax.set_title('🎬 Movies vs TV Shows on Netflix', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Titles')
ax.set_xlabel('Content Type')
plt.tight_layout()
plt.savefig('charts/eda_content_by_type.png', dpi=150)
plt.close()

# 3.2 Content Added Over Time
added_by_year = df_clean.groupby(['year_added', 'type']).size().unstack(fill_value=0)
added_by_year = added_by_year[added_by_year.index.notna() & (added_by_year.index >= 2008)]
fig, ax = plt.subplots(figsize=(12, 5))
added_by_year.plot(kind='bar', stacked=True, ax=ax, color=['#E50914', '#564d4d'])
ax.set_title('📅 Netflix Content Added Over the Years', fontsize=14, fontweight='bold')
ax.set_xlabel('Year Added')
ax.set_ylabel('Number of Titles')
ax.legend(title='Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('charts/eda_content_added_over_time.png', dpi=150)
plt.close()

# 3.3 Top 10 Countries
country_series = df_clean['country'].str.split(',').explode().str.strip()
country_series = country_series[country_series != 'Unknown']
top10_countries = country_series.value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
palette_c = sns.color_palette('Reds_r', 10)
ax.barh(top10_countries.index[::-1], top10_countries.values[::-1], color=palette_c[::-1])
ax.set_title('🌍 Top 10 Countries by Netflix Content', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Titles')
for i, v in enumerate(top10_countries.values[::-1]):
    ax.text(v + 10, i, str(v), va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/eda_top10_countries.png', dpi=150)
plt.close()

# 3.4 Top 10 Directors
director_series = df_clean['director']
director_series = director_series[director_series != 'Unknown']
director_series = director_series.str.split(',').explode().str.strip()
top10_directors = director_series.value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
palette_d = sns.color_palette('Blues_r', 10)
ax.barh(top10_directors.index[::-1], top10_directors.values[::-1], color=palette_d[::-1])
ax.set_title('🎥 Top 10 Directors on Netflix', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Titles')
for i, v in enumerate(top10_directors.values[::-1]):
    ax.text(v + 0.2, i, str(v), va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/eda_top10_directors.png', dpi=150)
plt.close()

# 3.5 Top Genres
genre_series = df_clean['listed_in'].str.split(',').explode().str.strip()
top15_genres = genre_series.value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 6))
palette_g = sns.color_palette('viridis', 15)
ax.barh(top15_genres.index[::-1], top15_genres.values[::-1], color=palette_g[::-1])
ax.set_title('🎭 Top 15 Genres on Netflix', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Titles')
for i, v in enumerate(top15_genres.values[::-1]):
    ax.text(v + 10, i, str(v), va='center', fontsize=9)
plt.tight_layout()
plt.savefig('charts/eda_top15_genres.png', dpi=150)
plt.close()

# 3.6 Rating Distribution
rating_counts = df_clean['rating'].value_counts().head(12)
fig, ax = plt.subplots(figsize=(10, 5))
palette_r = sns.color_palette('rocket', len(rating_counts))
bars = ax.bar(rating_counts.index, rating_counts.values, color=palette_r, edgecolor='white')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{bar.get_height():,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('⭐ Rating Distribution on Netflix', fontsize=14, fontweight='bold')
ax.set_xlabel('Rating')
ax.set_ylabel('Number of Titles')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('charts/eda_rating_distribution.png', dpi=150)
plt.close()

# 3.7 Release Year Trend
release_trend = df_clean['release_year'].value_counts().sort_index()
release_trend = release_trend[release_trend.index >= 1990]
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(release_trend.index, release_trend.values, color='#E50914', linewidth=2.5, marker='o', markersize=4)
ax.fill_between(release_trend.index, release_trend.values, alpha=0.2, color='#E50914')
ax.set_title('📈 Netflix Title Release Year Trend', fontsize=14, fontweight='bold')
ax.set_xlabel('Release Year')
ax.set_ylabel('Number of Titles')
plt.tight_layout()
plt.savefig('charts/eda_release_year_trend.png', dpi=150)
plt.close()

# 3.8 Movie Duration Distribution
movie_durations = df_clean[df_clean['type'] == 'Movie']['duration_minutes']
movie_durations = movie_durations[movie_durations > 0]
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(movie_durations, bins=40, color='#E50914', edgecolor='white', alpha=0.9)
ax.axvline(movie_durations.mean(),   color='gold',  linewidth=2, linestyle='--',
           label=f'Mean: {movie_durations.mean():.0f} min')
ax.axvline(movie_durations.median(), color='white', linewidth=2, linestyle='-.',
           label=f'Median: {movie_durations.median():.0f} min')
ax.set_title('⏱️ Movie Duration Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Duration (minutes)')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('charts/eda_movie_duration_distribution.png', dpi=150)
plt.close()

# 3.9 TV Seasons Distribution
tv_seasons = df_clean[df_clean['type'] == 'TV Show']['seasons']
tv_seasons = tv_seasons[tv_seasons > 0]
season_counts = tv_seasons.value_counts().sort_index().head(15)
fig, ax = plt.subplots(figsize=(8, 5))
palette_s = sns.color_palette('Blues_r', len(season_counts))
ax.bar(season_counts.index.astype(str), season_counts.values, color=palette_s, edgecolor='white')
ax.set_title('📺 TV Show Seasons Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Seasons')
ax.set_ylabel('Number of TV Shows')
plt.tight_layout()
plt.savefig('charts/eda_tv_seasons_distribution.png', dpi=150)
plt.close()

# --- Task 6 Plots ---

# Chart 1: Bar + Pie — Movies vs TV Shows
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
type_counts = df_clean['type'].value_counts()
axes[0].bar(type_counts.index, type_counts.values,
            color=['#E50914', '#564d4d'], edgecolor='white', linewidth=1.2)
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                 f'{bar.get_height():,}', ha='center', va='bottom', fontweight='bold')
axes[0].set_title('Movies vs TV Shows (Count)', fontweight='bold')
axes[0].set_ylabel('Titles')

axes[1].pie(type_counts.values, labels=type_counts.index,
            autopct='%1.1f%%', colors=['#E50914', '#564d4d'],
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Movies vs TV Shows (Share)', fontweight='bold')
plt.suptitle('Chart 1 — Content Type Distribution', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('charts/chart1_content_type_distribution.png', dpi=150)
plt.close()

# Chart 2: Top 10 Countries
country_series  = df_clean['country'].str.split(',').explode().str.strip()
country_series  = country_series[country_series != 'Unknown']
top10_countries = country_series.value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 6))
palette = sns.color_palette('YlOrRd', 10)[::-1]
bars = ax.barh(top10_countries.index[::-1], top10_countries.values[::-1], color=palette)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 15, bar.get_y() + bar.get_height()/2, f'{w:,.0f}', va='center', fontweight='bold')
ax.set_title('Chart 2 — Top 10 Countries by Netflix Content', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Titles')
ax.set_xlim(0, top10_countries.max() * 1.15)
plt.tight_layout()
plt.savefig('charts/chart2_top10_countries.png', dpi=150)
plt.close()

# Chart 3: Top Genres
genre_series  = df_clean['listed_in'].str.split(',').explode().str.strip()
top12_genres  = genre_series.value_counts().head(12)
fig, ax = plt.subplots(figsize=(10, 7))
palette_g = sns.color_palette('plasma', 12)
bars = ax.barh(top12_genres.index[::-1], top12_genres.values[::-1], color=palette_g[::-1])
for bar in bars:
    ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():,.0f}', va='center', fontsize=9)
ax.set_title('Chart 3 — Top 12 Netflix Genres', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Titles')
ax.set_xlim(0, top12_genres.max() * 1.15)
plt.tight_layout()
plt.savefig('charts/chart3_top12_genres.png', dpi=150)
plt.close()

# Chart 4: Movie Duration Histogram
movie_durations = df_clean[df_clean['type'] == 'Movie']['duration_minutes']
movie_durations = movie_durations[movie_durations > 0]
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(movie_durations, bins=50, color='#E50914', edgecolor='white', alpha=0.85)
ax.axvline(movie_durations.mean(),   color='gold',  linewidth=2.5, linestyle='--',
           label=f'Mean = {movie_durations.mean():.0f} min')
ax.axvline(movie_durations.median(), color='white', linewidth=2.5, linestyle='-.',
           label=f'Median = {movie_durations.median():.0f} min')
ax.set_title('Chart 4 — Movie Duration Distribution (Histogram)', fontsize=14, fontweight='bold')
ax.set_xlabel('Duration (minutes)')
ax.set_ylabel('Number of Movies')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('charts/chart4_movie_duration_distribution.png', dpi=150)
plt.close()

# Chart 5: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, best_y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['TV Show', 'Movie'],
            yticklabels=['TV Show', 'Movie'],
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title(f'Chart 5 — Confusion Matrix\n({best_model_name})', fontsize=13, fontweight='bold')
ax.set_ylabel('Actual Label')
ax.set_xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('charts/chart5_confusion_matrix.png', dpi=150)
plt.close()

# Chart 6: ROC Curve
colors_roc = ['#E50914', '#2196F3', '#4CAF50']
fig, ax = plt.subplots(figsize=(8, 6))
for (name, (model, y_pred, y_prob)), color in zip(trained_models.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='grey')
ax.set_title('Chart 6 (Bonus) — ROC Curve: All Models', fontsize=14, fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig('charts/chart6_roc_curve.png', dpi=150)
plt.close()

# Chart 7: Feature Importances
fig, ax = plt.subplots(figsize=(10, 6))
palette_fi = sns.color_palette('coolwarm', 10)
ax.barh(feat_imp_df['Feature'][::-1], feat_imp_df['Importance'][::-1], color=palette_fi)
ax.set_title(f'Chart 7 (Bonus) — Top 10 Feature Importances\n({best_model_name})',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
for i, (feat, imp) in enumerate(zip(feat_imp_df['Feature'][::-1], feat_imp_df['Importance'][::-1])):
    ax.text(imp + 0.001, i, f'{imp:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('charts/chart7_feature_importances.png', dpi=150)
plt.close()

print("All charts successfully generated and saved in 'charts/' directory.")
