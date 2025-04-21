# Objective 1: Clean and Manipulate Data using Pandas and NumPy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "#ffffff"

# Load dataset
df = pd.read_csv("AirQuality Python Dataset.csv")
print("Original Data:\n", df.head())

# Add a randomized datetime column
date_range = pd.date_range(start='2022-01-01', end='2025-12-31', periods=len(df))
np.random.seed(42)
df['last_update'] = np.random.permutation(date_range)

# Save and re-load to mimic real workflow
df.to_csv('Modified_AirQuality.csv', index=False)
df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')

# Remove rows with missing essential data
df_cleaned = df.dropna(subset=['pollutant_id', 'pollutant_avg', 'last_update'])

# Objective 2: Transform Data with Pivot Table

# Create pivot table (pollutant_id values as columns, using average)
pivot_df = df_cleaned.pivot_table(
    index=['country', 'state', 'city', 'station'],
    columns='pollutant_id',
    values='pollutant_avg',
    aggfunc='mean'
).reset_index()

# Add back the datetime information by averaging per station
last_update_map = df_cleaned.groupby(['country', 'state', 'city', 'station'])['last_update'].max().reset_index()
pivot_df = pd.merge(pivot_df, last_update_map, on=['country', 'state', 'city', 'station'], how='left')

# Rename pollutant columns to uppercase
pivot_df.columns.name = None
pivot_df = pivot_df.rename(columns=lambda col: col.upper() if isinstance(col, str) else col)

# Confirm structure
print("\nPivoted and Cleaned Data:")
print(pivot_df.head())

# Objective 3: Data Summary and Statistics
print("\nDescriptive Stats:")
print(pivot_df.describe())

print("\nInfo:")
print(pivot_df.info())

# Objective 4: Visualize Pollutant Trends using Line Plot

# Line Plot: PM2.5 trend over time
if 'PM2.5' in pivot_df.columns:
    pm25_df = pivot_df[['LAST_UPDATE', 'PM2.5']].dropna().sort_values(by='LAST_UPDATE')

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=pm25_df, x='LAST_UPDATE', y='PM2.5', marker='o')
    plt.title('Trend of PM2.5 Over Time')
    plt.xlabel('Date')
    plt.ylabel('PM2.5 Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Column 'PM2.5' not found in dataset.")

# Line Plot: PM10 trend over time
if 'PM10' in pivot_df.columns:
    pm10_data = pivot_df[['LAST_UPDATE', 'PM10']].dropna()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=pm10_data, x='LAST_UPDATE', y='PM10', label='PM10', color='orange')
    plt.title('PM10 Trend Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# Objective 5: Box Plot and Strip Plot to Show Spread and Distribution of Each Pollutant

pollutant_columns = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE']
available_pollutants = [col for col in pollutant_columns if col in pivot_df.columns]

#Box Plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=pivot_df[available_pollutants], palette='Set2')
plt.title("Box Plot of Pollutant Distributions")
plt.ylabel("Concentration")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Strip Plot
melted_df = pivot_df.melt(value_vars=available_pollutants, var_name='Pollutant', value_name='Concentration')
plt.figure(figsize=(12, 6))
sns.stripplot(data=melted_df, x='Pollutant', y='Concentration', jitter=True, palette='Dark2', alpha=0.5)
plt.title("Strip Plot of Individual Pollutant Values")
plt.ylabel("Concentration")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 6: Simplified Correlation Heatmaps for Pollutants

corr_matrix = pivot_df[available_pollutants].corr()

#heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix of Pollutants")
plt.tight_layout()
plt.show()

# Objective 7: Bar Plot - Average Pollutant Levels by City

city_avg = df_cleaned.groupby('city')['pollutant_avg'].mean().nlargest(10).reset_index()

plt.figure(figsize=(12, 5))
sns.barplot(data=city_avg, x='city', y='pollutant_avg', palette='viridis')
plt.title("Top 10 Polluted Cities (Average Pollutant Levels)")
plt.xlabel("City")
plt.ylabel("Average Level")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Objective 8: # Advanced Scatter Plots - PM2.5 vs PM10

#Scatter Plot
if 'PM2.5' in pivot_df.columns and 'PM10' in pivot_df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pivot_df, x='PM2.5', y='PM10', alpha=0.6, color='green')
    plt.title("Scatter Plot: PM2.5 vs PM10")
    plt.xlabel("PM2.5")
    plt.ylabel("PM10")
    plt.tight_layout()
    plt.show()


    top_cities = pivot_df['CITY'].value_counts().head(5).index
    city_filtered_df = pivot_df[pivot_df['CITY'].isin(top_cities)]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=city_filtered_df, x='PM2.5', y='PM10', hue='CITY', palette='Set2', alpha=0.7)
    plt.title("PM2.5 vs PM10 by City")
    plt.xlabel("PM2.5")
    plt.ylabel("PM10")
    plt.legend(title="City", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

#  Objective 9: Air Quality Index (AQI) Categorization (Binning)
def categorize_pm25(val):
    if val <= 50:
        return 'Good'
    elif val <= 100:
        return 'Moderate'
    elif val <= 150:
        return 'Unhealthy for Sensitive'
    elif val <= 200:
        return 'Unhealthy'
    else:
        return 'Very Unhealthy'

if 'PM2.5' in pivot_df.columns:
    pivot_df['PM2.5_Level'] = pivot_df['PM2.5'].apply(categorize_pm25)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=pivot_df, x='PM2.5_Level', order=['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy'], palette='Reds')
    plt.title("PM2.5 Level Categories")
    plt.ylabel("Number of Stations")
    plt.tight_layout()
    plt.show()