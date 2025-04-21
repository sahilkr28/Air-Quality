# Objective 1: Clean and Manipulate Data using Pandas and NumPy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def load_and_prepare_data(file_path):
    """
    Load and prepare the air quality dataset.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned and prepared dataframe
    """
    try:
        df = pd.read_csv(file_path)
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
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def create_pivot_table(df_cleaned):
    """
    Create a pivot table from the cleaned data.
    
    Args:
        df_cleaned (pd.DataFrame): Cleaned dataframe
        
    Returns:
        pd.DataFrame: Pivot table with pollutants as columns
    """
    # Create pivot table
    pivot_df = df_cleaned.pivot_table(
        index=['country', 'state', 'city', 'station'],
        columns='pollutant_id',
        values='pollutant_avg',
        aggfunc='mean'
    ).reset_index()
    
    # Add back the datetime information
    last_update_map = df_cleaned.groupby(['country', 'state', 'city', 'station'])['last_update'].max().reset_index()
    pivot_df = pd.merge(pivot_df, last_update_map, on=['country', 'state', 'city', 'station'], how='left')
    
    # Rename pollutant columns to uppercase
    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns=lambda col: col.upper() if isinstance(col, str) else col)
    
    return pivot_df

def save_plot(fig, plot_name):
    """
    Save the plot to a plots directory.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save
        plot_name (str): Name of the plot
    """
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{plots_dir}/{plot_name}_{timestamp}.png"
    fig.savefig(filename)
    print(f"Plot saved as {filename}")

def plot_pollutant_trends(pivot_df):
    """
    Create and save line plots for PM2.5 and PM10 trends.
    
    Args:
        pivot_df (pd.DataFrame): Pivot table dataframe
    """
    # PM2.5 trend
    if 'PM2.5' in pivot_df.columns:
        pm25_df = pivot_df[['LAST_UPDATE', 'PM2.5']].dropna().sort_values(by='LAST_UPDATE')
        fig = plt.figure(figsize=(12, 6))
        sns.lineplot(data=pm25_df, x='LAST_UPDATE', y='PM2.5', marker='o')
        plt.title('Trend of PM2.5 Over Time')
        plt.xlabel('Date')
        plt.ylabel('PM2.5 Level')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_plot(fig, 'pm25_trend')
        plt.close()

    # PM10 trend
    if 'PM10' in pivot_df.columns:
        pm10_data = pivot_df[['LAST_UPDATE', 'PM10']].dropna()
        fig = plt.figure(figsize=(10, 5))
        sns.lineplot(data=pm10_data, x='LAST_UPDATE', y='PM10', label='PM10', color='orange')
        plt.title('PM10 Trend Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_plot(fig, 'pm10_trend')
        plt.close()

def plot_distributions(pivot_df):
    """
    Create box plots and strip plots for pollutant distributions.
    
    Args:
        pivot_df (pd.DataFrame): Pivot table dataframe
    """
    pollutant_columns = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE']
    available_pollutants = [col for col in pollutant_columns if col in pivot_df.columns]

    # Box Plot
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(data=pivot_df[available_pollutants], palette='Set2')
    plt.title("Box Plot of Pollutant Distributions")
    plt.ylabel("Concentration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(fig, 'pollutant_boxplot')
    plt.close()

    # Strip Plot
    melted_df = pivot_df.melt(value_vars=available_pollutants, var_name='Pollutant', value_name='Concentration')
    fig = plt.figure(figsize=(12, 6))
    sns.stripplot(data=melted_df, x='Pollutant', y='Concentration', jitter=True, palette='Dark2', alpha=0.5)
    plt.title("Strip Plot of Individual Pollutant Values")
    plt.ylabel("Concentration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(fig, 'pollutant_stripplot')
    plt.close()

def plot_correlation(pivot_df):
    """
    Create correlation heatmap for pollutants.
    
    Args:
        pivot_df (pd.DataFrame): Pivot table dataframe
    """
    pollutant_columns = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE']
    available_pollutants = [col for col in pollutant_columns if col in pivot_df.columns]
    
    corr_matrix = pivot_df[available_pollutants].corr()
    
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Matrix of Pollutants")
    plt.tight_layout()
    save_plot(fig, 'pollutant_correlation')
    plt.close()

def plot_city_analysis(pivot_df):
    """
    Create bar plots for city-wise analysis and scatter plots for PM2.5 vs PM10.
    
    Args:
        pivot_df (pd.DataFrame): Pivot table dataframe
    """
    # Bar Plot - Top 10 Polluted Cities
    city_avg = pivot_df.groupby('CITY')['PM2.5'].mean().nlargest(10).reset_index()
    fig = plt.figure(figsize=(12, 5))
    sns.barplot(data=city_avg, x='CITY', y='PM2.5', palette='viridis')
    plt.title("Top 10 Polluted Cities (PM2.5 Levels)")
    plt.xlabel("City")
    plt.ylabel("PM2.5 Level")
    plt.xticks(rotation=30)
    plt.tight_layout()
    save_plot(fig, 'top_cities_pm25')
    plt.close()

    # Scatter Plot - PM2.5 vs PM10
    if 'PM2.5' in pivot_df.columns and 'PM10' in pivot_df.columns:
        fig = plt.figure(figsize=(8, 6))
        sns.scatterplot(data=pivot_df, x='PM2.5', y='PM10', alpha=0.6, color='green')
        plt.title("Scatter Plot: PM2.5 vs PM10")
        plt.xlabel("PM2.5")
        plt.ylabel("PM10")
        plt.tight_layout()
        save_plot(fig, 'pm25_vs_pm10')
        plt.close()

        # Scatter Plot by City
        top_cities = pivot_df['CITY'].value_counts().head(5).index
        city_filtered_df = pivot_df[pivot_df['CITY'].isin(top_cities)]
        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(data=city_filtered_df, x='PM2.5', y='PM10', hue='CITY', palette='Set2', alpha=0.7)
        plt.title("PM2.5 vs PM10 by City")
        plt.xlabel("PM2.5")
        plt.ylabel("PM10")
        plt.legend(title="City", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_plot(fig, 'pm25_vs_pm10_by_city')
        plt.close()

def categorize_aqi(pivot_df):
    """
    Categorize PM2.5 levels into AQI categories and create visualization.
    
    Args:
        pivot_df (pd.DataFrame): Pivot table dataframe
    """
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
        fig = plt.figure(figsize=(8, 5))
        sns.countplot(data=pivot_df, x='PM2.5_Level', 
                     order=['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy'], 
                     palette='Reds')
        plt.title("PM2.5 Level Categories")
        plt.ylabel("Number of Stations")
        plt.tight_layout()
        save_plot(fig, 'aqi_categories')
        plt.close()

def main():
    """Main function to run the air quality analysis."""
    # Set visual style
    sns.set_style("whitegrid")
    
    # Load and prepare data
    df_cleaned = load_and_prepare_data("AirQuality Python Dataset.csv")
    if df_cleaned is None:
        return
    
    # Create pivot table
    pivot_df = create_pivot_table(df_cleaned)
    print("\nPivoted and Cleaned Data:")
    print(pivot_df.head())
    
    # Display statistics
    print("\nDescriptive Stats:")
    print(pivot_df.describe())
    print("\nInfo:")
    print(pivot_df.info())
    
    # Create all visualizations
    plot_pollutant_trends(pivot_df)
    plot_distributions(pivot_df)
    plot_correlation(pivot_df)
    plot_city_analysis(pivot_df)
    categorize_aqi(pivot_df)

if __name__ == "__main__":
    main()