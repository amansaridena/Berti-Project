import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gmean # For geometric mean calculation
import argparse
import os

# --- Configuration ---
PLOT_STYLE = 'seaborn-v0_8-whitegrid' # Use a seaborn style for plots
OUTPUT_DIR = "plots" # Directory to save plots

# Define the desired order for prefetchers in plots
PREFETCHER_ORDER = [
    'No-Prefetch',
    'Next-Line',
    'IP-Stride',
    'MLOP',
    'Best-Offset',
    'IPCP',
    'Berti'
]

# --- Plotting Functions ---

def plot_metric_comparison(df, metric, title, ylabel, is_speedup=False, output_dir=OUTPUT_DIR):
    """Generates a bar plot comparing the average (or gmean) of a metric across prefetchers."""
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate mean or geometric mean
    if is_speedup:
        # Geometric mean for speedup, handle potential errors if values <= 0 exist
        # Filter out non-positive speedups before calculating gmean
        valid_speedups = df[df[metric] > 0]
        if valid_speedups.empty and not df[metric].isnull().all():
             print(f"Warning: No positive values found for {metric}, cannot calculate geometric mean.")
             avg_data = df.groupby('Prefetcher')[metric].mean().reindex(PREFETCHER_ORDER).reset_index()
             plot_title = f"Average {title} (Arithmetic Mean used due to non-positive values)"
        elif df[metric].isnull().all():
             print(f"Warning: All values for {metric} are NaN, cannot plot.")
             return # Skip plotting if all data is NaN
        else:
             # Calculate gmean only for groups with valid data
             avg_data = valid_speedups.groupby('Prefetcher')[metric].apply(
                 lambda x: gmean(x) if len(x) > 0 and np.all(x > 0) else np.nan
             ).reindex(PREFETCHER_ORDER).reset_index()
             plot_title = f"Average {title} (Geometric Mean)"
    else:
        # Arithmetic mean for other metrics
        avg_data = df.groupby('Prefetcher')[metric].mean().reindex(PREFETCHER_ORDER).reset_index()
        plot_title = f"Average {title} (Arithmetic Mean)"

    # Filter out prefetchers not present in the data after reindexing
    avg_data = avg_data.dropna(subset=[metric])

    if avg_data.empty:
         print(f"Warning: No data to plot for {metric} after filtering.")
         return

    sns.barplot(x='Prefetcher', y=metric, data=avg_data, ax=ax, palette='viridis', order=PREFETCHER_ORDER)

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')

    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel("Prefetcher", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    # Save plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"{metric}_comparison.png")
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close(fig) # Close the figure to free memory

# --- Main Analysis Function ---

def analyze_and_plot(csv_filepath):
    """Loads data from CSV and generates standard analysis plots."""
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Input CSV file '{csv_filepath}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file '{csv_filepath}': {e}")
        return

    print(f"Loaded data from {csv_filepath}")
    print("Data Summary:")
    print(df.info())
    print("\nPrefetcher Counts:")
    print(df['Prefetcher'].value_counts())

    # --- Generate Plots ---

    # 1. IPC Speedup (Geometric Mean)
    if 'Speedup' in df.columns:
        # Exclude baseline for speedup plot (it's always 1)
        plot_metric_comparison(df[df['Prefetcher'] != 'No-Prefetch'], 'Speedup', 'IPC Speedup vs. No-Prefetch', 'Geometric Mean Speedup', is_speedup=True)
    else:
        print("Warning: 'Speedup' column not found. Skipping IPC Speedup plot.")

    # 2. L1D MPKI (Arithmetic Mean)
    if 'L1D_MPKI' in df.columns:
        plot_metric_comparison(df, 'L1D_MPKI', 'L1D Miss Rate', 'Average L1D MPKI')
    else:
        print("Warning: 'L1D_MPKI' column not found. Skipping L1D MPKI plot.")

    # 3. Prefetch Accuracy (Arithmetic Mean)
    if 'Accuracy' in df.columns:
        # Exclude baseline for accuracy plot (it's NaN)
        plot_metric_comparison(df[df['Prefetcher'] != 'No-Prefetch'], 'Accuracy', 'Prefetch Accuracy', 'Average Accuracy')
    else:
        print("Warning: 'Accuracy' column not found. Skipping Accuracy plot.")

    # 4. Prefetch Coverage (Arithmetic Mean)
    if 'Coverage' in df.columns:
         # Exclude baseline for coverage plot (it's 0)
        plot_metric_comparison(df[df['Prefetcher'] != 'No-Prefetch'], 'Coverage', 'Prefetch Coverage', 'Average Coverage')
    else:
        print("Warning: 'Coverage' column not found. Skipping Coverage plot.")

    # 5. LLC APKI (Arithmetic Mean - Bandwidth Proxy)
    if 'LLC_APKI' in df.columns:
        plot_metric_comparison(df, 'LLC_APKI', 'LLC Access Rate (Bandwidth Proxy)', 'Average LLC APKI')
    else:
        print("Warning: 'LLC_APKI' column not found. Skipping LLC APKI plot.")

    # Optional: Add scatter plots for trade-offs here if desired
    # e.g., plot_tradeoff(df, x_metric='Accuracy', y_metric='Speedup', title='Speedup vs. Accuracy')

    print("\nAnalysis and plotting complete.")
    print(f"Plots saved in directory: {OUTPUT_DIR}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze processed ChampSim results and generate plots.")
    parser.add_argument("input_csv", help="Path to the CSV file generated by parse_champsim.py")
    parser.add_argument("-p", "--plot_dir", default=OUTPUT_DIR, help=f"Directory to save plots (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    analyze_and_plot(args.input_csv)
