import re
import pandas as pd
import numpy as np
import os # Using os/glob is fine in Python 3 too
import glob
import argparse
import sys
from pathlib import Path # Can use pathlib again in Python 3

# --- Configuration ---
# Mapping of filename identifiers to consistent prefetcher names
PREFETCHER_MAP = {
    # Order matters slightly if keys are substrings of others (put longer first)
    'ipcp_isca2020': 'IPCP', # Example if full name is used
    'ipcp': 'IPCP',
    'ip_stride': 'IP-Stride',
    'mlop_dpc3': 'MLOP', # Example if full name is used
    'mlop': 'MLOP',
    'berti': 'Berti',
    'bo': 'Best-Offset',
    'next_line': 'Next-Line',
    'no': 'No-Prefetch' # Baseline
}

# Regex patterns to extract data from ChampSim output files
PATTERNS = {
    'ipc': re.compile(r"CPU 0 cumulative IPC:\s+(\d+\.?\d*)"),
    'instructions': re.compile(r"CPU 0 cumulative IPC:.*?instructions:\s+(\d+)"),
    # L1D Stats
    'l1d_load_miss': re.compile(r"L1D LOAD\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)"),
    'l1d_load_access': re.compile(r"L1D LOAD\s+ACCESS:\s+(\d+)"),
    'l1d_rfo_miss': re.compile(r"L1D RFO\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)"),
    'l1d_rfo_access': re.compile(r"L1D RFO\s+ACCESS:\s+(\d+)"),
    'l1d_total_miss': re.compile(r"L1D TOTAL\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)"),
    'l1d_total_access': re.compile(r"L1D TOTAL\s+ACCESS:\s+(\d+)"),
    'l1d_prefetch_issued': re.compile(r"L1D PREFETCH\s+REQUESTED:\s+\d+\s+ISSUED:\s+(\d+)"),
    'l1d_prefetch_useful': re.compile(r"L1D PREFETCH\s+REQUESTED:\s+.*?\s+USEFUL:\s+(\d+)"),
    # L2C Stats (as fallback for prefetch stats if L1D is zero)
    'l2c_prefetch_issued': re.compile(r"L2C PREFETCH\s+REQUESTED:\s+\d+\s+ISSUED:\s+(\d+)"),
    'l2c_prefetch_useful': re.compile(r"L2C PREFETCH\s+REQUESTED:\s+.*?\s+USEFUL:\s+(\d+)"),
    # LLC Stats (for bandwidth proxy)
    'llc_total_access': re.compile(r"LLC TOTAL\s+ACCESS:\s+(\d+)"),
}

# --- Helper Functions ---
def extract_metric(text, pattern, metric_name):
    """Extracts a single metric using regex, returns float or NaN."""
    match = pattern.search(text)
    if match:
        try:
            # Handle potential non-numeric values like '-nan' before converting
            value_str = match.group(1)
            if 'nan' in value_str.lower():
                 return np.nan
            return float(value_str)
        except ValueError:
            # Use f-string (Python 3.6+)
            print(f"Warning: Could not convert {metric_name} value '{match.group(1)}' to float.")
            return np.nan
    # print(f"Warning: Could not find {metric_name} in file.") # Optional: uncomment for debugging
    return np.nan

def identify_preftcher_and_benchmark(filepath, file_map):
    """Identifies prefetcher and benchmark from filepath (using pathlib)."""
    fname = filepath.name # Use pathlib's .name attribute
    identified_prefetcher = None
    identified_benchmark = None

    # --- Identify Prefetcher ---
    # Iterate through map keys (potentially ordered longer first)
    # Use a simple substring check with separators for robustness
    for key, name in file_map.items():
        # Check for key surrounded by common separators or at start/end with one separator
        pattern = r'(?:[-_.]|^)' + re.escape(key) + r'(?:[-_.]|$)'
        if re.search(pattern, fname, re.IGNORECASE): # Ignore case for flexibility
            identified_prefetcher = name
            break # Found the prefetcher

    # --- Identify Benchmark ---
    if identified_prefetcher:
        # Try specific SPEC-like pattern first, anchored before .champsimtrace.xz
        spec_match = re.search(r'(\d{3}\.[a-zA-Z0-9]+(?:_[rs])?-\d+B)\.champsimtrace\.xz', fname)
        if spec_match:
            identified_benchmark = spec_match.group(1)
        else:
            # Fallback: Try to grab the part between likely separators and before .champsimtrace.xz
            # This assumes prefetcher name isn't part of the benchmark name itself
            general_match = re.search(r'(?:---|\.\.\.|[-_])(.*?)\.champsimtrace\.xz', fname)
            if general_match:
                 potential_bench = general_match.group(1).strip('-_')
                 # Basic check to see if it looks like a benchmark name
                 if re.search(r'\d+B$', potential_bench): # Ends with NUMBER+B
                      identified_benchmark = potential_bench
                      # Clean core count prefix if present
                      identified_benchmark = re.sub(r'^\d+core[-_]+', '', identified_benchmark)


    if not identified_prefetcher or not identified_benchmark:
        print(f"Warning: Could not identify prefetcher/benchmark reliably for {fname}")
        # Return None, None if either part failed
        return None, None

    return identified_prefetcher, identified_benchmark


# --- Main Parsing Function ---
def parse_output_files(input_dir):
    """Parses all specified files in the input directory."""
    results = []
    input_path = Path(input_dir) # Use pathlib

    if not input_path.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.")
        sys.exit(1)

    print(f"Parsing files in directory: {input_path}")

    # Use pathlib's glob - adjust pattern as needed
    # Prioritize .txt, then .xz, then all files
    output_files = list(input_path.glob('*.txt'))
    file_type_msg = "Info: Found files ending in .txt."
    if not output_files:
        output_files = list(input_path.glob('*.xz'))
        file_type_msg = "Info: Found files ending in .xz, assuming these are ChampSim outputs."
        if not output_files:
            all_items = list(input_path.glob('*'))
            output_files = [f for f in all_items if f.is_file()] # Filter out directories
            file_type_msg = "Warning: Found files using '*', ensure these are ChampSim outputs."
            if not output_files:
                print(f"Error: No files found in directory '{input_dir}'. Searched for *.txt, *.xz, and *")
                sys.exit(1)

    print(file_type_msg)


    for filepath in output_files:
        # filepath is now a Path object
        print(f"Processing: {filepath.name}...")
        prefetcher_name, benchmark_name = identify_preftcher_and_benchmark(filepath, PREFETCHER_MAP)

        if not prefetcher_name or not benchmark_name:
            print(f"Skipping file due to identification failure: {filepath.name}")
            continue

        try:
            # Read with error handling for potential encoding issues
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {filepath.name}: {e}")
            continue

        # Check if file content seems valid (e.g., contains expected markers)
        if "ChampSim completed all CPUs" not in content or "Region of Interest Statistics" not in content:
             print(f"Warning: Skipping file {filepath.name} as it might not be a complete/valid ChampSim output.")
             continue


        data = {
            'Benchmark': benchmark_name,
            'Prefetcher': prefetcher_name,
            'File': filepath.name # Store filename
        }

        # Extract core metrics
        data['IPC'] = extract_metric(content, PATTERNS['ipc'], 'IPC')
        instructions = extract_metric(content, PATTERNS['instructions'], 'Instructions')
        data['Instructions'] = instructions

        # Extract L1D Misses/Accesses
        data['L1D_Load_Miss'] = extract_metric(content, PATTERNS['l1d_load_miss'], 'L1D Load Miss')
        data['L1D_Load_Access'] = extract_metric(content, PATTERNS['l1d_load_access'], 'L1D Load Access')
        data['L1D_RFO_Miss'] = extract_metric(content, PATTERNS['l1d_rfo_miss'], 'L1D RFO Miss')
        data['L1D_RFO_Access'] = extract_metric(content, PATTERNS['l1d_rfo_access'], 'L1D RFO Access')
        data['L1D_Total_Miss'] = extract_metric(content, PATTERNS['l1d_total_miss'], 'L1D Total Miss')
        data['L1D_Total_Access'] = extract_metric(content, PATTERNS['l1d_total_access'], 'L1D Total Access')


        # Extract Prefetch Stats (Prioritize L1D, fallback to L2C if L1D is 0)
        l1d_issued = extract_metric(content, PATTERNS['l1d_prefetch_issued'], 'L1D Prefetch Issued')
        l1d_useful = extract_metric(content, PATTERNS['l1d_prefetch_useful'], 'L1D Prefetch Useful')

        if l1d_issued is not np.nan and l1d_issued > 0:
             data['Prefetch_Issued'] = l1d_issued
             data['Prefetch_Useful'] = l1d_useful
             data['Prefetch_Level'] = 'L1D'
             # print(f"  Found L1D prefetch stats for {filepath.name}") # Debug print
        else:
             l2c_issued = extract_metric(content, PATTERNS['l2c_prefetch_issued'], 'L2C Prefetch Issued')
             l2c_useful = extract_metric(content, PATTERNS['l2c_prefetch_useful'], 'L2C Prefetch Useful')
             if l2c_issued is not np.nan and l2c_issued > 0:
                  data['Prefetch_Issued'] = l2c_issued
                  data['Prefetch_Useful'] = l2c_useful
                  data['Prefetch_Level'] = 'L2C'
                  # print(f"  L1D prefetch stats zero/missing, using L2C stats for {filepath.name}") # Debug print
             else:
                  data['Prefetch_Issued'] = 0.0 # Default to 0.0 float
                  data['Prefetch_Useful'] = 0.0
                  data['Prefetch_Level'] = 'None'
                  if prefetcher_name != 'No-Prefetch':
                      # print(f"  Warning: No non-zero L1D or L2C prefetch stats found for {filepath.name} (Prefetcher: {prefetcher_name})") # Debug print
                      pass # Avoid excessive warnings if pref stats are expected to be 0


        # Extract LLC Accesses
        data['LLC_Total_Access'] = extract_metric(content, PATTERNS['llc_total_access'], 'LLC Total Access')

        # Calculate L1D MPKI if possible
        # Python 3 handles float division by default
        if instructions and instructions > 0 and data['L1D_Total_Miss'] is not np.nan:
            data['L1D_MPKI'] = (data['L1D_Total_Miss'] * 1000.0) / instructions
        else:
            data['L1D_MPKI'] = np.nan

        # Calculate LLC APKI if possible
        if instructions and instructions > 0 and data['LLC_Total_Access'] is not np.nan:
             data['LLC_APKI'] = (data['LLC_Total_Access'] * 1000.0) / instructions
        else:
             data['LLC_APKI'] = np.nan


        results.append(data)

    if not results:
        print("Error: No valid data extracted from any files.")
        return None

    df = pd.DataFrame(results)

    # --- Calculate Derived Metrics (Speedup, Accuracy, Coverage) ---

    # Ensure required columns exist
    required_cols = ['Benchmark', 'Prefetcher', 'IPC', 'L1D_Total_Miss', 'Prefetch_Issued', 'Prefetch_Useful']
    if not all(col in df.columns for col in required_cols):
        print("Error: Missing required columns in DataFrame. Cannot calculate derived metrics.")
        print("Columns found:", df.columns)
        # Try to proceed without derived metrics if baseline isn't the issue
        baseline_present = 'No-Prefetch' in df['Prefetcher'].unique()
        if not baseline_present:
             print("Error: 'No-Prefetch' baseline data not found. Cannot calculate Speedup or Coverage.")
        return df # Return partially processed DF

    # Check if all benchmarks have a baseline entry
    benchmarks_in_data = df['Benchmark'].nunique()
    benchmarks_in_baseline = df[df['Prefetcher'] == 'No-Prefetch']['Benchmark'].nunique()
    if benchmarks_in_data > benchmarks_in_baseline:
         print(f"Warning: Only found 'No-Prefetch' baseline data for {benchmarks_in_baseline} out of {benchmarks_in_data} unique benchmarks.")
         print("Speedup and Coverage will be NaN for benchmarks missing a baseline.")


    # Separate baseline ('No-Prefetch') data
    baseline_df = df[df['Prefetcher'] == 'No-Prefetch'].set_index('Benchmark')
    if baseline_df.empty:
        print("Error: 'No-Prefetch' baseline data not found. Cannot calculate Speedup or Coverage.")
        # Set derived metrics to NaN if baseline is missing
        df['Baseline_IPC'] = np.nan
        df['Speedup'] = np.nan
        df['Baseline_L1D_Miss'] = np.nan
        df['Coverage'] = np.nan

    else:
        # Calculate Speedup vs. No-Prefetch
        df['Baseline_IPC'] = df['Benchmark'].map(baseline_df['IPC'])
        # Ensure baseline IPC is not NaN or zero before dividing
        df['Speedup'] = np.where(
            (df['Baseline_IPC'].notna()) & (df['Baseline_IPC'] != 0) & (df['IPC'].notna()),
            df['IPC'] / df['Baseline_IPC'], # Python 3 handles float division
            np.nan # Assign NaN if baseline IPC is missing, zero, or current IPC is missing
        )
        df.loc[df['Prefetcher'] == 'No-Prefetch', 'Speedup'] = 1.0 # Baseline speedup is 1

        # Calculate Coverage vs. No-Prefetch L1D Misses
        df['Baseline_L1D_Miss'] = df['Benchmark'].map(baseline_df['L1D_Total_Miss'])
        # Avoid division by zero for baseline misses and useful prefetches
        df['Coverage'] = np.where(
            (df['Baseline_L1D_Miss'].notna()) & (df['Baseline_L1D_Miss'] > 0) & (df['Prefetch_Useful'].notna()),
            df['Prefetch_Useful'] / df['Baseline_L1D_Miss'], # Python 3 handles float division
            0.0 # Set coverage to 0.0 float if baseline miss is 0 or NaN, or useful is NaN
        )
        df.loc[df['Prefetcher'] == 'No-Prefetch', 'Coverage'] = 0.0 # Coverage is 0.0 for baseline itself


    # Calculate Accuracy
    # Avoid division by zero if Prefetch_Issued is 0 or NaN
    df['Accuracy'] = np.where(
        (df['Prefetch_Issued'].notna()) & (df['Prefetch_Issued'] > 0) & (df['Prefetch_Useful'].notna()),
        df['Prefetch_Useful'] / df['Prefetch_Issued'], # Python 3 handles float division
        0.0 # Set accuracy to 0.0 float if issued is 0 or NaN, or useful is NaN
    )
    # For 'No-Prefetch', accuracy is meaningless, set to NaN or 0.0
    df.loc[df['Prefetcher'] == 'No-Prefetch', 'Accuracy'] = np.nan


    return df

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse ChampSim output files and generate a results CSV.")
    parser.add_argument("input_dir", help="Directory containing ChampSim output files")
    parser.add_argument("-o", "--output_csv", default="champsim_results.csv", help="Output CSV file name (default: champsim_results.csv)")
    args = parser.parse_args()

    results_df = parse_output_files(args.input_dir)

    if results_df is not None:
        # Optional: Clean up columns before saving
        # Convert relevant columns to numeric, coercing errors to NaN
        numeric_cols = ['IPC', 'Instructions', 'L1D_Load_Miss', 'L1D_Load_Access',
                        'L1D_RFO_Miss', 'L1D_RFO_Access', 'L1D_Total_Miss',
                        'L1D_Total_Access', 'Prefetch_Issued', 'Prefetch_Useful',
                        'LLC_Total_Access', 'L1D_MPKI', 'LLC_APKI',
                        'Baseline_IPC', 'Speedup', 'Baseline_L1D_Miss',
                        'Coverage', 'Accuracy']
        for col in numeric_cols:
            if col in results_df.columns:
                # Use apply with pd.to_numeric for better NaN handling if needed
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

        results_df = results_df.round(6) # Round float values for cleaner output
        try:
            results_df.to_csv(args.output_csv, index=False)
            print(f"\nResults successfully saved to {args.output_csv}")
            print("\nDataFrame Head:")
            # Use pandas default string representation
            print(results_df.head().to_string())
            print("\nDataFrame Info:")
            results_df.info(verbose=True) # Use verbose=True for more details
            print("\nPrefetcher Value Counts:")
            print(results_df['Prefetcher'].value_counts().to_string())
            print("\nBenchmarks found:")
            print(results_df['Benchmark'].unique())
            print(f"\nTotal unique benchmarks: {results_df['Benchmark'].nunique()}")


        except IOError as e:
             print(f"\nError saving results to {args.output_csv}: {e}")
        except Exception as e:
             print(f"\nAn unexpected error occurred while saving CSV: {e}")

