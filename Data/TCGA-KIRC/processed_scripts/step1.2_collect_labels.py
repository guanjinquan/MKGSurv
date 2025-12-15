import os
import pandas as pd
import sys
import pickle
import string

# --- Treatment Category Definitions ---
# Based on the user-provided JSON structure
TREATMENT_CATEGORIES = [
  {
    "category_name": "Radiation Therapy",
    "sub_types": [
      "Radiation Therapy, NOS",
      "Radiation, External Beam",
      "Radiation, Cyberknife",
      "Radiation, Internal",
      "Radiation, Implants",
      "Radiation, Stereotactic/Gamma Knife/SRS",
      "Brachytherapy, Low Dose",
      "Radiation, Intensity-Modulated Radiotherapy",
      "Radiation, Systemic",
    ]
  },
  {
    "category_name": "Pharmaceutical Therapy",
    "sub_types": ["Pharmaceutical Therapy, NOS",
    ]
  },
  {
    "category_name": "Chemotherapy",
    "sub_types": [
      "Chemotherapy",
      "Targeted Molecular Therapy",
      "Immunotherapy (Including Vaccines)",
      "Hormone Therapy",
    ]
  },
  {
    "category_name": "Surgery",
    "sub_types": ["Surgery, NOS",
    ]
  },
  {
    "category_name": "Other",
    "sub_types": ["'--", "Ancillary Treatment", "Unknown",
    ]
  }
]

# --- Column Definitions ---

# Key columns for patient identification
KEY_CASE_ID = 'cases.case_id'
KEY_SUBMITTER_ID = 'cases.submitter_id'

# Columns to determine Disease-Free Survival (DFS)
# DFS_RECURRENCE_COL = 'follow_ups.days_to_recurrence' # 'diagnoses.days_to_recurrence'
DFS_DEATH_COL = 'demographic.days_to_death'
DFS_FOLLOW_UP_COL = 'diagnoses.days_to_last_follow_up'
Treatment_COL = 'treatments.treatment_type' # This will contain SUB-TYPES initially
Treatment_ID_COL = '5_classes' # This will contain SUB-TYPES initially

# --- File Paths ---
# Ensure these paths are correct for your environment
BASE_PATH = '/home/Zhengzx/MedAlignFusion/Data/TCGA-KIRC'
SOURCE_PATH = os.path.join(BASE_PATH, 'source')
PROCESSED_PATH = os.path.join(BASE_PATH, 'processed')

# Input data directories
BIOSPECIMEN_DIR = os.path.join(SOURCE_PATH, 'biospecimen.project-tcga-kirc.2025-12-11')
CLINICAL_DIR = os.path.join(SOURCE_PATH, 'clinical.project-tcga-kirc.2025-12-11')

# Input patient list
PATIENT_LIST_PKL = os.path.join(SOURCE_PATH, 'kirc_patients.pkl')

# Output file
OUTPUT_CSV = os.path.join(PROCESSED_PATH, 'kirc_patient_labels.csv')
# OUTPUT_ID_LIST = os.path.join(PROCESSED_PATH, 'lusc_final_patient_ids.txt')


def build_id_map(all_files):
    """
    Scans all TSV files to build a comprehensive map between
    case_id and submitter_id.
    """
    print("Building ID map...")
    case_to_submitter = {}
    submitter_to_case = {}
    
    for file_path in all_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping for ID map: {file_path}")
            continue
        try:
            df_ids = pd.read_csv(
                file_path,
                sep='\t',
                header=0,
                low_memory=False,
                usecols=lambda c: c in [KEY_CASE_ID, KEY_SUBMITTER_ID]
            )
            
            if KEY_CASE_ID in df_ids.columns and KEY_SUBMITTER_ID in df_ids.columns:
                id_pairs = df_ids[[KEY_CASE_ID, KEY_SUBMITTER_ID]].dropna().drop_duplicates().values
                for case, submitter in id_pairs:
                    if case not in case_to_submitter:
                        case_to_submitter[case] = submitter
                    if submitter not in submitter_to_case:
                        submitter_to_case[submitter] = case
                        
        except Exception as e:
            print(f"Error reading {file_path} for ID mapping: {e}")
            
    print(f"ID map built: {len(case_to_submitter)} entries.")
    return case_to_submitter, submitter_to_case


def load_and_merge_data(all_files, id_maps):
    """
    Loads all TSV files, ensures both ID keys are present,
    and concatenates them into a single DataFrame.
    """
    case_to_submitter, submitter_to_case = id_maps
    dfs_to_merge = []
    
    print("Loading and merging files...")
    for file_path in all_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping merge: {file_path}")
            continue
        try:
            df = pd.read_csv(
                file_path, 
                sep='\t', 
                header=0, 
                low_memory=False,
                na_values=['--', "'--", "Not Reported", "Unknown", "."] # 在这里添加所有你想转为NaN的怪异字符
            )
            
            # Fix missing IDs using the map
            if KEY_CASE_ID in df.columns and KEY_SUBMITTER_ID not in df.columns:
                df[KEY_SUBMITTER_ID] = df[KEY_CASE_ID].map(case_to_submitter)
            elif KEY_CASE_ID not in df.columns and KEY_SUBMITTER_ID in df.columns:
                df[KEY_CASE_ID] = df[KEY_SUBMITTER_ID].map(submitter_to_case)

            # Only add if it has the keys and some relevant data
            if KEY_CASE_ID in df.columns and KEY_SUBMITTER_ID in df.columns:
                dfs_to_merge.append(df)
            else:
                print(f"Warning: Skipping {file_path}, could not find or map both index keys.")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not dfs_to_merge:
        print("No data was loaded. Exiting.")
        return None

    print("Concatenating all DataFrames...")
    try:
        merged_df = pd.concat(dfs_to_merge, ignore_index=True, sort=False)
        print(f"Combined DataFrame shape: {merged_df.shape}")
        # Drop rows that are exact duplicates
        merged_df.drop_duplicates(inplace=True)
        print(f"Shape after dropping duplicates: {merged_df.shape}")
        return merged_df
    except Exception as e:
        print(f"Error during concatenation: {e}")
        return None


def join_unique_strings(series):
    """
    Custom aggregation function to join unique non-null string values with '+'
    Filters out strings that are empty or contain only punctuation
    """
    # Drop null/NaN values
    non_null_series = series.dropna()
    if non_null_series.empty:
        return pd.NA
    
    # Get unique values as strings and filter
    filtered_strings = set()
    for s in non_null_series.astype(str):
        stripped_s = s.strip()  # Remove leading/trailing whitespace
        # Keep if not empty and not all punctuation
        if stripped_s and not all(char in string.punctuation for char in stripped_s):
            filtered_strings.add(stripped_s)
    
    if not filtered_strings:
        return pd.NA
    
    # Sort and join
    sorted_unique_strings = sorted(list(filtered_strings))
    return '+'.join(sorted_unique_strings)


def build_treatment_maps():
    """
    Builds mapping dictionaries from the TREATMENT_CATEGORIES constant.
    1. sub_type -> category_name
    2. category_name -> 0-based-index
    """
    print("Building treatment category maps...")
    sub_type_to_category_map = {}
    category_to_index_map = {}
    
    for idx, category_data in enumerate(TREATMENT_CATEGORIES):
        category_name = category_data["category_name"]
        
        # Map category name to its index
        category_to_index_map[category_name] = idx
        
        # Map all sub-types to this category name
        for sub_type in category_data["sub_types"]:
            sub_type_to_category_map[sub_type] = category_name
            
    # Add a fallback for "Unknown" just in case it's not in the list
    if "Unknown" not in sub_type_to_category_map:
        sub_type_to_category_map["Unknown"] = "Other"
    if "Other" not in category_to_index_map:
        # This should not happen if "Other" is in TREATMENT_CATEGORIES
        # Find the "Other" category index
        other_idx = next((i for i, cat in enumerate(TREATMENT_CATEGORIES) if cat["category_name"] == "Other"), None)
        if other_idx is not None:
             category_to_index_map["Other"] = other_idx
        else:
            # Fallback if "Other" is missing entirely
             category_to_index_map["Other"] = len(TREATMENT_CATEGORIES) -1

    print("Treatment maps built.")
    return sub_type_to_category_map, category_to_index_map



def process_treatment_types(treatment_sub_types_str, sub_type_map, category_index_map):
    """
    Maps a '+' separated string of treatment sub-types to:
    1. A '+' separated string of unique, sorted category NAMES.
    2. A ',' separated string of unique, sorted category 0-INDEXES.
    
    Handles pd.NA, None, or empty strings by mapping to "Unknown" (class 4).
    """
    # Handle missing data
    if pd.isna(treatment_sub_types_str) or not treatment_sub_types_str.strip():
        unknown_category_name = "Unknown"
        # Find "Unknown" sub-type's category, default to "Other"
        category_name = sub_type_map.get(unknown_category_name, "Other")
        category_index = category_index_map.get(category_name, 4)  # Default to 4
        return category_name, str(category_index)

    sub_types = treatment_sub_types_str.split('+')
    
    found_categories = set()
    found_indices = set()
    
    for sub_type in sub_types:
        # Find the category for the sub-type
        # Default to "Other" if a sub-type is not in our map
        category_name = sub_type_map.get(sub_type, "Other")
        
        # Find the index for that category
        # Default to 4 (index of "Other") if not in map
        category_index = category_index_map.get(category_name, 4) 
        
        found_categories.add(category_name)
        found_indices.add(category_index)
        
    # Sort and join
    sorted_category_names = sorted(list(found_categories))
    sorted_category_indices = sorted(list(found_indices))
    
    # Create final strings
    final_category_str = '+'.join(sorted_category_names)
    final_indices_str = ','.join(map(str, sorted_category_indices))
    
    return final_category_str, final_indices_str

def main():
    # --- 0. Setup ---
    # Try to change to script directory (robust for different execution contexts)
    try:
        script_dir = os.path.dirname(__file__)
        if script_dir: # Only change if __file__ is defined
             os.chdir(os.path.join(script_dir, '..'))
        print(f"Changed working directory to: {os.getcwd()}")
    except NameError:
        print(f"Running in an environment where __file__ is not defined. Using current dir: {os.getcwd()}")
    except Exception as e:
        print(f"Could not change directory: {e}. Using current dir: {os.getcwd()}")

    # --- 1. Find all files ---
    all_files = []
    for dir_path in [BIOSPECIMEN_DIR, CLINICAL_DIR]:
        if not os.path.exists(dir_path):
            print(f"Warning: Directory not found: {dir_path}")
            continue
        for filename in os.listdir(dir_path):
            if filename.endswith('.tsv'):
                all_files.append(os.path.join(dir_path, filename))
    
    if not all_files:
        print("Error: No .tsv files found. Check paths.")
        print(f"Checked: {BIOSPECIMEN_DIR} and {CLINICAL_DIR}")
        return

    # --- 2. Build ID Map ---
    id_maps = build_id_map(all_files)
    
    # --- 3. Build Treatment Maps ---
    sub_type_to_category_map, category_to_index_map = build_treatment_maps()
    
    # --- 4. Load and Merge Data ---
    merged_df = load_and_merge_data(all_files, id_maps)
    if merged_df is None:
        return

    # --- 5. Load Patient List ---
    if not os.path.exists(PATIENT_LIST_PKL):
        print(f"Error: Patient list not found at {PATIENT_LIST_PKL}")
        return
        
    print(f"Loading patient list from {PATIENT_LIST_PKL}...")
    with open(PATIENT_LIST_PKL, 'rb') as f:
        patient_list = pickle.load(f)
    
    initial_patient_set = set(patient_list)
    print(f"Loaded {len(initial_patient_set)} patient IDs.")
    
    # --- 6. Filter Merged Data by Patient List ---
    initial_rows = len(merged_df)
    merged_df = merged_df[merged_df[KEY_SUBMITTER_ID].isin(patient_list)]
    print(f"Filtered merged data to {len(merged_df)} rows for {len(patient_list)} patients (from {initial_rows}).")
    
    if merged_df.empty:
        print("Error: No data remaining after filtering by patient list.")
        return

    # --- 7. Prepare Data for Aggregation ---
    print("Preparing data for aggregation...")
    
    # Convert key numeric columns to numeric, coercing errors
    cols_to_convert = [DFS_DEATH_COL, DFS_FOLLOW_UP_COL]
    for col in cols_to_convert:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    # --- 8. Aggregate Data Per Patient ---
    print("Aggregating data per patient...")
    
    # We only need these columns for final output
    required_columns = [KEY_CASE_ID, KEY_SUBMITTER_ID, Treatment_COL, 
                        DFS_DEATH_COL, DFS_FOLLOW_UP_COL]
    
    # Check which required columns exist in the data
    available_columns = [col for col in required_columns if col in merged_df.columns]
    print(f"Available columns for aggregation: {available_columns}")
    
    if len(available_columns) < 3:  # At minimum we need the ID columns
        print("Error: Not enough required columns found for aggregation.")
        return
    
    # Select only the columns we need
    df_to_aggregate = merged_df[available_columns].copy()
    
    # Define aggregation rules
    agg_rules = {}
    
    # For submitter_id, take first (should be consistent per case_id)
    if KEY_SUBMITTER_ID in df_to_aggregate.columns:
        agg_rules[KEY_SUBMITTER_ID] = 'first'
    
    # For DFS time columns, take minimum
    for col in [DFS_DEATH_COL, DFS_FOLLOW_UP_COL]:
        if col in df_to_aggregate.columns:
            agg_rules[col] = 'max'
    
    # For treatment_type, use custom string joining function
    if Treatment_COL in df_to_aggregate.columns:
        agg_rules[Treatment_COL] = join_unique_strings
    
    # Group by case_id and apply aggregation
    agg_df = df_to_aggregate.groupby(KEY_CASE_ID).agg(agg_rules).reset_index()
    print(f"Aggregated data to {len(agg_df)} unique patients.")

    # --- 9. Map Treatments to Categories and 5_classes ---
    print(f"Mapping {Treatment_COL} to categories and Treatment_ID_COL...")
    if Treatment_COL in agg_df.columns:
        # Apply the function to get a Series of tuples
        result_tuples = agg_df[Treatment_COL].apply(
            lambda x: process_treatment_types(x, sub_type_to_category_map, category_to_index_map)
        )
        # Split tuples into two new columns
        # Overwrite Treatment_COL with category names
        agg_df[Treatment_COL] = result_tuples.str[0]
        # Create new 5_classes column
        agg_df[Treatment_ID_COL] = result_tuples.str[1]
        print("Mapping complete.")
    else:
        # If no treatment column exists, default all to "Unknown" / class 4
        print(f"Warning: {Treatment_COL} not in aggregated data. Defaulting to 'Unknown' / '4'.")
        agg_df[Treatment_COL] = "Other"
        agg_df[Treatment_ID_COL] = "4"

    # --- 10. Calculate DFS Time and Event ---
    print("Calculating DFS_time and DFS_event...")
    
    # Get the minimum time from the three key columns
    time_columns = [col for col in [DFS_DEATH_COL, DFS_FOLLOW_UP_COL] if col in agg_df.columns]
    if not time_columns:
        print("Error: None of the key survival time columns are in the aggregated data.")
        return
        
    agg_df['DFS_time'] = agg_df[time_columns].min(axis=1, skipna=True)
    agg_df['DFS_event'] = agg_df[DFS_DEATH_COL].notnull().astype(int)

    # --- 11. Final Check and Filter ---
    print("Checking for missing labels...")
    initial_patient_count = len(agg_df)
    missing_time = agg_df['DFS_time'].isnull()
    num_missing = missing_time.sum()
    
    if num_missing > 0:
        print(f"Warning: {num_missing} patients have no time data (DFS_time is NaN). Removing them.")
        agg_df = agg_df[~missing_time].copy()
    
    print(f"Final patient count with labels: {len(agg_df)} (out of {initial_patient_count}).")

    # --- 12. Create Final Output with Only Required Columns ---
    print("Creating final output with only required columns...")
    
    # Define final columns we want to keep
    # Add Treatment_ID_COL to this list
    final_columns = [KEY_CASE_ID, KEY_SUBMITTER_ID, 'DFS_time', 'DFS_event', Treatment_ID_COL]
    
    if Treatment_COL in agg_df.columns:
        final_columns.append(Treatment_COL)
    
    # Select only the final columns
    final_df = agg_df[final_columns].copy()
    
    # --- 13. Save Output ---
    print(f"Saving final processed data to {OUTPUT_CSV}...")
    try:
        # Ensure output directory exists
        os.makedirs(PROCESSED_PATH, exist_ok=True)
        
        final_df.to_csv(OUTPUT_CSV, index=False)
        print("Process complete.")
        print(f"\nFinal DataFrame shape: {final_df.shape}")
        print("\nColumns in final output:")
        print(final_df.columns.tolist())
        print("\nExample rows:")
        print(final_df.head())


        # Compare initial vs final patient IDs
        final_ids = final_df[KEY_SUBMITTER_ID].tolist()
     
        print("\n--- Patient ID Comparison ---")
        print(f"Initial patient list ({PATIENT_LIST_PKL}): {len(initial_patient_set)} IDs")
        print(f"Final patient list (in CSV): {len(final_ids)} IDs")
        
        final_ids_set = set(final_ids)
        missing_ids = initial_patient_set - final_ids_set
        
        if missing_ids:
            print(f"Found {len(missing_ids)} patient IDs that were in '{PATIENT_LIST_PKL}' but dropped (due to missing time data):")
            # Print only first 10 missing IDs for brevity
            for i, patient_id in enumerate(sorted(list(missing_ids))[:10]):
                print(f"  {i+1}: {patient_id}")
            if len(missing_ids) > 10:
                print(f"  ... and {len(missing_ids) - 10} more.")
        else:
            print("All initial patient IDs from the pickle file (that had time data) are present in the final CSV.")
        
        print("---------------------------------")

    except Exception as e:
        print(f"Error saving files: {e}")


if __name__ == '__main__':
    main()