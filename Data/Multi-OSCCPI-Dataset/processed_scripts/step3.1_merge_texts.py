import pandas as pd
import json
import os

def merge_medical_data():
    # 1. File Paths (As specified in the prompt)
    csv_path = '/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/multimodal_texts.csv'
    json_path = '/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/medical_analysis_deepseek.json'
    output_path = 'LLM_output_analysis.xlsx'

    print("Loading files...")

    # 2. Load CSV Data
    try:
        # Load with pandas
        # First try tab separator as originally requested
        df = pd.read_csv(csv_path, sep='\t', dtype=str, encoding='utf-8-sig')
        
        # Check if parsing failed (everything in one column)
        if len(df.columns) <= 1:
            print(f"Warning: Only 1 column found with tab separator: {df.columns.tolist()}")
            print("Retrying with comma separator...")
            df = pd.read_csv(csv_path, sep=',', dtype=str, encoding='utf-8-sig')

        # Clean column names (strip whitespace and potential quotes)
        df.columns = df.columns.str.strip().str.replace('"', '')
        print(f"Columns found: {df.columns.tolist()}")

        # Check if PID is in columns
        if 'PID' not in df.columns:
            print("'PID' column not explicitly found. Checking alternatives...")
            
            # Strategy 1: Check if PID is the index name
            if df.index.name and df.index.name.strip().upper() == 'PID':
                print("Found 'PID' as index. Resetting index...")
                df.reset_index(inplace=True)
            
            # Strategy 2: The file might have been read such that the first column is the index
            # We try resetting the index blindly to see if a new column appears
            else:
                print("Attempting to reset index to recover potential ID column...")
                df.reset_index(inplace=True)
                # After reset, the old index becomes the first column (often named 'index' or 'level_0')
                # Let's rename the very first column to PID if we still don't have it
                df.columns = df.columns.str.strip() # Clean again
                
            # Strategy 3: Force rename the first column to PID if still missing
            if 'PID' not in df.columns:
                first_col = df.columns[0]
                print(f"Still no 'PID'. Assuming the first column '{first_col}' is the PID. Renaming...")
                df.rename(columns={first_col: 'PID'}, inplace=True)

        # Final Verification
        if 'PID' not in df.columns:
            print("Critical Error: Could not locate or create a 'PID' column. Aborting.")
            return

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Load JSON Data
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    # 4. Initialize new columns with empty strings
    df['clinical + pathology'] = ""
    df['clinical + treatment'] = ""
    df['pathology + treatment'] = ""

    print("Merging data...")

    # 5. Iterate through DataFrame and merge JSON data
    for index, row in df.iterrows():
        # Use str() to ensure it matches JSON keys safely
        pid = str(row['PID']).strip()
        
        # Check if this PID exists in the JSON data
        if pid in analysis_data:
            analysis_entries = analysis_data[pid]
            
            # Iterate through the list of analyses for this PID
            for entry in analysis_entries:
                pairs = entry.get('modalPairs', [])
                
                # Extract text
                relationship = entry.get('relationship', '')
                survival = entry.get('survival', '')
                
                # Combine relationship and survival
                combined_text = f"{relationship} {survival}".strip()
                
                # Determine which column to place the data in
                if 'clinical' in pairs and 'pathology' in pairs:
                    df.at[index, 'clinical + pathology'] = combined_text
                    
                elif 'clinical' in pairs and 'treatment' in pairs:
                    df.at[index, 'clinical + treatment'] = combined_text
                    
                elif 'pathology' in pairs and 'treatment' in pairs:
                    df.at[index, 'pathology + treatment'] = combined_text

    # 6. Reorder columns
    desired_columns = [
        'PID', 
        'clinical', 
        'treatment', 
        'pathology', 
        'clinical + pathology', 
        'clinical + treatment', 
        'pathology + treatment'
    ]
    
    # Filter/Reorder columns (intersection to be safe)
    final_cols = [c for c in desired_columns if c in df.columns]
    df_final = df[final_cols]

    # 7. Save to Excel
    print(f"Saving to {output_path}...")
    try:
        df_final.to_excel(output_path, index=False)
        print("Success! File generated.")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

if __name__ == "__main__":
    merge_medical_data()