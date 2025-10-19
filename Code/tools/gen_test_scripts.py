# -*- coding: utf-8 -*-

import os
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import stat

def convert_train_script_to_test(train_script_path):
    directory, filename = os.path.split(train_script_path)

    if not filename.endswith(".sh"):
        return None, None
        
    try:
        with open(train_script_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading file {train_script_path}: {e}")
        return None, None


    train_py_script = 'main_train.py'
    test_py_script = 'main_test.py'


    if train_py_script not in content:
        print(f"🟡 '{os.path.basename(train_py_script)}' not found in {train_script_path}. Skipping file.")
        return None, None

    # Execute core replacement logic
    test_content = content.replace(train_py_script, test_py_script)

    # Generate new test script filename
    if "train" in filename:
        test_filename = filename.replace("train", "test", 1)
    else:
        test_filename = "test_" + filename
    
    return test_content, test_filename

def generate_scripts_in_directory(source_dir, target_dir):
    """
    Traverse the specified directory, find all training scripts, and generate corresponding test scripts in the target directory while maintaining the directory structure.
    
    Args:
        source_dir (str): The root directory containing the training scripts.
        target_dir (str): The target directory for storing the generated test scripts.
    """
    print(f"🔍 Starting to scan directory: {source_dir}")
    print(f"🎯 Generated scripts will be stored in: {target_dir}\n")
    generated_count = 0
    
    # Use os.walk to recursively traverse all subdirectories
    for root, _, files in os.walk(source_dir):
        print("Root = ", root)
        for filename in files:
            # Filter out eligible training scripts
            if filename.endswith(".sh"):
                train_script_path = os.path.join(root, filename)
                
                # Call the conversion function
                test_content, test_filename = convert_train_script_to_test(train_script_path)
                
                # If conversion is successful
                if test_content and test_filename:
                    test_script_path = None # Initialize for error message
                    try:
                        # Calculate relative path to replicate source directory structure in target
                        relative_path = os.path.relpath(root, source_dir)
                        final_target_dir = os.path.join(target_dir, relative_path)
                        
                        # Create target subdirectory (if it doesn't exist)
                        os.makedirs(final_target_dir, exist_ok=True)
                        
                        test_script_path = os.path.join(final_target_dir, test_filename)

                        with open(test_script_path, 'w', encoding='utf-8') as f:
                            f.write(test_content)
                        
                        # Copy original file permissions and ensure the new file is executable
                        original_permissions = os.stat(train_script_path).st_mode
                        os.chmod(test_script_path, original_permissions | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
                        
                        print(f"✅ Successfully created test script: {test_script_path}")
                        generated_count += 1
                        
                    except Exception as e:
                        error_path = test_script_path if test_script_path else "unknown path"
                        print(f"❌ Error writing file {error_path}: {e}")
                    print("-" * 20)

    print(f"\n🎉 Script generation completed. A total of {generated_count} test scripts were generated.")

def main():
    """
    Main function for parsing command-line arguments and starting the script generation process.
    """
    parser = argparse.ArgumentParser(
        description="Automatically convert training scripts (train_script) to test scripts (test_script) and store them in the specified directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Set default paths based on your project structure
    default_source_path = "../TrainScripts"
    default_target_path = "../TestScripts"
    
    parser.add_argument(
        "--source_dir",
        type=str,
        default=default_source_path,
        help=f"Source directory containing training scripts.\nDefault path: {default_source_path}"
    )

    parser.add_argument(
        "--target_dir",
        type=str,
        default=default_target_path,
        help=f"Target directory for storing generated test scripts.\nDefault path: {default_target_path}"
    )
    
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print(f"Error: Source directory '{args.source_dir}' does not exist. Please provide a valid directory.")
        return

    # Automatically create target directory if it doesn't exist
    if not os.path.isdir(args.target_dir):
        print(f"Note: Target directory '{args.target_dir}' does not exist, it will be created automatically.")
        os.makedirs(args.target_dir, exist_ok=True)

    generate_scripts_in_directory(args.source_dir, args.target_dir)

if __name__ == "__main__":
    main()