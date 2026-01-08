# -*- coding: utf-8 -*-

import os
# 确保脚本在正确的工作目录下运行（根据你的原逻辑保留）
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

    # 定义训练脚本名和对应的测试脚本名列表
    train_py_scripts = ['main_train.py', 'main_traintest_5fold.py', 'main_traintest.py']
    test_py_scripts = ['main_test.py', 'main_test_5fold.py', 'main_test.py']

    # 1. 检查文件中是否包含任何一个已知的训练脚本名
    found_match = False
    for train_s in train_py_scripts:
        if train_s in content:
            found_match = True
            break

    if not found_match:
        # 如果都不存在，打印跳过信息
        print(f"🟡 No matching training script name ({train_py_scripts}) found in {train_script_path}. Skipping file.")
        return None, None

    # 2. 执行核心替换逻辑：遍历列表一一对应替换
    test_content = content
    for train_s, test_s in zip(train_py_scripts, test_py_scripts):
        # 比如：先替换 main_train.py -> main_test.py
        # 再替换 main_traintest_5fold.py -> main_test_5fold.py
        test_content = test_content.replace(train_s, test_s)

    # 生成新的测试脚本文件名
    if "train" in filename:
        test_filename = filename.replace("train", "test", 1)
    else:
        test_filename = "test_" + filename
    
    return test_content, test_filename

def generate_scripts_in_directory(source_dir, target_dir):
    """
    遍历指定目录，找到所有训练脚本，并在目标目录中生成相应的测试脚本，同时保持目录结构。
    """
    print(f"🔍 Starting to scan directory: {source_dir}")
    print(f"🎯 Generated scripts will be stored in: {target_dir}\n")
    generated_count = 0
    
    # 使用 os.walk 递归遍历所有子目录
    for root, _, files in os.walk(source_dir):
        # print("Root = ", root) # 可选：减少打印杂乱信息
        for filename in files:
            # 筛选出 .sh 脚本
            if filename.endswith(".sh"):
                train_script_path = os.path.join(root, filename)
                
                # 调用转换函数
                test_content, test_filename = convert_train_script_to_test(train_script_path)
                
                # 如果转换成功（即找到了匹配的内容并生成了新内容）
                if test_content and test_filename:
                    test_script_path = None # 初始化以防报错
                    try:
                        # 计算相对路径以在目标文件夹中复制目录结构
                        relative_path = os.path.relpath(root, source_dir)
                        final_target_dir = os.path.join(target_dir, relative_path)
                        
                        # 创建目标子目录（如果不存在）
                        os.makedirs(final_target_dir, exist_ok=True)
                        
                        test_script_path = os.path.join(final_target_dir, test_filename)

                        with open(test_script_path, 'w', encoding='utf-8') as f:
                            f.write(test_content)
                        
                        # 复制原始文件权限，并确保新文件可执行
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
    解析命令行参数并启动脚本生成过程的主函数。
    """
    parser = argparse.ArgumentParser(
        description="Automatically convert training scripts (train_script) to test scripts (test_script) and store them in the specified directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 根据你的项目结构设置默认路径
    default_source_path = "../TrainScripts-Kimi"
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

    # 自动创建目标目录（如果不存在）
    if not os.path.isdir(args.target_dir):
        print(f"Note: Target directory '{args.target_dir}' does not exist, it will be created automatically.")
        os.makedirs(args.target_dir, exist_ok=True)

    generate_scripts_in_directory(args.source_dir, args.target_dir)

if __name__ == "__main__":
    main()