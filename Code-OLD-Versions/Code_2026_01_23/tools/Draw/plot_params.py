import matplotlib.pyplot as plt
import numpy as np
import os

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
# 支持中文显示（如果环境不支持 SimHei，请替换为 Arial 或 sans-serif）
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

def draw_and_save_plot(x_data, data_dict, mean_data, x_label, title, filename):
    """
    绘制并保存单个超参数变化曲线图
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.figure(figsize=(6, 5))
    
    datasets = ['LUAD', 'LUSC', 'BRCA', 'KIRC', 'Inhouse']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. 绘制每个数据集的虚线
    for idx, ds_name in enumerate(datasets):
        y_vals = data_dict[ds_name]
        plt.plot(x_data, y_vals, linestyle='--', marker='o', 
                markersize=5, alpha=0.5, color=colors[idx], label=ds_name)
    
    # 2. 绘制 Mean 曲线 (黑线改为 2 磅，并添加菱形点)
    plt.plot(x_data, mean_data, linestyle='-', linewidth=2, marker='d', 
            markersize=8, color='black', label='Mean', zorder=5)

    # 3. 找到 Mean 的最高点并打星
    max_idx = np.argmax(mean_data)
    max_x = x_data[max_idx]
    max_y = mean_data[max_idx]
    
    plt.scatter(max_x, max_y, color='red', marker='*', s=350, zorder=10, label='Best Mean')
    
    # 在星星上方标注具体数值
    plt.annotate(f'{max_y:.4f}', xy=(max_x, max_y), xytext=(0, 15), 
                textcoords='offset points', ha='center', fontweight='bold', 
                color='red', fontsize=11)

    # 设置标签和标题
    plt.xlabel(x_label, fontsize=13)
    plt.ylabel('Performance (AUC/Metric)', fontsize=13)
    # plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # 保存为 SVG 格式
    plt.tight_layout()
    plt.savefig(filename, format='svg', bbox_inches='tight')
    print(f"成功保存图表至: {filename}")
    plt.close()

def main():
    # ==========================================
    # 数据准备：表格 1 (Num Layers 变化, Loss Weight=1)
    # ==========================================
    x_layers = [1, 2, 3, 4, 5]
    layers_data = {
        'LUAD':    [0.6636, 0.6676, 0.6744, 0.6733, 0.6729],
        'LUSC':    [0.6638, 0.6637, 0.6602, 0.6570, 0.6563],
        'BRCA':    [0.7026, 0.7212, 0.7323, 0.7190, 0.7052],
        'KIRC':    [0.7829, 0.7855, 0.7959, 0.7937, 0.7971],
        'Inhouse': [0.8166, 0.8105, 0.8231, 0.8198, 0.8183]
    }
    layers_mean = [0.7259, 0.7297, 0.7372, 0.7326, 0.7300]

    # ==========================================
    # 数据准备：表格 2 (Loss Weight 变化, Num Layers=3)
    # ==========================================
    x_weights = [1, 2, 3, 4, 5, 6, 7, 8]
    weights_data = {
        'LUAD':    [0.6697, 0.6547, 0.6567, 0.6633, 0.6701, 0.6638, 0.6598, 0.6495],
        'LUSC':    [0.6597, 0.6629, 0.6588, 0.6621, 0.6570, 0.6640, 0.6602, 0.6571],
        'BRCA':    [0.7314, 0.7302, 0.7478, 0.7302, 0.7317, 0.7317, 0.7340, 0.7333],
        'KIRC':    [0.7780, 0.7748, 0.7840, 0.7871, 0.7812, 0.7901, 0.7896, 0.7787],
        'Inhouse': [0.8280, 0.8289, 0.8245, 0.8311, 0.8285, 0.8293, 0.8235, 0.8188]
    }
    weights_mean = [0.7334, 0.7303, 0.7344, 0.7348, 0.7334, 0.7358, 0.7334, 0.7275]

    # 执行绘图与保存
    # 1. 保存 Num Layers 图
    draw_and_save_plot(
        x_layers, 
        layers_data, 
        layers_mean, 
        'Number of Layers', 
        'Performance vs. Num Layers (Loss Weight=1)', 
        '/home/Guanjq/NewWork/MedAlignFusion/Code/tools/Draw/Results/num_layers_tuning.svg'
    )

    # 2. 保存 Loss Weight 图
    draw_and_save_plot(
        x_weights, 
        weights_data, 
        weights_mean, 
        'Loss Weight', 
        'Performance vs. Loss Weight (Num Layers=3)', 
        '/home/Guanjq/NewWork/MedAlignFusion/Code/tools/Draw/Results/loss_weight_tuning.svg'
    )

if __name__ == "__main__":
    main()