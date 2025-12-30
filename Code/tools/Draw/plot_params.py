import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
# 为了支持中文显示（如果您的环境没有SimHei，可以注释掉下面两行，或者换成 Arial/sans-serif）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

def plot_chart():
    # ==========================================
    # 数据准备
    # ==========================================
    datasets = ['LUAD', 'LUSC', 'BRCA', 'KIRC']
    
    # ------------------------------------------
    # 表格 1 数据: 固定 Loss Weight=1, 调整 Num Layers
    # ------------------------------------------
    x1_layers = [1, 2, 3, 4, 5]
    
    # 手动录入的均值数据 (去除 ± 标准差)
    # 更新了 Layer 5 的数据 (LUAD: 0.6682, BRCA: 0.7078)
    data1 = {
        'LUAD': [0.6611, 0.6678, 0.6697, 0.6727, 0.6682],
        'LUSC': [0.6512, 0.6616, 0.6597, 0.6459, 0.6468],
        'BRCA': [0.6929, 0.7234, 0.7314, 0.6903, 0.7078],
        'KIRC': [0.7611, 0.7736, 0.7780, 0.7766, 0.7758]
    }

    # ------------------------------------------
    # 表格 2 数据: 固定 Num Layers=3, 调整 Loss Weight
    # ------------------------------------------
    x2_weights = [1, 2, 3, 4, 5, 6, 7, 8]
    
    data2 = {
        'LUAD': [0.6697, 0.6547, 0.6567, 0.6633, 0.6701, 0.6638, 0.6598, 0.6495],
        'LUSC': [0.6597, 0.6629, 0.6588, 0.6621, 0.6570, 0.6640, 0.6602, 0.6571],
        'BRCA': [0.7314, 0.7302, 0.7478, 0.7302, 0.7317, 0.7317, 0.7340, 0.7333],
        'KIRC': [0.7780, 0.7748, 0.7840, 0.7871, 0.7812, 0.7901, 0.7896, 0.7787]
    }

    # ==========================================
    # 开始绘图
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- 辅助函数：绘制单个子图 ---
    def draw_subplot(ax, x_data, y_data_dict, x_label, title):
        # 1. 绘制每个数据集的虚线
        mean_curve = np.zeros(len(x_data))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'] # 常用配色
        
        for idx, ds_name in enumerate(datasets):
            y_vals = y_data_dict[ds_name]
            ax.plot(x_data, y_vals, linestyle='--', marker='o', 
                    markersize=4, alpha=0.7, color=colors[idx], label=ds_name)
            mean_curve += np.array(y_vals)
        
        # 2. 计算并绘制 Mean 曲线 (实线)
        mean_curve /= len(datasets)
        ax.plot(x_data, mean_curve, linestyle='-', linewidth=3, 
                color='black', label='Mean')

        # 3. 找到 Mean 的最高点并打星
        max_idx = np.argmax(mean_curve)
        max_x = x_data[max_idx]
        max_y = mean_curve[max_idx]
        
        ax.scatter(max_x, max_y, color='red', marker='*', s=300, zorder=10, label='Best Mean')
        
        # 在星星旁边标注数值
        ax.annotate(f'{max_y:.4f}', xy=(max_x, max_y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontweight='bold', color='red')

        # 设置标签和标题
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Performance (AUC/Metric)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    # --- 绘制左图 (Num Layers) ---
    draw_subplot(axes[0], x1_layers, data1, 'Number of Layers', 
                 'Performance vs. Num Layers (Loss Weight=1)')
    
    # --- 绘制右图 (Loss Weight) ---
    draw_subplot(axes[1], x2_weights, data2, 'Loss Weight', 
                 'Performance vs. Loss Weight (Num Layers=3)')

    plt.tight_layout()
    
    # 保存为 SVG
    filename = '/home/Guanjq/NewWork/MedAlignFusion/Code/tools/Draw/Results/hyperparameter_tuning_curves.png'
    plt.savefig(filename, format='png')
    print(f"图表已成功保存为: {filename}")
    plt.show()

if __name__ == "__main__":
    plot_chart()