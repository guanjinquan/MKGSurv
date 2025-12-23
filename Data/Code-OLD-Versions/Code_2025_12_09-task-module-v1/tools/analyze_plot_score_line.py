import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

# 1. 定义文件路径
# 输入文件路径 (保持您原本的路径)
file_path = '/home/Guanjq/NewWork/MedAlignFusion/Results/draw/points_oscc_inhouse_run004+medkgat_fusion.jsonl'
# 输出图片路径
output_path = os.path.join(os.path.dirname(file_path), 'results_nature_style.png')

def configure_nature_style():
    """
    配置 Matplotlib 以生成符合 Nature 风格的图表
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 0.2
    plt.rcParams['ytick.major.width'] = 0.2
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['legend.frameon'] = False  # 图例无边框

def plot_trend_nature_style(path, save_path):
    # 应用样式配置
    configure_nature_style()

    try:
        x_vals = []
        y_vals = []

        # 2. 读取数据
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            return

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    point = json.loads(line)
                    if isinstance(point, list) and len(point) == 2:
                        x_vals.append(point[0])
                        y_vals.append(point[1])
                except json.JSONDecodeError:
                    continue

        if not x_vals:
            print("No valid data points found.")
            return

        x = np.array(x_vals)
        y = np.array(y_vals)

        # 3. 统计分析 (Pearson 相关系数 & 线性拟合)
        # 计算 Pearson r 和 p-value
        r, p_value = stats.pearsonr(x, y)
        
        # 线性拟合 (y = kx + b)
        slope, intercept = np.polyfit(x, y, 1)
        trend_fn = np.poly1d([slope, intercept])
        
        trend_x = np.linspace(min(x), max(x), 100)
        trend_y = trend_fn(trend_x)

        # 4. 绘图 (Nature Style)
        # 推荐使用正方形或接近正方形的比例
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        # 绘制散点 (更小、半透明)
        # s=10 控制点的大小, alpha=0.6 控制透明度
        ax.scatter(x, y, color='#2C3E50', s=15, alpha=0.6, edgecolors='none', label='Data Points')

        # 绘制趋势线 (红色实线)
        ax.plot(trend_x, trend_y, color='#E74C3C', linewidth=2, linestyle='-')

        # 5. 添加统计标注 (Equation, Pearson r, P-value)
        # 格式化 P 值显示
        if p_value < 0.001:
            p_text = "P < 0.001"
        else:
            p_text = f"P = {p_value:.3f}"

        stats_text = (
            f"$y = {slope:.2f}x + {intercept:.2f}$\n"
            f"Pearson $r = {r:.2f}$\n"
            f"{p_text}"
        )

        # 将文本放置在左上角 (transform=ax.transAxes 使用相对坐标 0-1)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.0, linewidth=0))

        # 6. 轴标签与边框处理
        ax.set_xlabel('Similarity Score (LLM)', fontsize=14, labelpad=10) # 示例标签，请根据实际含义修改
        ax.set_ylabel('Ground Truth', fontsize=14, labelpad=10)       # 示例标签，请根据实际含义修改

        # 移除顶部和右侧的边框 (Spines) —— 科研绘图常见风格
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 调整布局以防止标签被截断
        plt.tight_layout()

        # 保存图片
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Success: Plot saved to {save_path}")
        print(f"Stats: r={r:.4f}, p={p_value}, y={slope:.4f}x+{intercept:.4f}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    plot_trend_nature_style(file_path, output_path)