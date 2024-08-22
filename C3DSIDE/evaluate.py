import argparse
import numpy as np
import os
from evaluation_utils import *

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
parser.add_argument('--predicted_disp_path', default='', type=str, help='Path to the estimated disparities (predictions)')
parser.add_argument('--gt_path', default='', help='Path to the ground truth disparities')
parser.add_argument('--min_depth', type=float, help='Minimum depth for evaluation', default=1000)
parser.add_argument('--max_depth', type=float, help='Maximum depth for evaluation', default=3000)

args = parser.parse_args()

# 定义计算误差距离的函数
def dist(gt, dp):
    """Compute the relative squared error between ground truth and predicted depth."""
    dist = ((gt - dp) ** 2) / gt
    return dist

if __name__ == '__main__':

    # 加载预测的视差图
    pred_disparities = np.load(args.predicted_disp_path)

    # 加载真实深度图
    path = args.gt_path
    gt_depths_files = os.listdir(path)
    gt_depths_files = [os.path.join(path, f) for f in gt_depths_files]

    # 初始化用于存储深度数据的数组
    gt_depthss = np.zeros((669, 480, 640))
    for i in range(669):
        gt_depthss[i] = np.load(gt_depths_files[i])

    gt_depths = gt_depthss
    num_samples = 669

    # 将视差图转换为深度图
    pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_depths, pred_disparities)

    # 初始化误差度量数组
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)
    mse = np.zeros(num_samples, np.float32)
    mae = np.zeros(num_samples, np.float32)
    r_squared = np.zeros(num_samples, np.float32)

    # 记录效果好的样本索引
    ids = []
    for i in range(num_samples):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]
        pred_disp = pred_disparities_resized[i]
        mask = gt_depth > 0  # 只考虑有效深度值

        # 计算误差度量
        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i], mse[i], mae[i], r_squared[i] = compute_errors(
            gt_depth[mask], pred_depth[mask])

        # 筛选效果好的样本
        if abs_rel[i] < 0.04 and a1[i] > 0.99:
            ids.append(i + 1)

    # 打印结果
    print("{:>12}, {:>12}, {:>12}, {:>12}, {:>12}, {:>12}, {:>12}, {:>12}, {:>12}, {:>12}".format(
        'abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3', 'mse', 'mae', 'r_squared'))

    print("{:12.4f}, {:12.4f}, {:12.3f}, {:12.3f}, {:12.3f}, {:12.3f}, {:12.3f}, {:12.3f}, {:12.3f}, {:12.3f}".format(
        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean(),
        mse.mean(), mae.mean(), r_squared.mean()))
