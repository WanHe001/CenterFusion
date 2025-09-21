#!/bin/bash

# 设置CUDA设备 (使用第2块GPU，索引从0开始)
export CUDA_VISIBLE_DEVICES=1

# 进入源代码目录
cd src

# 执行3D目标检测和评估
python test.py ddd \  # 指定任务类型为3D检测(Decode, Detect, Detect)
    --exp_id centerfusion \  # 实验ID，用于标识当前实验
    --dataset nuscenes \  # 使用nuScenes自动驾驶数据集
    --val_split mini_val \  # 使用mini_val验证集(nuScenes的小型验证集)
    --run_dataset_eval \  # 运行数据集官方评估
    --num_workers 4 \  # 数据加载使用4个工作进程
    
    # 模型功能配置
    --nuscenes_att \  # 启用nuScenes属性预测
    --velocity \  # 启用速度预测
    --gpus 0 \  # 使用GPU 0(实际对应CUDA_VISIBLE_DEVICES=1)
    
    # 点云处理配置
    --pointcloud \  # 启用点云处理
    --radar_sweeps 3 \  # 使用3次雷达扫描数据
    --max_pc_dist 60.0 \  # 最大点云距离60米(超出此距离的点将被过滤)
    --pc_z_offset -0.0 \  # 点云Z轴偏移量
    
    # 模型加载配置
    --load_model ../models/centerfusion_e60.pth \  # 加载预训练模型
    --flip_test \  # 启用测试时翻转增强
    
    # 注释掉的选项(需要时可取消注释)
    # --resume \  # 恢复训练(当前为测试模式，不需要此选项)