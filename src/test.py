# Python 2/3 兼容性导入
from __future__ import absolute_import  # 确保导入是绝对路径
from __future__ import division         # 确保除法运算行为一致
from __future__ import print_function   # 确保print函数行为一致

# 项目初始化路径
import _init_paths

# 标准库导入
import os       # 操作系统接口
import json     # JSON数据处理
import cv2      # OpenCV图像处理
import numpy as np  # 数值计算
import time     # 时间相关功能
import copy     # 对象深拷贝

# 第三方库导入
from progress.bar import Bar  # 进度条显示
import torch    # PyTorch深度学习框架

# 项目内部模块导入
from opts import opts        # 命令行参数解析
from logger import Logger    # 日志记录
from utils.utils import AverageMeter  # 平均值计算工具
from dataset.dataset_factory import dataset_factory  # 数据集工厂
from detector import Detector  # 检测器主类


class PrefetchDataset(torch.utils.data.Dataset):
  """预取数据集类，继承自PyTorch的Dataset类
  
  用于高效地预加载和预处理测试数据，支持多尺度测试和点云数据
  
  Args:
      opt: 配置参数对象
      dataset: 原始数据集对象
      pre_process_func: 预处理函数
  """
  def __init__(self, opt, dataset, pre_process_func):
    """初始化预取数据集"""
    self.images = dataset.images  # 图像ID列表
    self.load_image_func = dataset.coco.loadImgs  # 图像信息加载函数
    self.img_dir = dataset.img_dir  # 图像目录路径
    self.pre_process_func = pre_process_func  # 预处理函数
    self.get_default_calib = dataset.get_default_calib  # 默认标定数据获取函数
    self.opt = opt  # 配置参数
    self.dataset = dataset  # 原始数据集
  
  def __getitem__(self, index):
    """获取单个数据项
    
    Args:
        index: 数据索引
        
    Returns:
        tuple: (img_id, ret) 图像ID和包含预处理结果的字典
    """
    # 获取图像ID和信息
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)  # 读取图像
    
    # 多尺度处理
    images, meta = {}, {}
    for scale in opt.test_scales:
      input_meta = {}
      # 获取标定数据，如果没有则使用默认值
      calib = img_info['calib'] if 'calib' in img_info \
        else self.get_default_calib(image.shape[1], image.shape[0])
      input_meta['calib'] = calib
      # 预处理图像
      images[scale], meta[scale] = self.pre_process_func(
        image, scale, input_meta)
      
    # 构建返回结果字典
    ret = {
      'images': images,    # 多尺度处理后的图像
      'image': image,      # 原始图像
      'meta': meta         # 元数据
    }
    
    # 如果是视频的第一帧，添加标记
    if 'frame_id' in img_info and img_info['frame_id'] == 1:
      ret['is_first_frame'] = 1
      ret['video_id'] = img_info['video_id']
    
    # 如果启用点云，添加点云数据
    if opt.pointcloud:
      assert len(opt.test_scales)==1, "多尺度测试不支持点云数据"
      scale = opt.test_scales[0]
      # 加载点云数据
      pc_2d, pc_N, pc_dep, pc_3d = self.dataset._load_pc_data(
        image, img_info, 
        meta[scale]['trans_input'], 
        meta[scale]['trans_output'])
      ret.update({
        'pc_2d': pc_2d,  # 2D点云数据
        'pc_N': pc_N,    # 点云数量
        'pc_dep': pc_dep, # 点云深度
        'pc_3d': pc_3d   # 3D点云数据
      })

    return img_id, ret

  def __len__(self):
    """返回数据集大小"""
    return len(self.images)

def prefetch_test(opt):
  """使用预取数据加载器进行测试
  
  使用PrefetchDataset进行高效数据加载和测试，支持跟踪和点云功能
  
  Args:
      opt: 配置参数对象
  """
  # 设置CUDA环境变量
  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  
  # 初始化数据集和检测器
  Dataset = dataset_factory[opt.test_dataset]  # 从工厂获取数据集类
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)  # 更新配置
  print(opt)  # 打印当前配置
  Logger(opt)  # 初始化日志
  
  # 确定数据集分割 (val/test)
  split = 'val' if not opt.trainval else 'test'
  if split == 'val':
    split = opt.val_split  # 使用指定的验证集分割
  
  dataset = Dataset(opt, split)  # 初始化数据集
  detector = Detector(opt)       # 初始化检测器
  
  # 加载已有结果 (如果指定)
  if opt.load_results != '':
    load_results = json.load(open(opt.load_results, 'r'))
    # 过滤掉指定类别的结果
    for img_id in load_results:
      for k in range(len(load_results[img_id])):
        if load_results[img_id][k]['class'] - 1 in opt.ignore_loaded_cats:
          load_results[img_id][k]['score'] = -1
  else:
    load_results = {}

  # 创建数据加载器
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process),  # 使用预取数据集
    batch_size=1,         # 批大小为1
    shuffle=False,        # 不随机打乱
    num_workers=1,        # 使用1个工作进程
    pin_memory=True)      # 使用固定内存

  # 初始化结果存储和时间统计
  results = {}
  num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)  # 进度条
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'track']
  avg_time_stats = {t: AverageMeter() for t in time_stats}  # 时间统计器
  
  # 如果使用预加载结果，直接填充结果
  if opt.use_loaded_results:
    for img_id in data_loader.dataset.images:
      results[img_id] = load_results['{}'.format(img_id)]
    num_iters = 0  # 跳过实际推理
  
  # 主测试循环
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    if ind >= num_iters:  # 达到最大迭代次数时退出
      break
    
    # 跟踪相关处理
    if opt.tracking and ('is_first_frame' in pre_processed_images):
      # 设置前一帧的检测结果
      if '{}'.format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
        pre_processed_images['meta']['pre_dets'] = \
          load_results['{}'.format(int(img_id.numpy().astype(np.int32)[0]))]
      else:
        print('\nNo pre_dets for', int(img_id.numpy().astype(np.int32)[0]), 
          '. Use empty initialization.')
        pre_processed_images['meta']['pre_dets'] = []
      detector.reset_tracking()  # 重置跟踪器
      print('Start tracking video', int(pre_processed_images['video_id']))
    
    # 公共检测处理
    if opt.public_det:
      if '{}'.format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
        pre_processed_images['meta']['cur_dets'] = \
          load_results['{}'.format(int(img_id.numpy().astype(np.int32)[0]))]
      else:
        print('No cur_dets for', int(img_id.numpy().astype(np.int32)[0]))
        pre_processed_images['meta']['cur_dets'] = []
    
    # 运行检测器
    ret = detector.run(pre_processed_images)
    results[int(img_id.numpy().astype(np.int32)[0])] = ret['results']
    
    # 更新进度条和时间统计
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    
    # 定期打印进度
    if opt.print_iter > 0:
      if ind % opt.print_iter == 0:
        print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
    else:
      bar.next()
  
  bar.finish()  # 完成进度条
  
  # 保存结果
  if opt.save_results:
    save_path = opt.save_dir + '/save_results_{}{}.json'.format(
      opt.test_dataset, opt.dataset_version)
    print('saving results to', save_path)
    json.dump(_to_list(copy.deepcopy(results)), open(save_path, 'w'))
  
  # 运行评估
  dataset.run_eval(results, opt.save_dir, 
                  n_plots=opt.eval_n_plots, 
                  render_curves=opt.eval_render_curves)

def test(opt):
  """不使用预取数据加载器的测试函数
  
  直接加载图像进行测试，适用于简单测试场景
  
  Args:
      opt: 配置参数对象
  """
  # 设置CUDA环境变量
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  # 初始化数据集和检测器
  Dataset = dataset_factory[opt.test_dataset]  # 从工厂获取数据集类
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)  # 更新配置
  print(opt)  # 打印当前配置
  Logger(opt)  # 初始化日志
  
  # 确定数据集分割 (val/test)
  split = 'val' if not opt.trainval else 'test'
  if split == 'val':
    split = opt.val_split  # 使用指定的验证集分割
  
  dataset = Dataset(opt, split)  # 初始化数据集
  detector = Detector(opt)       # 初始化检测器

  # 加载已有结果 (如果指定)
  if opt.load_results != '':
    load_results = json.load(open(opt.load_results, 'r'))

  # 初始化结果存储和时间统计
  results = {}
  num_iters = len(dataset) if opt.num_iters < 0 else opt.num_iters
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)  # 进度条
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}  # 时间统计器
  
  # 主测试循环
  for ind in range(num_iters):
    img_id = dataset.images[ind]  # 获取图像ID
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]  # 加载图像信息
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])  # 图像路径
    
    # 准备输入元数据
    input_meta = {}
    if 'calib' in img_info:
      input_meta['calib'] = img_info['calib']  # 标定数据
    
    # 跟踪相关处理
    if (opt.tracking and ('frame_id' in img_info) and img_info['frame_id'] == 1):
      detector.reset_tracking()  # 重置跟踪器
      input_meta['pre_dets'] = load_results[img_id]  # 设置前一帧检测结果
    
    # 运行检测器
    ret = detector.run(img_path, input_meta)
    results[img_id] = ret['results']  # 存储结果

    # 更新进度条和时间统计
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  
  bar.finish()  # 完成进度条
  
  # 保存结果
  if opt.save_results:
    save_path = opt.save_dir + '/save_results_{}{}.json'.format(
      opt.test_dataset, opt.dataset_version)
    print('saving results to', save_path)
    json.dump(_to_list(copy.deepcopy(results)), open(save_path, 'w'))
  
  # 运行评估
  dataset.run_eval(results, opt.save_dir, 
                  n_plots=opt.eval_n_plots, 
                  render_curves=opt.eval_render_curves)


def _to_list(results):
  """将结果中的numpy数组转换为Python列表
  
  用于JSON序列化，因为JSON不支持numpy数据类型
  
  Args:
      results: 包含检测结果的字典，可能包含numpy数组
      
  Returns:
      dict: 转换后的结果字典，所有numpy数组已转为列表
  """
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  """主程序入口"""
  opt = opts().parse()  # 解析命令行参数
  
  # 根据参数选择测试模式
  if opt.not_prefetch_test:
    test(opt)      # 不使用预取数据加载器的测试
  else:
    prefetch_test(opt)  # 使用预取数据加载器的测试
