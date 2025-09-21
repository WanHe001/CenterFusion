# 导入Python 2/3兼容性模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入项目路径初始化模块
import _init_paths
import os

# 导入PyTorch相关模块
import torch
import torch.utils.data

# 导入项目自定义模块
from opts import opts  # 命令行参数解析
from model.model import create_model, load_model, save_model  # 模型创建和加载
from model.data_parallel import DataParallel  # 数据并行处理
from logger import Logger  # 训练日志记录
from dataset.dataset_factory import get_dataset  # 数据集加载
from trainer import Trainer  # 训练器
from test import prefetch_test  # 测试预处理
import json  # JSON数据处理

def get_optimizer(opt, model):
  """根据配置获取优化器
  
  参数:
    opt: 配置对象，包含优化器类型和学习率等参数
    model: 需要优化的模型
  
  返回:
    配置好的优化器对象
  """
  if opt.optim == 'adam':
    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  elif opt.optim == 'sgd':
    # 使用SGD优化器，带动量和权重衰减
    print('Using SGD')
    optimizer = torch.optim.SGD(
      model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
  else:
    # 不支持的优化器类型
    assert 0, opt.optim
  return optimizer

def main(opt):
  """主训练函数
  
  参数:
    opt: 配置对象，包含所有训练参数
  
  功能:
    1. 初始化随机种子和CUDA环境
    2. 加载数据集和模型
    3. 设置优化器和训练器
    4. 执行训练循环
  """
  # 设置随机种子以保证可重复性
  torch.manual_seed(opt.seed)
  # 配置CUDA基准测试
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.eval
  
  # 获取数据集类并更新配置
  Dataset = get_dataset(opt.dataset)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)  # 打印当前配置
  
  # 设置CUDA设备环境
  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  # 初始化日志记录器
  logger = Logger(opt)

  # 创建模型
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
  
  # 初始化优化器
  optimizer = get_optimizer(opt, model)
  start_epoch = 0  # 起始epoch
  lr = opt.lr  # 初始学习率

  # 如果指定了预训练模型，则加载模型参数
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, opt, optimizer)

  # 创建训练器并设置设备
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
  
  # 设置验证数据加载器
  if opt.val_intervals < opt.num_epochs or opt.eval:
    print('Setting up validation data...')
    # 创建验证集数据加载器
    val_loader = torch.utils.data.DataLoader(
      Dataset(opt, opt.val_split), batch_size=1, shuffle=False, 
              num_workers=1, pin_memory=True)

    # 如果只是评估模式，不进行训练
    if opt.eval:
      # 运行验证并获取预测结果
      _, preds = trainer.val(0, val_loader)
      # 使用数据集特定的评估器运行评估
      val_loader.dataset.run_eval(preds, opt.save_dir, n_plots=opt.eval_n_plots, 
                                  render_curves=opt.eval_render_curves)
      return  # 评估完成后直接返回

  # 设置训练数据加载器
  print('Setting up train data...')
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, opt.train_split), batch_size=opt.batch_size, 
        shuffle=opt.shuffle_train, num_workers=opt.num_workers, 
        pin_memory=True, drop_last=True  # 丢弃最后一个不完整的batch
  )

  # 开始训练循环
  print('Starting training...')
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    # 确定当前epoch的标记，用于模型保存
    mark = epoch if opt.save_all else 'last'

    # 记录当前学习率
    for param_group in optimizer.param_groups:
      lr = param_group['lr']
      logger.scalar_summary('LR', lr, epoch)  # 记录学习率到日志
      break
    
    # 训练一个epoch
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))  # 记录当前epoch
    
    # 记录训练结果
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)  # 记录指标到tensorboard
      logger.write('{} {:8f} | '.format(k, v))  # 写入日志文件
    
    # 定期评估模型
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      # 保存当前模型
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      
      # 在验证集上评估模型
      with torch.no_grad():  # 禁用梯度计算以节省内存
        log_dict_val, preds = trainer.val(epoch, val_loader)
        
        # evaluate val set using dataset-specific evaluator
        if opt.run_dataset_eval:
          out_dir = val_loader.dataset.run_eval(preds, opt.save_dir, 
                                                n_plots=opt.eval_n_plots, 
                                                render_curves=opt.eval_render_curves)
          
          # log dataset-specific evaluation metrics
          with open('{}/metrics_summary.json'.format(out_dir), 'r') as f:
            metrics = json.load(f)
          logger.scalar_summary('AP/overall', metrics['mean_ap']*100.0, epoch)
          for k,v in metrics['mean_dist_aps'].items():
            logger.scalar_summary('AP/{}'.format(k), v*100.0, epoch)
          for k,v in metrics['tp_errors'].items():
            logger.scalar_summary('Scores/{}'.format(k), v, epoch)
          logger.scalar_summary('Scores/NDS', metrics['nd_score'], epoch)
      
      # log eval results
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
    
    # save this checkpoint
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.save_point:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
    
    # 更新学习率
    if epoch in opt.lr_step:
      # 按照配置降低学习率
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      # 更新优化器中的学习率
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

  # 关闭日志记录器
  logger.close()

if __name__ == '__main__':
  # 解析命令行参数
  opt = opts().parse()
  # 运行主函数
  main(opt)
