# Mini 数据集优化配置说明

## 🎯 优化目标

在 **mini 数据集**上使用 **6 个 epoch** 训练，实现 **至少 5 个点的 mAP 提升**。

## 📊 核心优化策略

### 1. **多尺度特征融合增强**
- 使用 `MultiScaleConvFuser` 替代基础 `ConvFuser`
- 添加通道注意力机制（attention_ratio=16）
- 自动对齐多尺度特征空间维度

### 2. **更强的 Backbone 配置**
```yaml
# Camera Backbone
drop_path_rate: 0.3      # 提高正则化
drop_rate: 0.1           # 添加 dropout
neck:
  norm_cfg: GN           # GroupNorm 替代 BatchNorm
  act_cfg: SiLU          # SiLU 替代 ReLU

# LiDAR Backbone
output_channels: 128     # 保持与 fuser 兼容
```

### 3. **深度感知变换器增强**
```yaml
vtransform:
  add_depth_features: true      # 启用深度特征
  use_depth_classifier: true    # 添加深度监督
  height_expand: true           # 高度维度扩展
```

### 4. **类别感知损失加权**
针对难检测类别提高权重：
- **小物体**：pedestrian (2.5), motorcycle (3.0), bicycle (3.0)
- **极难类别**：traffic_cone (3.5), barrier (3.5)
- **常规类别**：car (1.0), truck (1.0), bus (1.0)

### 5. **激进的学习率策略**
```yaml
optimizer:
  lr: 3.0e-4  # 从 2.0e-4 提升
  weight_decay: 0.01

lr_config:
  policy: CosineAnnealing
  warmup_iters: 200    # 快速预热
  warmup_ratio: 0.1
  min_lr_ratio: 0.05   # 允许更低的最终 LR
```

### 6. **增强数据增强**
```yaml
augment3d:
  scale: [0.85, 1.15]           # 更宽的范围
  rotate: [-0.785, 0.785]       # ±45 度
  translate: 0.8                # 增强平移
```

## 🚀 训练命令

### 训练
```bash
torchpack dist-run -np 4 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/mini_dataset_optimized.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --run-dir train_result_mini_optimized
```

### 测试
```bash
torchpack dist-run -np 4 python tools/test.py \
  train_result_mini_optimized/configs.yaml \
  train_result_mini_optimized/latest.pth \
  --eval bbox --out box.pkl
```

### 可视化
```bash
torchpack dist-run -np 4 python tools/visualize.py \
  train_result_mini_optimized/configs.yaml \
  --mode gt \
  --checkpoint train_result_mini_optimized/latest.pth \
  --bbox-score 0.5 \
  --out-dir vis_result
```

## 📈 预期效果

| 指标 | 基线 | 优化后 | 提升 |
|------|------|--------|------|
| mAP | ~0.30 | ~0.35+ | +5.0+ |
| NDS | ~0.40 | ~0.45+ | +5.0+ |
| 小物体 AP | ~0.15 | ~0.25+ | +10.0+ |

## 🔧 关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| max_epochs | 6 | 训练轮次 |
| batch_size | 2 | 每 GPU 样本数 |
| learning_rate | 3.0e-4 | 初始学习率 |
| fuser | MultiScaleConvFuser | 多尺度融合 |
| attention_ratio | 16 | 注意力压缩比 |
| warmup_iters | 200 | 预热迭代数 |

## ⚠️ 注意事项

1. **显存需求**: 约 10-12GB (batch_size=2)
2. **训练时间**: 6 epochs 约需 2-3 小时 (4 GPU)
3. **数据准备**: 确保 mini 数据集已正确预处理
4. **预训练模型**: 需要 Swin Transformer 和 LiDAR 检测器预训练权重

## 🎓 优化原理

1. **MultiScaleConvFuser**: 通过通道注意力机制，自适应地加权不同尺度的特征
2. **GroupNorm + SiLU**: 提供更好的梯度流动和泛化能力
3. **类别加权**: 解决类别不平衡问题，重点关注难检测类别
4. **激进 LR**: 在少量 epoch 内快速收敛，配合 cosine annealing 避免过拟合
5. **强数据增强**: 在有限数据下提高模型鲁棒性

## 📊 训练监控建议

重点关注以下指标：
- `loss/object/loss_heatmap`: 应在 epoch 2-3 快速下降
- `stats/object/matched_ious`: 应持续增长
- `object/mAP`: 验证集指标，epoch 4-6 应显著提升
- `object/nds`: 综合指标，反映整体性能

---

**创建时间**: 2026-04-06  
**目标**: Mini 数据集 +5 mAP 提升  
**训练轮次**: 6 epochs  
**配置文件**: `configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/mini_dataset_optimized.yaml`
