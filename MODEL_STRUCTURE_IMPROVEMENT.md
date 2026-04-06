# 模型结构改进方案 - 基于架构的优化

## 🎯 策略转变

**从**: 调整训练超参数（学习率、数据增强等）  
**到**: 改进模型架构结构（保持训练参数与基线一致）

## 📊 基线模型分析（convfuser.yaml）

### 当前配置

```yaml
# Camera Backbone
SwinTransformer:
  embed_dims: 96
  depths: [2, 2, 6, 2]  # Tiny
  drop_path_rate: 0.2

# Camera Neck
GeneralizedLSSFPN:
  out_channels: 256
  # No activation function

# LiDAR Backbone
output_channels: 128  # 较窄
encoder_channels:
  - [16, 16, 32]     # 较窄的通道
  - [32, 32, 64]
  - [64, 64, 128]
  - [128, 128]

# Fusion
ConvFuser  # 基础融合

# VTransform
add_depth_features: false  # 未启用深度特征
```

## 🔧 模型结构改进

### 1. Camera Backbone 改进

**改进点**: 降低 drop_path_rate，提升收敛稳定性

```yaml
drop_path_rate: 0.1  # 从 0.2 降低
```

**原因**:
- Mini 数据集样本少，过高的 dropout 导致欠拟合
- 降低随机深度丢弃率，提升特征提取能力

### 2. Camera Neck 改进

**改进点**: 添加激活函数

```yaml
neck:
  act_cfg:
    type: SiLU  # 新增激活函数
```

**原因**:
- SiLU 提供非线性变换，增强特征表达能力
- 与 Swin Transformer 配合更好

### 3. VTransform 改进

**改进点**: 启用深度特征

```yaml
vtransform:
  add_depth_features: true  # 从 false 改为 true
  height_expand: true
  point_feature_dims: 5
  use_attention: true
```

**原因**:
- 深度特征对 3D 检测至关重要
- 提供额外的几何信息
- 注意力机制增强特征选择

### 4. LiDAR Backbone 改进

**改进点**: 增加通道宽度

```yaml
output_channels: 256  # 从 128 增加

encoder_channels:
  - [32, 32, 64]     # 从 [16, 16, 32] 增加
  - [64, 64, 128]    # 从 [32, 32, 64] 增加
  - [128, 128, 256]  # 从 [64, 64, 128] 增加
  - [256, 256]       # 从 [128, 128] 增加
```

**原因**:
- 更宽的通道 = 更强的特征提取能力
- 提升点云特征表示
- 与 camera 特征更好匹配（都是 256 通道）

### 4. Fusion 改进

**保持**: ConvFuser（不改变）

**原因**:
- `MultiScaleConvFuser` 返回多尺度特征（3 个 tensor）
- `TransFusionHead` 期望单尺度特征（1 个 tensor）
- 两者不兼容，强行使用会导致 assertion error
- ConvFuser 已经足够满足融合需求

**替代方案**:
如果确实需要多尺度融合，需要同时修改 TransFusionHead：
```python
# 需要修改 head 的 forward 方法
# 从 features[0] 改为 features[:3]
```

## 📈 预期效果

| 改进项 | 预期提升 | 风险 |
|--------|---------|------|
| Camera Neck 激活 | +0.5-1% mAP | 低 |
| VTransform 深度特征 | +1-2% mAP | 低 |
| LiDAR Backbone 加宽 | +1-2% mAP | 中（显存增加） |
| Camera Backbone drop_path | +0.5% mAP | 低 |
| **总计** | **+3-5% mAP** | **可控** |

## ⚠️ 保持不变的部分

**关键**: 所有训练超参数与基线保持一致！

```yaml
# ✅ 不变的部分
optimizer.lr: 2.0e-4  # 保持基线值
augment3d:  # 保持基线增强
  scale: [0.9, 1.1]
  rotate: [-0.39, 0.39]
train_cfg:
  max_epochs: 6  # 保持 6 轮
lr_config:  # 保持基线调度
  policy: CosineAnnealing
  warmup_iters: 500
```

## 🚀 训练命令

```bash
# 模型增强版本（推荐）
torchpack dist-run -np 4 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/mini_model_enhanced.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --run-dir train_result_model_enhanced

# 基线对比
torchpack dist-run -np 4 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --run-dir train_result_baseline
```

## 📊 显存影响

| 配置 | 显存占用 | 增加 |
|------|---------|------|
| 基线 (convfuser) | ~8GB | - |
| 模型增强 | ~10-11GB | +2-3GB |

**建议**: batch_size=2 仍然安全

## 📝 评估指标

训练完成后对比：
1. **mAP** - 主要指标
2. **NDS** - 综合指标
3. **各类别 AP** - 分析改进效果
4. **训练稳定性** - loss 曲线波动
5. **收敛速度** - 达到稳定所需 epoch

## 🔬 消融实验（可选）

如果模型增强版本有效，可以进一步测试：

1. **单独启用深度特征**
   ```yaml
   vtransform:
     add_depth_features: true
   ```

2. **单独加宽 LiDAR**
   ```yaml
   lidar:
     backbone:
       output_channels: 256
   ```

3. **单独使用 MultiScaleFuser**
   ```yaml
   fuser:
     type: MultiScaleConvFuser
   ```

---

**创建时间**: 2026-04-06  
**策略**: 模型架构改进  
**目标**: +3-5 mAP（稳定提升）  
**风险**: 低（保持训练参数不变）
