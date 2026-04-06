# 训练结果分析与改进建议

## 📊 当前结果（mini_conservative.yaml 前的尝试）

```
mAP: 0.3191
NDS: 0.3601
```

### 各类别表现

| 类别 | AP | 状态 | 分析 |
|------|-----|------|------|
| car | 0.827 | ✅ 良好 | 基线水平 |
| truck | 0.563 | ✅ 正常 | 基线水平 |
| bus | 0.646 | ⚠️ 速度误差大 | 2.855 的 AVE |
| trailer | 0.000 | ❌ 失效 | 训练样本可能不足 |
| construction_vehicle | 0.000 | ❌ 失效 | 训练样本可能不足 |
| pedestrian | 0.640 | ✅ 正常 | 可接受 |
| motorcycle | 0.309 | ⚠️ 偏低 | 可提升 |
| bicycle | 0.000 | ❌ 失效 | 样本极少 |
| traffic_cone | 0.206 | ⚠️ 偏低 | 可提升 |
| barrier | 0.000 | ❌ 失效 | 样本极少 |

## 🔍 失败原因分析

### 1. 过度优化问题

**原配置（mini_dataset_optimized.yaml）的问题**：
```yaml
# ❌ 过于激进
optimizer.lr: 3.0e-4  # 过高
augment3d.scale: [0.85, 1.15]  # 过强
augment3d.rotate: [-0.785, 0.785]  # ±45 度，过强
class_weights: [1.0, 1.0, 1.5, ..., 3.5]  # 破坏平衡
```

### 2. 类别不平衡问题

**零 AP 类别分析**：
- `trailer`, `construction_vehicle`, `barrier`, `bicycle` 全部为 0
- 这些类别在 mini 数据集中**样本极少**
- 过度加权反而导致模型无法学习通用特征

### 3. 学习率策略问题

```yaml
# ❌ 原配置
lr: 3.0e-4
warmup_iters: 200
min_lr_ratio: 0.05

# 问题：
# - 初始 LR 过高，6 轮内无法稳定
# - warmup 不足，前期梯度爆炸
# - 最终 LR 过低，后期无法收敛
```

## ✅ 改进策略（mini_conservative.yaml）

### 1. 保守的学习率

```yaml
optimizer:
  lr: 2.5e-4  # 从 2.0e-4 小幅提升
  
lr_config:
  warmup_iters: 300  # 增加 warmup
  min_lr_ratio: 0.1  # 保持较高最终 LR
```

### 2. 保守的数据增强

```yaml
augment3d:
  scale: [0.9, 1.1]  # 从 [0.85, 1.15] 降低
  rotate: [-0.39, 0.39]  # ±22.5 度，从±45 度降低
  translate: 0.5  # 从 0.8 降低
```

### 3. 移除类别加权

**原因**：
- Mini 数据集本身样本少
- 类别加权破坏原有平衡
- 让模型自然学习特征

### 4. 保持 ConvFuser

**原因**：
- MultiScaleConvFuser 需要更多调参
- ConvFuser 已经足够稳定
- 减少变量，专注核心优化

## 📈 预期效果

| 指标 | 当前 | 预期 | 提升 |
|------|------|------|------|
| mAP | 0.3191 | 0.34-0.36 | +2-4% |
| NDS | 0.3601 | 0.38-0.40 | +2-4% |
| trailer | 0.000 | 0.1-0.2 | 恢复 |
| construction_vehicle | 0.000 | 0.1-0.2 | 恢复 |

## 🚀 训练命令

```bash
# 保守优化版本（推荐）
torchpack dist-run -np 4 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/mini_conservative.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --run-dir train_result_conservative

# 原始 convfuser 版本（基线对比）
torchpack dist-run -np 4 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --run-dir train_result_baseline
```

## 📝 后续优化方向

### 如果保守版本有效（mAP > 0.34）：

1. **逐步增加学习率**: 2.5e-4 → 2.7e-4 → 3.0e-4
2. **逐步增强数据增强**: scale → [0.85, 1.15]
3. **添加温和的类别加权**: max_weight ≤ 2.0

### 如果保守版本无效（mAP < 0.34）：

1. **检查数据预处理**: 确认 mini 数据集正确
2. **检查预训练权重**: 确认加载正确
3. **增加训练轮次**: 6 → 10 → 15
4. **考虑使用全量数据集**: 6 轮可能不够

## ⚠️ 重要提醒

**不要过度优化！**

- 每次只改 1-2 个超参数
- 小步快跑，逐步验证
- 保留基线配置作为对比
- 记录每次实验结果

---

**创建时间**: 2026-04-06  
**策略**: 保守优化，稳定提升  
**目标**: +2-3 mAP
