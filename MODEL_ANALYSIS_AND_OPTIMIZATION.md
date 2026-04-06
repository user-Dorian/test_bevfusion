# BEVFusion 模型分析与优化报告

## 1. 训练数据分析

### 1.1 性能指标分析（基于已验证微提升模型训练数据）

| Epoch | mAP | mATE | mASE | mAOE | mAVE | mAAE | NDS |
|-------|-----|------|------|------|------|------|-----|
| 1     | 0.317 | 0.4662 | 0.4874 | 0.8234 | 0.7500 | 0.4479 | 0.357 |
| 2     | 0.324 | 0.4888 | 0.4861 | 0.8141 | 0.6408 | 0.3761 | 0.362 |
| 3     | 0.329 | 0.4928 | 0.4915 | 0.7597 | 0.6866 | 0.3686 | 0.365 |
| 4     | 0.323 | 0.4789 | 0.4819 | 0.6905 | 0.6363 | 0.3891 | 0.363 |
| 5     | 0.328 | 0.4817 | 0.4838 | 0.7345 | 0.6564 | 0.3803 | 0.364 |
| 6     | 0.326 | 0.4817 | 0.4813 | 0.7414 | 0.6482 | 0.3883 | 0.363 |

### 1.2 训练曲线分析

**Loss 曲线趋势：**
- 初始 loss: ~11.44 (epoch 1, iter 50)
- 最终 loss: ~1.13 (epoch 6, iter 800)
- 整体下降趋势明显，说明模型正在学习

**Grad Norm 分析：**
- 初始 grad_norm: 82.95 (epoch 1, iter 100)
- 最终 grad_norm: ~3.18 (epoch 6, iter 800)
- 梯度范数逐渐稳定，训练趋于收敛

**Matched IoU 分析：**
- 初始: 0.0357 (epoch 1, iter 50)
- 最终: 0.5552 (epoch 6, iter 800)
- 模型定位精度显著提升

## 2. 性能瓶颈分析

### 2.1 主要问题

1. **深度特征未利用**：
   - 配置中 `add_depth_features: false`
   - 深度信息对 3D 检测至关重要

2. **过拟合风险**：
   - `drop_path_rate: 0.2` 对于 mini 数据集可能过高
   - 导致模型在验证集上表现不稳定

3. **训练稳定性**：
   - 验证集 mAP 波动较大（0.317-0.329）
   - 说明模型训练存在一定不稳定性

4. **小物体检测**：
   - 部分类别（如 trailer, construction_vehicle, barrier）AP 为 0
   - 模型对小物体和稀有类别的检测能力弱

## 3. 优化策略

### 3.1 模型结构改进

| 改进项 | 基线 | 优化后 | 预期效果 |
|--------|------|--------|----------|
| **深度特征** | `add_depth_features: false` | `add_depth_features: true` | +1-2% mAP |
| **深度监督** | 无 | `use_depth_classifier: true` | +0.5-1% mAP |
| **Dropout 率** | `drop_path_rate: 0.2` | `drop_path_rate: 0.1` | +0.5-1% mAP |
| **注意力机制** | `use_attention: true` | 保持 | 维持性能 |
| **特征维度** | `point_feature_dims: 5` | 保持 | 维持性能 |

### 3.2 训练稳定性保证

- **保持学习率**：2.0e-4（不变）
- **保持优化器**：AdamW（不变）
- **保持学习率策略**：Cosine Annealing（不变）
- **保持批量大小**：2（不变）
- **保持训练轮数**：6（不变）

## 4. 预期效果

### 4.1 性能提升

| 指标 | 基线 | 预期 | 提升 |
|------|------|------|------|
| mAP | ~0.326 | ~0.35-0.37 | +2.4-4.4% |
| NDS | ~0.363 | ~0.38-0.40 | +1.7-3.7% |
| 训练稳定性 | 波动 | 稳定 | 明显改善 |
| 收敛速度 | 较慢 | 更快 | 提升 1-2 个 epoch |

### 4.2 小物体检测改进

- **预期**：trailer、construction_vehicle、barrier 等类别 AP 从 0 提升到 >0.1
- **原因**：深度特征提供了更准确的 3D 位置信息

## 5. 配置文件

**优化配置**：`configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/optimized.yaml`

**核心改进**：
```yaml
model:
  encoders:
    camera:
      backbone:
        drop_path_rate: 0.1  # 降低过拟合
      vtransform:
        add_depth_features: true  # 启用深度特征
        use_depth_classifier: true  # 深度监督
        height_expand: true
        point_feature_dims: 5
        use_attention: true
```

## 6. 训练命令

```bash
# 优化模型训练
torchpack dist-run -np 4 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/optimized.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --run-dir train_result_optimized

# 基线模型训练（对比）
torchpack dist-run -np 4 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/default.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --run-dir train_result_baseline
```

## 7. 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 显存增加 | 高 | 可能 OOM | 保持 batch_size=2 |
| 训练时间增加 | 中 | 增加 10-15% | 可接受，性能提升显著 |
| 过拟合 | 低 | 验证集性能下降 | 降低 drop_path_rate=0.1 |
| 收敛不稳定 | 低 | loss 波动 | 保持学习率和优化器不变 |

## 8. 结论

通过启用深度特征和深度监督，同时降低 dropout 率，预计可以在保持训练稳定性的前提下，实现 2-4% 的 mAP 提升，特别是在小物体检测和 3D 定位精度方面。

**优化策略**：精准定位关键改进点，最小化配置变动，确保训练过程稳定，同时实现性能的显著提升。
