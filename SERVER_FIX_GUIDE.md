# 服务器训练错误修复指南

## 问题症状

训练时报错：
```
TypeError: DepthLSSTransform: __init__() got an unexpected keyword argument 'depth_input'
```

## 根本原因

`mmdet3d/models/vtransforms/depth_lss.py` 文件缺少 `use_points` 参数定义。

## 解决方案

### 方案 1：使用自动修复脚本（推荐）

```bash
# 1. 拉取最新代码
cd ~/workbench/test_bevfusion
git pull origin main

# 2. 运行修复脚本
bash fix_server.sh
```

### 方案 2：手动修复

#### 步骤 1：确认代码路径

训练命令中使用的代码路径是：`/root/workbench/bevfusion`

但 Git 仓库在：`/root/workbench/test_bevfusion`

**需要将 test_bevfusion 的代码链接到 bevfusion！**

#### 步骤 2：修复 depth_lss.py

编辑文件：`/root/workbench/bevfusion/mmdet3d/models/vtransforms/depth_lss.py`

在第 46 行添加 `use_points` 参数：

```python
@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        use_points: str = 'lidar',  # ← 添加这一行
        depth_input: str = 'scalar',
        add_depth_features: bool = True,
        height_expand: bool = True,
        point_feature_dims: int = 5,
        use_attention: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            use_points=use_points,  # ← 添加这一行
            depth_input=depth_input,
            add_depth_features=add_depth_features,
            height_expand=height_expand,
        )
        self.use_attention = use_attention
```

#### 步骤 3：重新安装 mmdet3d

```bash
cd /root/workbench/bevfusion
python setup.py develop --uninstall
python setup.py develop
```

#### 步骤 4：验证修复

```bash
python -c "from mmdet3d.models.vtransforms import DepthLSSTransform; import inspect; sig = inspect.signature(DepthLSSTransform.__init__); print('参数列表:', list(sig.parameters.keys()))"
```

应该看到输出包含 `use_points`。

### 方案 3：直接使用软链接（最简单）

```bash
# 备份原有 bevfusion
mv /root/workbench/bevfusion /root/workbench/bevfusion.backup

# 创建软链接
ln -s /root/workbench/test_bevfusion /root/workbench/bevfusion

# 验证
ls -la /root/workbench/bevfusion/mmdet3d/models/vtransforms/depth_lss.py
```

## 验证修复成功

运行训练命令：
```bash
torchpack dist-run -np 4 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/mini_dataset_optimized.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --run-dir train_result_mini_optimized
```

如果不再报错 `depth_input`，说明修复成功！

## 常见问题

### Q: 为什么修改了代码还报错？

A: 因为 `python setup.py develop` 安装了代码，修改后需要重新运行：
```bash
python setup.py develop --uninstall
python setup.py develop
```

### Q: 如何确认当前使用的代码路径？

A: 在训练脚本中添加一行打印：
```python
import mmdet3d
print("mmdet3d 路径:", mmdet3d.__file__)
```

### Q: Git 拉取了代码但还是报错？

A: Git 拉取的是 `test_bevfusion`，但训练使用的是 `bevfusion` 目录。需要：
1. 要么使用软链接
2. 要么手动复制文件
3. 要么直接修改 `bevfusion` 目录的代码

## 最新提交

- 提交 ID: `f8972ed`
- 提交信息：添加服务器修复脚本
- 仓库：https://github.com/user-Dorian/test_bevfusion

---

**创建时间**: 2026-04-06
**目标**: 修复服务器训练错误
**关键**: 确保修改的代码被正确安装！
