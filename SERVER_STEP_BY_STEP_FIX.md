# 服务器代码修复 - 逐步操作指南

## ⚠️ 问题诊断

错误信息：
```
TypeError: DepthLSSTransform: __init__() got an unexpected keyword argument 'use_points'
```

**原因**：服务器上的 `mmdet3d/models/vtransforms/depth_lss.py` 文件未正确更新。

---

## 🔧 解决方案（按顺序执行）

### 步骤 1：确认当前代码状态

```bash
# 登录服务器后，首先检查当前代码状态
cd ~/workbench/test_bevfusion

# 检查 git 状态
git log --oneline -3
git status

# 检查 depth_lss.py 文件
echo "=== 检查 depth_lss.py 第 35-52 行 ==="
sed -n '35,52p' mmdet3d/models/vtransforms/depth_lss.py
```

**期望输出**应该包含：
```python
downsample: int = 1,
use_points: str = 'lidar',  # ← 必须有这一行
depth_input: str = 'scalar',
```

### 步骤 2：强制拉取最新代码

```bash
cd ~/workbench/test_bevfusion

# 强制拉取
git fetch origin
git reset --hard origin/main

# 验证拉取成功
echo "=== 最新提交 ==="
git log --oneline -1

echo ""
echo "=== 检查 use_points 参数 ==="
grep -n "use_points: str" mmdet3d/models/vtransforms/depth_lss.py
```

**期望输出**：
```
=== 最新提交 ===
4f88250 添加强制同步脚本 - 解决服务器代码未更新问题

=== 检查 use_points 参数 ===
46:        use_points: str = 'lidar',
```

### 步骤 3：如果 git 拉取失败，手动修复

如果 git 拉取后仍然没有 `use_points` 参数，手动编辑文件：

```bash
cd ~/workbench/test_bevfusion

# 备份原文件
cp mmdet3d/models/vtransforms/depth_lss.py mmdet3d/models/vtransforms/depth_lss.py.backup

# 使用 sed 添加 use_points 参数
# 在第 45 行（downsample: int = 1,）后添加 use_points 行
sed -i "45 a\\        use_points: str = 'lidar'," mmdet3d/models/vtransforms/depth_lss.py

# 在父类调用中添加 use_points 参数（在 dbound=dbound, 后）
sed -i "s/dbound=dbound,/dbound=dbound,\n            use_points=use_points,/" mmdet3d/models/vtransforms/depth_lss.py

# 验证修改
echo "=== 修改后的代码（第 40-65 行）==="
sed -n '40,65p' mmdet3d/models/vtransforms/depth_lss.py
```

### 步骤 4：重新安装 mmdet3d

```bash
cd ~/workbench/bevfusion

# 卸载旧版本
python setup.py develop --uninstall

# 清理 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# 重新安装
python setup.py develop

# 等待安装完成...
echo "安装完成！"
```

### 步骤 5：验证安装

```bash
# 验证参数是否正确
python << 'EOF'
from mmdet3d.models.vtransforms import DepthLSSTransform
import inspect

sig = inspect.signature(DepthLSSTransform.__init__)
params = list(sig.parameters.keys())

print("\nDepthLSSTransform.__init__ 参数列表:")
for i, param in enumerate(params, 1):
    print(f"  {i:2}. {param}")

# 检查关键参数
required_params = ['use_points', 'depth_input', 'add_depth_features', 'height_expand']
missing = [p for p in required_params if p not in params]

if missing:
    print(f"\n✗ 缺少参数：{missing}")
    print("需要重新执行步骤 3 和步骤 4")
else:
    print(f"\n✓ 所有必需参数都存在")
    print("✓ 代码修复成功！")
EOF
```

### 步骤 6：运行训练测试

```bash
cd ~/workbench/bevfusion

# 运行训练
torchpack dist-run -np 4 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --run-dir train_result

# 如果不再报错 use_points 或 depth_input，说明修复成功！
```

---

## 🚨 常见问题

### Q1: git pull 显示已经是最新的，但代码还是旧的

**解决**：
```bash
cd ~/workbench/test_bevfusion
git fetch origin
git reset --hard origin/main
```

### Q2: 修改了代码但仍然报错

**原因**：`python setup.py develop` 安装的代码没有更新

**解决**：
```bash
cd ~/workbench/bevfusion
python setup.py develop --uninstall
python setup.py develop
```

### Q3: 如何确认当前使用的代码路径？

**验证**：
```bash
python << 'EOF'
import mmdet3d
print("mmdet3d 安装路径:", mmdet3d.__file__)

from mmdet3d.models.vtransforms import DepthLSSTransform
import inspect
source_file = inspect.getfile(DepthLSSTransform)
print("DepthLSSTransform 源文件:", source_file)
EOF
```

### Q4: 软链接问题

如果使用软链接：
```bash
# 检查软链接
ls -la ~/workbench/bevfusion

# 如果不是链接到 test_bevfusion，重新创建
rm -rf ~/workbench/bevfusion
ln -s ~/workbench/test_bevfusion ~/workbench/bevfusion
```

---

## 📝 快速修复脚本

复制以下脚本并执行：

```bash
#!/bin/bash
# quick_fix.sh - 快速修复脚本

echo "=== 快速修复开始 ==="

cd ~/workbench/test_bevfusion
git fetch origin
git reset --hard origin/main

cd ~/workbench/bevfusion
python setup.py develop --uninstall
python setup.py develop

python -c "from mmdet3d.models.vtransforms import DepthLSSTransform; import inspect; params = list(inspect.signature(DepthLSSTransform.__init__).parameters.keys()); print('参数检查:', 'use_points' in params)"

echo "=== 修复完成 ==="
```

执行：
```bash
chmod +x quick_fix.sh
bash quick_fix.sh
```

---

## ✅ 验证成功标准

1. ✓ `git log` 显示最新提交 `4f88250`
2. ✓ `depth_lss.py` 第 46 行包含 `use_points: str = 'lidar'`
3. ✓ Python 验证脚本显示 `use_points` 在参数列表中
4. ✓ 训练命令不再报错 `use_points` 或 `depth_input`

---

**最新提交**: `4f88250`  
**提交信息**: 添加强制同步脚本 - 解决服务器代码未更新问题  
**仓库**: https://github.com/user-Dorian/test_bevfusion  
**更新时间**: 2026-04-06
