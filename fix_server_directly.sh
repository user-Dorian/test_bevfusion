#!/bin/bash
# 直接修复服务器代码安装问题

echo "=========================================="
echo "直接修复服务器代码"
echo "=========================================="

# 1. 确认 git 代码已更新
echo ""
echo "[1/4] 更新 git 代码..."
cd ~/workbench/test_bevfusion
git pull origin main
echo "当前提交："
git log --oneline -1

# 2. 直接复制文件到 bevfusion 目录
echo ""
echo "[2/4] 复制文件到 bevfusion 目录..."
SOURCE_FILE=~/workbench/test_bevfusion/mmdet3d/models/vtransforms/depth_lss.py
TARGET_FILE=~/workbench/bevfusion/mmdet3d/models/vtransforms/depth_lss.py

echo "源文件：$SOURCE_FILE"
echo "目标文件：$TARGET_FILE"

# 备份原文件
if [ -f "$TARGET_FILE" ]; then
    cp "$TARGET_FILE" "${TARGET_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "已备份原文件"
fi

# 复制文件
cp "$SOURCE_FILE" "$TARGET_FILE"
echo "✓ 文件复制完成"

# 3. 验证文件内容
echo ""
echo "[3/4] 验证文件内容..."
echo "=== 目标文件第 35-55 行 ==="
sed -n '35,55p' "$TARGET_FILE"

# 检查是否包含 kwargs
if grep -q "\*\*kwargs" "$TARGET_FILE"; then
    echo ""
    echo "✓ 文件包含 **kwargs 参数"
else
    echo ""
    echo "✗ 文件不包含 **kwargs 参数，复制可能失败"
fi

# 4. 清理 Python 缓存并重新导入
echo ""
echo "[4/4] 清理 Python 缓存..."
find ~/workbench/bevfusion -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find ~/workbench/bevfusion -name "*.pyc" -delete 2>/dev/null || true
echo "✓ 缓存清理完成"

# 验证
echo ""
echo "=========================================="
echo "验证修复"
echo "=========================================="
python << 'EOF'
import sys
print("Python 路径:", sys.executable)

import mmdet3d
print("\nmmdet3d 路径:", mmdet3d.__file__)

from mmdet3d.models.vtransforms import DepthLSSTransform
import inspect

# 获取源文件
source_file = inspect.getfile(DepthLSSTransform)
print("DepthLSSTransform 源文件:", source_file)

# 检查参数
sig = inspect.signature(DepthLSSTransform.__init__)
params = list(sig.parameters.keys())

print("\n参数列表:")
for i, param in enumerate(params, 1):
    print(f"  {i:2}. {param}")

# 检查关键特性
has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
has_depth_input = 'depth_input' in params

print(f"\n✓ 有 **kwargs: {has_kwargs}")
print(f"✓ 有 depth_input: {has_depth_input}")

if has_kwargs and has_depth_input:
    print("\n✅ 修复成功！代码已正确安装")
    print("\n现在可以运行训练命令了")
else
    print("\n❌ 修复失败！代码可能没有正确加载")
    print("请检查 mmdet3d 的安装路径是否正确")
EOF

echo ""
echo "=========================================="
echo "修复完成"
echo "=========================================="
