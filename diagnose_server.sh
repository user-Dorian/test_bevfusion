#!/bin/bash
# 诊断服务器代码安装问题

echo "=========================================="
echo "诊断服务器代码安装状态"
echo "=========================================="

# 1. 检查 git 提交
echo ""
echo "[1/5] 检查 git 提交..."
cd ~/workbench/test_bevfusion
git log --oneline -1

# 2. 检查 depth_lss.py 文件内容
echo ""
echo "[2/5] 检查 depth_lss.py 文件..."
echo "=== 第 35-55 行 ==="
sed -n '35,55p' ~/workbench/test_bevfusion/mmdet3d/models/vtransforms/depth_lss.py

# 3. 检查 mmdet3d 安装路径
echo ""
echo "[3/5] 检查 mmdet3d 安装路径..."
python << 'EOF'
import mmdet3d
print("mmdet3d 安装路径:", mmdet3d.__file__)

# 检查实际使用的 depth_lss.py
from mmdet3d.models.vtransforms import DepthLSSTransform
import inspect
source_file = inspect.getfile(DepthLSSTransform)
print("DepthLSSTransform 源文件:", source_file)

# 检查参数
sig = inspect.signature(DepthLSSTransform.__init__)
params = list(sig.parameters.keys())
print("\n参数列表:")
for p in params:
    print(f"  - {p}")

# 检查是否有 **kwargs
has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
print(f"\n是否有 **kwargs: {has_kwargs}")
EOF

# 4. 检查文件内容是否一致
echo ""
echo "[4/5] 比较文件内容..."
echo "=== test_bevfusion 中的 depth_lss.py ==="
md5sum ~/workbench/test_bevfusion/mmdet3d/models/vtransforms/depth_lss.py

echo ""
echo "=== mmdet3d 安装目录中的 depth_lss.py ==="
# 找到实际的安装目录
INSTALL_DIR=$(python -c "import mmdet3d; import os; print(os.path.dirname(mmdet3d.__file__))")
md5sum "$INSTALL_DIR/models/vtransforms/depth_lss.py" 2>/dev/null || echo "文件不存在或路径错误"

# 5. 提供修复建议
echo ""
echo "[5/5] 修复建议..."
echo ""
echo "如果上面的检查显示文件不一致，执行以下命令："
echo ""
echo "cd ~/workbench/bevfusion"
echo "python setup.py develop --uninstall"
echo "python setup.py develop"
echo ""
echo "或者强制复制文件："
echo "cp ~/workbench/test_bevfusion/mmdet3d/models/vtransforms/depth_lss.py ~/workbench/bevfusion/mmdet3d/models/vtransforms/depth_lss.py"
echo ""

echo "=========================================="
echo "诊断完成"
echo "=========================================="
