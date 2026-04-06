#!/bin/bash
# 服务器修复脚本 - 确保代码正确更新

echo "=========================================="
echo "开始修复服务器代码..."
echo "=========================================="

# 1. 进入 test_bevfusion 目录
cd ~/workbench/test_bevfusion

# 2. 拉取最新代码
echo "[1/4] 拉取最新代码..."
git pull origin main

# 3. 检查 depth_lss.py 文件
echo ""
echo "[2/4] 检查 depth_lss.py 文件..."
if grep -q "use_points: str = 'lidar'" mmdet3d/models/vtransforms/depth_lss.py; then
    echo "✓ depth_lss.py 已正确更新（包含 use_points 参数）"
else
    echo "✗ depth_lss.py 未更新，手动修复..."
    # 备份原文件
    cp mmdet3d/models/vtransforms/depth_lss.py mmdet3d/models/vtransforms/depth_lss.py.bak
    
    # 使用 sed 添加 use_points 参数
    sed -i "s/downsample: int = 1,/downsample: int = 1,\n        use_points: str = 'lidar',/" mmdet3d/models/vtransforms/depth_lss.py
    
    # 检查是否成功
    if grep -q "use_points: str = 'lidar'" mmdet3d/models/vtransforms/depth_lss.py; then
        echo "✓ depth_lss.py 修复成功"
    else
        echo "✗ depth_lss.py 修复失败，需要手动处理"
        exit 1
    fi
fi

# 4. 重新安装 mmdet3d
echo ""
echo "[3/4] 重新安装 mmdet3d..."
cd ~/workbench/bevfusion
python setup.py develop --uninstall
python setup.py develop

# 5. 验证安装
echo ""
echo "[4/4] 验证安装..."
python -c "from mmdet3d.models.vtransforms import DepthLSSTransform; import inspect; sig = inspect.signature(DepthLSSTransform.__init__); print('DepthLSSTransform 参数:', list(sig.parameters.keys()))"

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
echo ""
echo "现在可以运行训练："
echo "torchpack dist-run -np 4 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/mini_dataset_optimized.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/lidar-only-det.pth --run-dir train_result_mini_optimized"
