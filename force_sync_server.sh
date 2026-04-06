#!/bin/bash
# 强制同步服务器代码脚本

echo "=========================================="
echo "强制同步服务器代码"
echo "=========================================="

# 1. 检查当前目录
echo "当前目录：$(pwd)"
echo "工作目录：~/workbench"
echo ""

# 2. 进入 test_bevfusion
cd ~/workbench/test_bevfusion
echo "进入 test_bevfusion: $(pwd)"

# 3. 拉取最新代码
echo ""
echo "[1/5] 拉取最新代码..."
git pull origin main

# 4. 检查 depth_lss.py
echo ""
echo "[2/5] 检查 depth_lss.py..."
echo "检查 use_points 参数是否存在..."
if grep -q "use_points: str = 'lidar'" mmdet3d/models/vtransforms/depth_lss.py; then
    echo "✓ depth_lss.py 已包含 use_points 参数"
    grep -A 2 "use_points: str" mmdet3d/models/vtransforms/depth_lss.py | head -3
else
    echo "✗ depth_lss.py 缺少 use_points 参数，正在修复..."
    
    # 显示当前文件内容
    echo "当前文件内容（第 40-55 行）:"
    sed -n '40,55p' mmdet3d/models/vtransforms/depth_lss.py
    
    # 创建临时修复文件
    cat > /tmp/depth_lss_fix.py << 'PYTHON_EOF'
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
        use_points: str = 'lidar',
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
            use_points=use_points,
            depth_input=depth_input,
            add_depth_features=add_depth_features,
            height_expand=height_expand,
        )
        self.use_attention = use_attention
PYTHON_EOF
    
    echo "临时修复文件已创建：/tmp/depth_lss_fix.py"
fi

# 5. 检查 default.yaml 配置
echo ""
echo "[3/5] 检查 default.yaml 配置..."
if grep -q "use_points: lidar" configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/default.yaml; then
    echo "✓ default.yaml 已包含 use_points: lidar"
else
    echo "✗ default.yaml 缺少 use_points 参数"
fi

# 6. 重新安装 mmdet3d
echo ""
echo "[4/5] 重新安装 mmdet3d..."
cd ~/workbench/bevfusion
echo "当前目录：$(pwd)"

# 卸载
echo "卸载旧版本..."
python setup.py develop --uninstall 2>&1 | tail -2

# 清理缓存
echo "清理缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# 重新安装
echo "安装新版本..."
python setup.py develop 2>&1 | tail -5

# 7. 验证安装
echo ""
echo "[5/5] 验证安装..."
python << 'PYTHON_EOF'
from mmdet3d.models.vtransforms import DepthLSSTransform
import inspect

sig = inspect.signature(DepthLSSTransform.__init__)
params = list(sig.parameters.keys())

print("\nDepthLSSTransform.__init__ 参数列表:")
for param in params:
    print(f"  - {param}")

if 'use_points' in params:
    print("\n✓ 验证成功：use_points 参数存在")
else:
    print("\n✗ 验证失败：use_points 参数不存在")
    print("当前参数:", params)
PYTHON_EOF

echo ""
echo "=========================================="
echo "同步完成！"
echo "=========================================="
echo ""
echo "现在可以运行训练："
echo "torchpack dist-run -np 4 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/lidar-only-det.pth --run-dir train_result"
