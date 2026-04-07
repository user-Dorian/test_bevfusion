"""验证配置文件是否正确加载 fuser"""
from mmcv import Config

config_path = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/mini_depth_enhanced.yaml"
cfg = Config.fromfile(config_path)

print("=" * 80)
print("配置文件:", config_path)
print("=" * 80)

# 检查 fuser 配置
if hasattr(cfg, 'model') and 'fuser' in cfg.model:
    print("\n✅ fuser 配置存在:")
    print(cfg.model.fuser)
else:
    print("\n❌ fuser 配置不存在!")
    print("model 键:", cfg.model.keys() if hasattr(cfg, 'model') else "无")

# 检查 _base_
if hasattr(cfg, '_base_'):
    print("\n_base_:", cfg._base_)
else:
    print("\n❌ 没有 _base_ 配置")

print("=" * 80)
