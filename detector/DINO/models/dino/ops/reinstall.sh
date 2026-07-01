#!/bin/bash
# 清理并重新安装 MultiScaleDeformableAttention CUDA 扩展

set -e

OPS_DIR="/home/wh1234_/code/Counting/detector/DINO/models/dino/ops"
cd "$OPS_DIR"

echo "=========================================="
echo "清理并重新安装 MultiScaleDeformableAttention"
echo "=========================================="

# 1. 检查 PyTorch 版本
echo "检查 PyTorch 版本..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 2. 卸载旧版本
echo ""
echo "卸载旧版本..."
pip uninstall -y MultiScaleDeformableAttention 2>/dev/null || true
rm -rf build/ dist/ *.egg-info
rm -rf ~/.local/lib/python3.8/site-packages/MultiScaleDeformableAttention* 2>/dev/null || true
rm -rf ~/.conda/envs/dino/lib/python3.8/site-packages/MultiScaleDeformableAttention* 2>/dev/null || true

# 3. 清理 Python 缓存
echo "清理 Python 缓存..."
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# 4. 重新编译安装
echo ""
echo "重新编译安装..."
python setup.py clean --all
python setup.py build_ext --inplace
python setup.py install

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "测试安装..."
python -c "import MultiScaleDeformableAttention as MSDA; print('✓ 导入成功')"

