#!/bin/bash

# 脚本：配置数据集和模型软链接
# 功能：读取config.yaml配置，创建模型和数据集的软链接，更新scripts/config.yaml

set -e  # 遇到错误立即退出

echo "🚀 开始配置数据集和模型软链接..."

# 获取脚本所在的目录
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📁 项目根目录: $script_dir"
cd "$script_dir"

# 检查config.yaml是否存在
if [ ! -f "./config.yaml" ]; then
    echo "❌ 错误: 找不到配置文件 ./config.yaml"
    exit 1
fi

# 确保 ~/.local/bin 在 PATH 中（优先检查）
export PATH="$HOME/.local/bin:$PATH"

# 检查yq是否已安装
echo "🔍 检查yq工具..."
if command -v yq >/dev/null 2>&1; then
    echo "✅ yq已安装: $(which yq)"
    # 验证yq版本
    yq_version=$(yq --version 2>/dev/null | head -n1 || echo "未知版本")
    echo "   版本信息: $yq_version"
else
    echo "📦 yq未安装，开始安装..."
    # 创建 bin 目录（如果没有）
    mkdir -p ~/.local/bin
    # 下载 yq，可替换为最新版本
    echo "⬇️  下载yq工具..."
    if wget -q --timeout=30 https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O ~/.local/bin/yq; then
        # 添加执行权限
        chmod +x ~/.local/bin/yq
        echo "✅ yq安装完成"
        # 验证安装
        if command -v yq >/dev/null 2>&1; then
            yq_version=$(yq --version 2>/dev/null | head -n1 || echo "未知版本")
            echo "   安装版本: $yq_version"
        else
            echo "❌ yq安装失败，请检查网络连接或手动安装"
            exit 1
        fi
    else
        echo "❌ yq下载失败，请检查网络连接"
        echo "💡 你可以手动安装yq或使用系统包管理器:"
        echo "   Ubuntu/Debian: sudo apt install yq"
        echo "   CentOS/RHEL: sudo yum install yq"
        exit 1
    fi
fi

# 读取配置文件
echo "📖 读取配置文件..."

# 定义读取配置的函数，处理null值
read_config() {
    local key="$1"
    local value=$(yq e ".\"$key\"" ./config.yaml 2>/dev/null)
    if [ "$value" = "null" ] || [ -z "$value" ]; then
        echo ""
    else
        echo "$value"
    fi
}

# 读取各项配置
DATASETS_FOLDER=$(read_config "datasets-path")
qwen_path=$(read_config "Qwen-path")
blip_path=$(read_config "Blip-path")
clip_path=$(read_config "Clip-path")
conda_env=$(read_config "conda_env")

# 验证必需的配置项
echo "📋 配置信息:"
echo "  - 数据集路径: ${DATASETS_FOLDER:-'❌ 未配置'}"
echo "  - Qwen路径: ${qwen_path:-'⚠️  未配置'}"
echo "  - Blip路径: ${blip_path:-'⚠️  未配置'}"
echo "  - Clip路径: ${clip_path:-'⚠️  未配置'}"
echo "  - Conda环境: ${conda_env:-'❌ 未配置'}"

# 检查必需配置
if [ -z "$DATASETS_FOLDER" ]; then
    echo "❌ 错误: 数据集路径未配置"
    exit 1
fi

if [ -z "$conda_env" ]; then
    echo "❌ 错误: Conda环境未配置"
    exit 1
fi

# 创建必要的目录
echo "📁 创建必要目录..."
mkdir -p ./models
mkdir -p ./fgvr_awc/datasets

# 创建数据集软链接
echo "🔗 创建数据集软链接..."
if [ -L "./datasets" ] || [ -e "./datasets" ]; then
    echo "⚠️  ./datasets 已存在，删除后重新创建..."
    rm -rf ./datasets
fi

if [ -d "$DATASETS_FOLDER" ]; then
    ln -sf "$DATASETS_FOLDER" ./datasets
    echo "✅ 数据集软链接创建成功: ./datasets -> $DATASETS_FOLDER"
else
    echo "❌ 警告: 数据集目录不存在: $DATASETS_FOLDER"
fi

# 创建模型软链接
echo "🔗 创建模型软链接..."

# 定义创建模型软链接的函数
create_model_link() {
    local model_name="$1"
    local model_path="$2"
    local link_name="$3"
    
    # 如果路径为空，跳过创建
    if [ -z "$model_path" ]; then
        echo "⚠️  跳过 $model_name: 路径未配置"
        return 0
    fi
    
    # 删除已存在的软链接或目录
    if [ -L "./models/$link_name" ] || [ -e "./models/$link_name" ]; then
        echo "⚠️  ./models/$link_name 已存在，删除后重新创建..."
        rm -rf "./models/$link_name"
    fi
    
    # 检查源路径是否存在并创建软链接
    if [ -d "$model_path" ]; then
        ln -sf "$model_path" "./models/$link_name"
        echo "✅ $model_name 软链接创建成功: ./models/$link_name -> $model_path"
    else
        echo "❌ 警告: $model_name 目录不存在: $model_path"
    fi
}

# 创建各模型的软链接
create_model_link "Qwen" "$qwen_path" "Qwen"
create_model_link "Blip" "$blip_path" "Blip"
create_model_link "Clip" "$clip_path" "Clip"

# 检查scripts/config.yaml是否存在
echo "⚙️  更新scripts/config.yaml..."
if [ ! -f "./scripts/config.yaml" ]; then
    echo "❌ 错误: 找不到 ./scripts/config.yaml"
    exit 1
fi

# 处理用户名和home目录（包括root用户）
if [ "$USER" = "root" ]; then
    username="root"
    home_path="/root"
    conda_base="/root/miniconda3"
else
    username="$USER"
    home_path="/home/$username"
    conda_base="$home_path/miniconda3"
fi

echo "👤 用户信息:"
echo "  - 用户名: $username"
echo "  - 家目录: $home_path"
echo "  - Conda基础路径: $conda_base"

# 修改./scripts/config.yaml文件
echo "📝 更新配置文件..."
yq e ".environment.conda_env = \"$conda_env\"" -i ./scripts/config.yaml
yq e ".environment.project_root = \"$script_dir\"" -i ./scripts/config.yaml
yq e ".logging.base_dir = \"$script_dir/logs\"" -i ./scripts/config.yaml
yq e ".environment.conda_base = \"$conda_base\"" -i ./scripts/config.yaml

echo "✅ 配置文件更新完成"

# 验证软链接
echo "🔍 验证软链接状态..."
echo "数据集软链接:"
if [ -L "./datasets" ]; then
    echo "  ✅ ./datasets -> $(readlink ./datasets)"
else
    echo "  ❌ ./datasets 软链接不存在"
fi

echo "模型软链接:"
for model in Qwen Blip Clip; do
    if [ -L "./models/$model" ]; then
        echo "  ✅ ./models/$model -> $(readlink ./models/$model)"
    else
        echo "  ❌ ./models/$model 软链接不存在"
    fi
done

echo "🎉 配置完成！"
