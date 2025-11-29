#!/bin/bash

# 超参数设置脚本
# 用法: ./set_hyperparameters.sh [OPTIONS]
# 
# 选项:
#   --experience_number <num>  直接设置经验条数
#   --config                   从scripts/config.yaml读取配置并设置
#   --help                     显示帮助信息
# 
# 示例:
#   ./set_hyperparameters.sh --experience_number 10
#   ./set_hyperparameters.sh --config

# 默认值
EXPERIENCE_NUMBER=""
USE_CONFIG=false

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
HYPERPARAM_FILE="$PROJECT_ROOT/configs/hyperparameters.yaml"

# 帮助信息
show_help() {
    echo "超参数设置脚本"
    echo "用法: $0 [OPTIONS]"
    echo ""
    echo "选项:"
    echo "  --experience_number <num>  直接设置经验条数"
    echo "  --config                   从scripts/config.yaml读取配置并设置"
    echo "  --help                     显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --experience_number 10"
    echo "  $0 --config"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --experience_number)
            EXPERIENCE_NUMBER="$2"
            shift 2
            ;;
        --config)
            USE_CONFIG=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo "错误: 未知选项 $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查参数
if [[ "$USE_CONFIG" == false && -z "$EXPERIENCE_NUMBER" ]]; then
    echo "错误: 必须指定 --experience_number 或 --config 选项"
    echo "使用 --help 查看帮助信息"
    exit 1
fi

# 检查文件是否存在
if [[ ! -f "$HYPERPARAM_FILE" ]]; then
    echo "错误: 超参数配置文件不存在: $HYPERPARAM_FILE"
    exit 1
fi

if [[ "$USE_CONFIG" == true && ! -f "$CONFIG_FILE" ]]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 获取经验条数
get_experience_number() {
    local number=""
    
    if [[ "$USE_CONFIG" == true ]]; then
        # 从config.yaml读取
        echo "正在从 $CONFIG_FILE 读取配置..." >&2
        
        # 检查是否安装了yq或python
        if command -v yq >/dev/null 2>&1; then
            number=$(yq e '.hyperparameters.experience_base_max_number' "$CONFIG_FILE")
        elif command -v python3 >/dev/null 2>&1; then
            number=$(python3 -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    result = config.get('hyperparameters', {}).get('experience_base_max_number', 8)
    print(result)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
")
        else
            echo "错误: 需要安装 yq 或 python3 来解析YAML文件" >&2
            exit 1
        fi
        
        if [[ -z "$number" || "$number" == "None" ]]; then
            echo "错误: 无法从配置文件读取 experience_base_max_number" >&2
            exit 1
        fi
        
        echo "从配置文件读取到经验条数: $number" >&2
    else
        number="$EXPERIENCE_NUMBER"
        echo "使用命令行指定的经验条数: $number" >&2
    fi
    
    # 验证数字
    if ! [[ "$number" =~ ^[0-9]+$ ]] || [[ "$number" -lt 1 ]]; then
        echo "错误: 经验条数必须是正整数，当前值: $number" >&2
        exit 1
    fi
    
    echo "$number"
}

# 更新超参数文件
update_hyperparameter() {
    local number="$1"
    
    echo "正在更新超参数文件: $HYPERPARAM_FILE"
    
    # 更新文件
    if command -v yq >/dev/null 2>&1; then
        # 使用yq更新
        yq e ".experience_base_max_number = $number" -i "$HYPERPARAM_FILE"
    elif command -v python3 >/dev/null 2>&1; then
        # 使用python更新
        python3 -c "
import yaml
import sys

try:
    with open('$HYPERPARAM_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    config['experience_base_max_number'] = $number
    
    with open('$HYPERPARAM_FILE', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print('成功更新超参数文件')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
"
    else
        echo "错误: 需要安装 yq 或 python3 来更新YAML文件"
        exit 1
    fi
    
    # 验证更新
    local updated_number
    if command -v yq >/dev/null 2>&1; then
        updated_number=$(yq e '.experience_base_max_number' "$HYPERPARAM_FILE")
    else
        updated_number=$(python3 -c "
import yaml
with open('$HYPERPARAM_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('experience_base_max_number', 'Not found'))
")
    fi
    
    if [[ "$updated_number" == "$number" ]]; then
        echo "✅ 超参数更新成功: experience_base_max_number = $number"
    else
        echo "❌ 超参数更新失败"
        echo "期望值: $number, 实际值: $updated_number"
        exit 1
    fi
}

# 主函数
main() {
    echo "=== 超参数设置脚本 ==="
    echo "项目根目录: $PROJECT_ROOT"
    echo "超参数文件: $HYPERPARAM_FILE"
    
    if [[ "$USE_CONFIG" == true ]]; then
        echo "配置文件: $CONFIG_FILE"
    fi
    
    echo ""
    
    # 获取经验条数
    local experience_number
    experience_number=$(get_experience_number)
    
    if [[ $? -ne 0 ]]; then
        echo "错误: 获取经验条数失败"
        exit 1
    fi
    
    echo ""
    
    # 更新超参数文件
    update_hyperparameter "$experience_number"
    
    echo ""
    echo "=== 超参数设置完成 ==="
}

# 运行主函数
main