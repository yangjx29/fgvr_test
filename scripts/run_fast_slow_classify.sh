#!/bin/bash

# =============================================================================
# 快慢思考分类脚本 - Fast-Slow Thinking Classification
# =============================================================================
# 用法：直接运行 bash run_fast_slow_classify.sh
# 专门用于快慢思考分类阶段，基于推理结果进行分类，所有参数从 config.yaml 文件读取

# =============================================================================
# YAML配置读取函数
# =============================================================================

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

# 检查配置文件是否存在
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "错误: 配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi

# 简化的YAML解析函数
get_yaml_value() {
    local key=$1
    local file=$2
    grep "^[[:space:]]*${key}:" "$file" | sed 's/^[[:space:]]*[^:]*:[[:space:]]*//' | sed 's/[[:space:]]*#.*//' | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/"
}

# 读取配置参数
CUDA_VISIBLE_DEVICES_VALUE=$(get_yaml_value "cuda_visible_devices" "${CONFIG_FILE}")
DATASET_NAME=$(get_yaml_value "name" "${CONFIG_FILE}")
CONDA_ENV_VALUE=$(get_yaml_value "conda_env" "${CONFIG_FILE}")
CONDA_BASE_VALUE=$(get_yaml_value "conda_base" "${CONFIG_FILE}")
PROJECT_ROOT_VALUE=$(get_yaml_value "project_root" "${CONFIG_FILE}")
LOG_BASE_DIR_VALUE=$(get_yaml_value "base_dir" "${CONFIG_FILE}")

# =============================================================================
# 从配置文件读取参数
# =============================================================================

# GPU设置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"

# 数据集配置
DATASET="${DATASET_NAME}"

# 环境配置
CONDA_ENV="${CONDA_ENV_VALUE}"
CONDA_BASE="${CONDA_BASE_VALUE}"
PROJECT_ROOT="${PROJECT_ROOT_VALUE}"

# 日志配置
LOG_BASE_DIR="${LOG_BASE_DIR_VALUE}"

# 数据集映射配置
case "${DATASET}" in
    "dog")
        DATASET_NUM="120"
        CONFIG_FILE="dog120_all.yml"
        ;;
    "bird")
        DATASET_NUM="200"
        CONFIG_FILE="bird200_all.yml"
        DATASET_DIR="CUB_200_2011/CUB_200_2011"  # 鸟类数据集特殊路径
        ;;
    "flower")
        DATASET_NUM="102"
        CONFIG_FILE="flower102_all.yml"
        ;;
    "pet")
        DATASET_NUM="37"
        CONFIG_FILE="pet37_all.yml"
        ;;
    "car")
        DATASET_NUM="196"
        CONFIG_FILE="car196_all.yml"
        ;;
    *)
        echo "错误: 不支持的数据集 '${DATASET}'"
        echo "支持的数据集: dog, bird, flower, pet, car"
        exit 1
        ;;
esac

# 生成路径
INFER_DIR="./experiments/${DATASET}${DATASET_NUM}/infer"
CLASSIFY_DIR="./experiments/${DATASET}${DATASET_NUM}/classify"
LOG_DIR="${LOG_BASE_DIR}/fast_slow_classify/${DATASET}${DATASET_NUM}"

# 生成递增编号的日志文件名函数
generate_log_filename() {
    local base_name=$1
    local log_dir=$2
    local base_file="${log_dir}/${base_name}.log"
    
    if [ ! -f "${base_file}" ]; then
        echo "${base_file}"
        return
    fi
    
    local counter=1
    while [ -f "${log_dir}/${base_name}(${counter}).log" ]; do
        counter=$((counter + 1))
    done
    
    echo "${log_dir}/${base_name}(${counter}).log"
}

LOG_FILE=$(generate_log_filename "fast_slow_classify_${DATASET}" "${LOG_DIR}")

# =============================================================================
# 脚本执行区域 - SCRIPT EXECUTION SECTION
# =============================================================================

# 颜色输出函数
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查项目根目录是否存在
if [ ! -d "${PROJECT_ROOT}" ]; then
    print_error "项目根目录不存在: ${PROJECT_ROOT}"
    print_error "请检查PROJECT_ROOT配置是否正确"
    exit 1
fi

# 切换到项目根目录
print_info "切换到项目根目录: ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}" || {
    print_error "无法切换到项目根目录: ${PROJECT_ROOT}"
    exit 1
}

# 检查并创建必要目录
print_info "创建必要目录..."
mkdir -p "${LOG_DIR}"
mkdir -p "${CLASSIFY_DIR}"

# 检查推理结果目录是否存在
if [ ! -d "${INFER_DIR}" ]; then
    print_error "推理结果目录不存在: ${INFER_DIR}"
    print_info "请先运行 run_fast_slow_infer.sh 生成推理结果"
    exit 1
fi

# 检查推理结果文件是否存在
INFER_FILE_COUNT=$(find "${INFER_DIR}" -name "*.json" | wc -l)
if [ "${INFER_FILE_COUNT}" -eq 0 ]; then
    print_error "推理结果目录中没有找到JSON文件: ${INFER_DIR}"
    print_info "请先运行 run_fast_slow_infer.sh 生成推理结果"
    exit 1
fi

# 打印配置信息
print_info "=== 运行配置 ==="
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "数据集: ${DATASET}"
echo "配置文件: ${CONFIG_FILE}"
echo "运行模式: fast_slow_classify"
echo "推理结果目录: ${INFER_DIR}"
echo "分类结果目录: ${CLASSIFY_DIR}"
echo "推理文件数量: ${INFER_FILE_COUNT}"
echo "日志文件: ${LOG_FILE}"
echo "虚拟环境: ${CONDA_ENV}"
print_info "================"

# 检查conda环境是否存在
if [ ! -d "${CONDA_BASE}/envs/${CONDA_ENV}" ]; then
    print_error "Conda环境不存在: ${CONDA_BASE}/envs/${CONDA_ENV}"
    print_error "请检查CONDA_ENV和CONDA_BASE配置是否正确"
    exit 1
fi

# 构建命令
CMD="source /home/hdl/miniconda3/envs/${CONDA_ENV}/bin/activate && python discovering.py \
    --mode=fast_slow_classify \
    --config_file_env=./configs/env_machine.yml \
    --config_file_expt=./configs/expts/${CONFIG_FILE} \
    --infer_dir=${INFER_DIR} \
    --classify_dir=${CLASSIFY_DIR}"

# 创建启动脚本
TEMP_SCRIPT="/tmp/run_fast_slow_classify_${DATASET}_$$.sh"
cat > "${TEMP_SCRIPT}" << EOF
#!/bin/bash
${CMD}
EOF
chmod +x "${TEMP_SCRIPT}"

# 激活conda环境并运行
print_info "激活虚拟环境并开始运行..."
print_info "日志将实时写入: ${LOG_FILE}"
print_info "可以使用 'tail -f ${LOG_FILE}' 查看实时日志"

# 后台运行并记录日志
print_info "开始后台运行..."
nohup bash "${TEMP_SCRIPT}" > "${LOG_FILE}" 2>&1 &
PID=$!

print_success "任务已启动！"
print_info "进程ID: ${PID}"
print_info "日志文件: ${LOG_FILE}"
print_info "分类结果目录: ${CLASSIFY_DIR}"
print_info "查看实时日志: tail -f ${LOG_FILE}"
print_info "停止任务: kill ${PID}"

# 等待几秒钟检查进程是否正常启动
sleep 3
if kill -0 ${PID} 2>/dev/null; then
    print_success "进程运行正常"
else
    print_error "进程可能已经退出，请检查日志文件"
fi

# 清理临时文件
print_info "清理临时文件..."
rm -f "${TEMP_SCRIPT}"

print_info "脚本执行完成"
