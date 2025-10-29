#!/bin/bash

# =============================================================================
# 快慢思考分离流程脚本 - Fast-Slow Thinking Separated Pipeline
# =============================================================================
# 用法：
#   前台运行（显示启动信息）: bash run_fast_slow_pipeline.sh
#   完全后台运行: bash run_fast_slow_pipeline.sh --background
# 先执行推理阶段，然后执行分类阶段，所有参数从 config.yaml 文件读取

# 检查是否为后台运行模式
BACKGROUND_MODE=false
if [[ "$1" == "--background" ]]; then
    BACKGROUND_MODE=true
fi

# 如果是后台运行模式，将整个脚本重新启动为后台进程
if [[ "${BACKGROUND_MODE}" == "true" ]] && [[ "${PIPELINE_BACKGROUND_STARTED}" != "true" ]]; then
    export PIPELINE_BACKGROUND_STARTED=true
    
    # 获取脚本目录和配置
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CONFIG_FILE="${SCRIPT_DIR}/config.yaml"
    
    # 读取数据集名称用于日志文件名
    DATASET_NAME=$(grep "^[[:space:]]*name:" "$CONFIG_FILE" | sed 's/^[[:space:]]*[^:]*:[[:space:]]*//' | sed 's/[[:space:]]*#.*//' | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/")
    LOG_BASE_DIR=$(grep "^[[:space:]]*base_dir:" "$CONFIG_FILE" | sed 's/^[[:space:]]*[^:]*:[[:space:]]*//' | sed 's/[[:space:]]*#.*//' | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/")
    
    # 确定数据集编号
    case "${DATASET_NAME}" in
        "dog") DATASET_NUM="120" ;;
        "bird") DATASET_NUM="200" ;;
        "flower") DATASET_NUM="102" ;;
        "pet") DATASET_NUM="37" ;;
        "car") DATASET_NUM="196" ;;
        *) DATASET_NUM="" ;;
    esac
    
    LOG_DIR="${LOG_BASE_DIR}/fast_slow_pipeline/${DATASET_NAME}${DATASET_NUM}"
    mkdir -p "${LOG_DIR}"
    
    # 生成后台日志文件名
    BACKGROUND_LOG="${LOG_DIR}/fast_slow_pipeline_${DATASET_NAME}_background.log"
    
    echo "🚀 启动后台快慢思考分离流程任务..."
    echo "📋 数据集: ${DATASET_NAME}${DATASET_NUM}"
    echo "📁 日志目录: ${LOG_DIR}"
    echo "📄 后台日志: ${BACKGROUND_LOG}"
    echo ""
    
    # 启动后台进程
    nohup bash "$0" >> "${BACKGROUND_LOG}" 2>&1 &
    BACKGROUND_PID=$!
    
    echo "✅ 后台任务已启动！"
    echo "🔢 进程ID: ${BACKGROUND_PID}"
    echo "📝 查看实时日志: tail -f ${BACKGROUND_LOG}"
    echo "🛑 停止任务: kill ${BACKGROUND_PID}"
    echo ""
    echo "💡 提示: 脚本将在后台完整执行推理和分类流程"
    
    exit 0
fi

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
TEST_DATA_SUFFIX_VALUE=$(get_yaml_value "test_data_suffix" "${CONFIG_FILE}")
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
TEST_DATA_SUFFIX="${TEST_DATA_SUFFIX_VALUE}"

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
        DATASET_DIR="dogs_120"
        ;;
    "bird")
        DATASET_NUM="200"
        CONFIG_FILE="bird200_all.yml"
        DATASET_DIR="CUB_200_2011"
        ;;
    "flower")
        DATASET_NUM="102"
        CONFIG_FILE="flower102_all.yml"
        DATASET_DIR="flowers_102"
        ;;
    "pet")
        DATASET_NUM="37"
        CONFIG_FILE="pet37_all.yml"
        DATASET_DIR="pet_37"
        ;;
    "car")
        DATASET_NUM="196"
        CONFIG_FILE="car196_all.yml"
        DATASET_DIR="car_196"
        ;;
    *)
        echo "错误: 不支持的数据集 '${DATASET}'"
        echo "支持的数据集: dog, bird, flower, pet, car"
        exit 1
        ;;
esac

# 生成路径
KNOWLEDGE_BASE_DIR="./experiments/${DATASET}${DATASET_NUM}/knowledge_base"
TEST_DATA_DIR="./datasets/${DATASET_DIR}/images_discovery_all_${TEST_DATA_SUFFIX}"
INFER_DIR="./experiments/${DATASET}${DATASET_NUM}/infer"
CLASSIFY_DIR="./experiments/${DATASET}${DATASET_NUM}/classify"
LOG_DIR="${LOG_BASE_DIR}/fast_slow_pipeline/${DATASET}${DATASET_NUM}"

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

# 统一使用单一日志文件
PIPELINE_LOG=$(generate_log_filename "fast_slow_pipeline_${DATASET}" "${LOG_DIR}")

# 等待时间配置（秒）
INFERENCE_TIMEOUT=7200          # 推理阶段超时时间（2小时）
CLASSIFICATION_TIMEOUT=1800     # 分类阶段超时时间（30分钟）
CHECK_INTERVAL=30               # 检查间隔时间（30秒）

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
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${PIPELINE_LOG}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${PIPELINE_LOG}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "${PIPELINE_LOG}"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${PIPELINE_LOG}"
}

# 后台运行模式：仅写入日志，不输出到控制台
log_info() {
    echo -e "[INFO] $1" >> "${PIPELINE_LOG}"
}

log_success() {
    echo -e "[SUCCESS] $1" >> "${PIPELINE_LOG}"
}

log_warning() {
    echo -e "[WARNING] $1" >> "${PIPELINE_LOG}"
}

log_error() {
    echo -e "[ERROR] $1" >> "${PIPELINE_LOG}"
}

# 检查进程是否运行
check_process() {
    local pid=$1
    if kill -0 ${pid} 2>/dev/null; then
        return 0  # 进程运行中
    else
        return 1  # 进程已结束
    fi
}

# 等待进程完成（后台运行模式）
wait_for_process() {
    local pid=$1
    local timeout=$2
    local description=$3
    local elapsed=0
    
    log_info "等待${description}完成 (PID: ${pid}, 超时: ${timeout}秒)..."
    
    while [ ${elapsed} -lt ${timeout} ]; do
        if ! check_process ${pid}; then
            log_success "${description}已完成"
            return 0
        fi
        
        sleep ${CHECK_INTERVAL}
        elapsed=$((elapsed + CHECK_INTERVAL))
        log_info "${description}运行中... (已运行${elapsed}秒)"
    done
    
    log_error "${description}超时，强制终止进程"
    kill ${pid} 2>/dev/null
    return 1
}

# 检查并创建必要目录
print_info "=== 快慢思考分离流程开始 ==="

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

print_info "创建必要目录..."
mkdir -p "${LOG_DIR}"
mkdir -p "${INFER_DIR}"
mkdir -p "${CLASSIFY_DIR}"

# 打印配置信息
print_info "=== 运行配置 ==="
echo "GPU: ${CUDA_VISIBLE_DEVICES}" | tee -a "${PIPELINE_LOG}"
echo "数据集: ${DATASET}${DATASET_NUM}" | tee -a "${PIPELINE_LOG}"
echo "配置文件: ${CONFIG_FILE}" | tee -a "${PIPELINE_LOG}"
echo "流程模式: fast_slow_pipeline (推理 + 分类)" | tee -a "${PIPELINE_LOG}"
echo "知识库目录: ${KNOWLEDGE_BASE_DIR}" | tee -a "${PIPELINE_LOG}"
echo "测试数据: ${TEST_DATA_DIR}" | tee -a "${PIPELINE_LOG}"
echo "推理结果目录: ${INFER_DIR}" | tee -a "${PIPELINE_LOG}"
echo "分类结果目录: ${CLASSIFY_DIR}" | tee -a "${PIPELINE_LOG}"
echo "虚拟环境: ${CONDA_ENV}" | tee -a "${PIPELINE_LOG}"
print_info "================"

# =============================================================================
# 阶段1: 推理阶段 (fast_slow_infer)
# =============================================================================
print_info "=== 阶段1: 快慢思考推理 ==="

# 检查conda环境是否存在
if [ ! -d "${CONDA_BASE}/envs/${CONDA_ENV}" ]; then
    print_error "Conda环境不存在: ${CONDA_BASE}/envs/${CONDA_ENV}"
    print_error "请检查CONDA_ENV和CONDA_BASE配置是否正确"
    exit 1
fi

# 检查知识库是否存在
if [ ! -d "${KNOWLEDGE_BASE_DIR}" ]; then
    print_error "知识库目录不存在: ${KNOWLEDGE_BASE_DIR}"
    print_info "请先运行 run_build_knowledge_base.sh 构建知识库"
    exit 1
fi

# 检查测试数据目录
if [ ! -d "${TEST_DATA_DIR}" ]; then
    print_error "测试数据目录不存在: ${TEST_DATA_DIR}"
    exit 1
fi

# 检查推理结果是否已存在
EXISTING_INFER_COUNT=$(find "${INFER_DIR}" -name "*.json" 2>/dev/null | wc -l)
if [ "${EXISTING_INFER_COUNT}" -gt 0 ]; then
    print_info "推理结果已存在 (${EXISTING_INFER_COUNT} 个文件)，跳过推理阶段"
else
    print_info "推理结果不存在，开始推理阶段..."
    
    # 构建推理命令
    INFERENCE_CMD="export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} && \
    source ${CONDA_BASE}/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV} && \
    cd ${PROJECT_ROOT} && \
    python discovering.py \
        --mode=fast_slow_infer \
        --config_file_env=./configs/env_machine.yml \
        --config_file_expt=./configs/expts/${CONFIG_FILE} \
        --test_data_dir=${TEST_DATA_DIR} \
        --knowledge_base_dir=${KNOWLEDGE_BASE_DIR} \
        --infer_dir=${INFER_DIR}"
    
    # 创建推理启动脚本
    INFERENCE_SCRIPT="/tmp/run_inference_${DATASET}_$$.sh"
    cat > "${INFERENCE_SCRIPT}" << EOF
#!/bin/bash
${INFERENCE_CMD}
EOF
    chmod +x "${INFERENCE_SCRIPT}"
    
    print_info "启动推理阶段..." | tee -a "${PIPELINE_LOG}"
    print_info "统一日志: ${PIPELINE_LOG}" | tee -a "${PIPELINE_LOG}"
    nohup bash "${INFERENCE_SCRIPT}" >> "${PIPELINE_LOG}" 2>&1 &
    INFERENCE_PID=$!
    
    log_info "推理进程ID: ${INFERENCE_PID}"
    
    # 等待推理完成
    if wait_for_process ${INFERENCE_PID} ${INFERENCE_TIMEOUT} "推理阶段"; then
        log_success "推理阶段完成"
        
        # 检查推理结果文件是否生成
        INFER_FILE_COUNT=$(find "${INFER_DIR}" -name "*.json" | wc -l)
        if [ "${INFER_FILE_COUNT}" -gt 0 ]; then
            log_success "推理结果文件生成成功: ${INFER_DIR} (${INFER_FILE_COUNT} 个文件)"
        else
            log_error "推理结果文件未生成，检查日志: ${PIPELINE_LOG}"
            exit 1
        fi
    else
        log_error "推理阶段失败，检查日志: ${PIPELINE_LOG}"
        exit 1
    fi
    
    # 清理临时文件
    rm -f "${INFERENCE_SCRIPT}"
fi

# =============================================================================
# 阶段2: 分类阶段 (fast_slow_classify)
# =============================================================================
print_info "=== 阶段2: 快慢思考分类 ===" 

# 检查推理结果文件
INFER_FILE_COUNT=$(find "${INFER_DIR}" -name "*.json" | wc -l)
if [ "${INFER_FILE_COUNT}" -eq 0 ]; then
    print_error "推理结果目录中没有找到JSON文件: ${INFER_DIR}"
    exit 1
fi

# 构建分类命令
CLASSIFICATION_CMD="export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} && \
source ${CONDA_BASE}/etc/profile.d/conda.sh && \
conda activate ${CONDA_ENV} && \
cd ${PROJECT_ROOT} && \
python discovering.py \
    --mode=fast_slow_classify \
    --config_file_env=./configs/env_machine.yml \
    --config_file_expt=./configs/expts/${CONFIG_FILE} \
    --infer_dir=${INFER_DIR} \
    --classify_dir=${CLASSIFY_DIR}"

# 创建分类启动脚本
CLASSIFICATION_SCRIPT="/tmp/run_classification_${DATASET}_$$.sh"
cat > "${CLASSIFICATION_SCRIPT}" << EOF
#!/bin/bash
${CLASSIFICATION_CMD}
EOF
chmod +x "${CLASSIFICATION_SCRIPT}"

print_info "启动分类阶段..." | tee -a "${PIPELINE_LOG}"
echo "=== 阶段2: 快慢思考分类开始 ===" >> "${PIPELINE_LOG}"
nohup bash "${CLASSIFICATION_SCRIPT}" >> "${PIPELINE_LOG}" 2>&1 &
CLASSIFICATION_PID=$!

log_info "分类进程ID: ${CLASSIFICATION_PID}"

# 等待分类完成
if wait_for_process ${CLASSIFICATION_PID} ${CLASSIFICATION_TIMEOUT} "分类阶段"; then
    log_success "分类阶段完成"
    
    # 检查结果文件是否生成
    RESULTS_FILE="${CLASSIFY_DIR}/classification_results.json"
    if [ -f "${RESULTS_FILE}" ]; then
        log_success "分类结果文件生成成功: ${RESULTS_FILE}"
    else
        log_warning "分类结果文件未生成，但分类已完成"
    fi
else
    log_error "分类阶段失败，检查日志: ${PIPELINE_LOG}"
    exit 1
fi

# 清理临时文件
rm -f "${CLASSIFICATION_SCRIPT}"

# =============================================================================
# 完成总结
# =============================================================================
# 记录完成信息到日志
echo "=== 快慢思考分离流程完成 ===" >> "${PIPELINE_LOG}"
echo "总结信息:" >> "${PIPELINE_LOG}"
echo "- 统一日志文件: ${PIPELINE_LOG}" >> "${PIPELINE_LOG}"
echo "- 推理结果目录: ${INFER_DIR}" >> "${PIPELINE_LOG}"
echo "- 分类结果目录: ${CLASSIFY_DIR}" >> "${PIPELINE_LOG}"
echo "- 推理文件数量: ${INFER_FILE_COUNT}" >> "${PIPELINE_LOG}"
echo "- 执行时间: $(date)" >> "${PIPELINE_LOG}"

print_success "=== 快慢思考分离流程完成 ==="
print_info "统一日志文件: ${PIPELINE_LOG}"
print_info "推理结果目录: ${INFER_DIR}"
print_info "分类结果目录: ${CLASSIFY_DIR}"
print_info "查看最终结果: tail -20 ${PIPELINE_LOG}"

print_success "快慢思考分离流程脚本执行完成！"
