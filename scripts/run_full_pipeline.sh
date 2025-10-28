#!/bin/bash

# =============================================================================
# 全流程运行脚本 - Full Pipeline Runner (Knowledge Base + Fast-Slow Evaluation)
# =============================================================================
# 用法：直接运行 bash run_full_pipeline.sh
# 先构建知识库，然后进行快慢思考系统评估，所有参数从 config.yaml 文件读取

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
KSHOT_VALUE=$(get_yaml_value "kshot" "${CONFIG_FILE}")
EVAL_MODE_VALUE=$(get_yaml_value "eval_mode" "${CONFIG_FILE}")
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
KSHOT="${KSHOT_VALUE}"
EVAL_MODE="${EVAL_MODE_VALUE}"

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
PIPELINE_MODE="knowledge_base"
KNOWLEDGE_BASE_DIR="./experiments/${DATASET}${DATASET_NUM}/knowledge_base"
TEST_DATA_DIR="./datasets/${DATASET_DIR}/images_discovery_all_${TEST_DATA_SUFFIX}"
RESULTS_OUT="./results/${DATASET}_${EVAL_MODE}_pipeline_results.json"
LOG_DIR="${LOG_BASE_DIR}/full_pipeline/${DATASET}${DATASET_NUM}"

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
PIPELINE_LOG=$(generate_log_filename "full_pipeline_${DATASET}" "${LOG_DIR}")

# 等待时间配置（秒）
KNOWLEDGE_BASE_TIMEOUT=3600     # 知识库构建超时时间（1小时）
EVALUATION_TIMEOUT=3600         # 评估阶段超时时间（60分钟）
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
print_info "=== 全流程运行开始 ==="

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
mkdir -p "$(dirname "${RESULTS_OUT}")"

# 打印配置信息
print_info "=== 运行配置 ==="
echo "GPU: ${CUDA_VISIBLE_DEVICES}" | tee -a "${PIPELINE_LOG}"
echo "数据集: ${DATASET}${DATASET_NUM}" | tee -a "${PIPELINE_LOG}"
echo "配置文件: ${CONFIG_FILE}" | tee -a "${PIPELINE_LOG}"
echo "流程模式: ${PIPELINE_MODE}" | tee -a "${PIPELINE_LOG}"
echo "K-shot: ${KSHOT}" | tee -a "${PIPELINE_LOG}"
echo "评估模式: ${EVAL_MODE}" | tee -a "${PIPELINE_LOG}"
echo "知识库目录: ${KNOWLEDGE_BASE_DIR}" | tee -a "${PIPELINE_LOG}"
echo "测试数据: ${TEST_DATA_DIR}" | tee -a "${PIPELINE_LOG}"
echo "结果输出: ${RESULTS_OUT}" | tee -a "${PIPELINE_LOG}"
echo "虚拟环境: ${CONDA_ENV}" | tee -a "${PIPELINE_LOG}"
print_info "================"

# =============================================================================
# 阶段1: 构建知识库
# =============================================================================
print_info "=== 阶段1: 构建知识库 ==="

# 检查conda环境是否存在
if [ ! -d "${CONDA_BASE}/envs/${CONDA_ENV}" ]; then
    print_error "Conda环境不存在: ${CONDA_BASE}/envs/${CONDA_ENV}"
    print_error "请检查CONDA_ENV和CONDA_BASE配置是否正确"
    exit 1
fi

# 检查知识库是否已存在
if [ -d "${KNOWLEDGE_BASE_DIR}" ] && [ -f "${KNOWLEDGE_BASE_DIR}/image_knowledge_base.json" ]; then
    print_info "知识库已存在，跳过构建阶段"
else
    print_info "知识库不存在，开始构建..."
    
    # 构建知识库命令
    KNOWLEDGE_BASE_CMD="export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} && \
    source ${CONDA_BASE}/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV} && \
    cd ${PROJECT_ROOT} && \
    python discovering.py \
        --mode=build_knowledge_base \
        --config_file_env=./configs/env_machine.yml \
        --config_file_expt=./configs/expts/${CONFIG_FILE} \
        --num_per_category=${KSHOT} \
        --knowledge_base_dir=${KNOWLEDGE_BASE_DIR}"
    
    # 创建知识库启动脚本
    KNOWLEDGE_BASE_SCRIPT="/tmp/run_knowledge_base_${DATASET}_$$.sh"
    cat > "${KNOWLEDGE_BASE_SCRIPT}" << EOF
#!/bin/bash
${KNOWLEDGE_BASE_CMD}
EOF
    chmod +x "${KNOWLEDGE_BASE_SCRIPT}"
    
    print_info "启动知识库构建阶段..." | tee -a "${PIPELINE_LOG}"
    print_info "统一日志: ${PIPELINE_LOG}" | tee -a "${PIPELINE_LOG}"
    nohup bash "${KNOWLEDGE_BASE_SCRIPT}" >> "${PIPELINE_LOG}" 2>&1 &
    KNOWLEDGE_BASE_PID=$!
    
    log_info "知识库构建进程ID: ${KNOWLEDGE_BASE_PID}"
    
    # 等待知识库构建完成
    if wait_for_process ${KNOWLEDGE_BASE_PID} ${KNOWLEDGE_BASE_TIMEOUT} "知识库构建阶段"; then
        log_success "知识库构建阶段完成"
        
        # 检查知识库文件是否生成
        if [ -f "${KNOWLEDGE_BASE_DIR}/image_knowledge_base.json" ]; then
            log_success "知识库文件生成成功: ${KNOWLEDGE_BASE_DIR}"
        else
            log_error "知识库文件未生成，检查日志: ${PIPELINE_LOG}"
            exit 1
        fi
    else
        log_error "知识库构建阶段失败，检查日志: ${PIPELINE_LOG}"
        exit 1
    fi
    
    # 清理临时文件
    rm -f "${KNOWLEDGE_BASE_SCRIPT}"
fi

# =============================================================================
# 阶段2: 评估系统
# =============================================================================
print_info "=== 阶段2: 评估系统 ===" 

# 检查测试数据目录
if [ ! -d "${TEST_DATA_DIR}" ]; then
    print_error "测试数据目录不存在: ${TEST_DATA_DIR}"
    exit 1
fi

# 构建评估命令
EVALUATION_CMD="export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} && \
source ${CONDA_BASE}/etc/profile.d/conda.sh && \
conda activate ${CONDA_ENV} && \
cd ${PROJECT_ROOT} && \
python discovering.py \
    --mode=${EVAL_MODE} \
    --config_file_env=./configs/env_machine.yml \
    --config_file_expt=./configs/expts/${CONFIG_FILE} \
    --test_data_dir=${TEST_DATA_DIR} \
    --knowledge_base_dir=${KNOWLEDGE_BASE_DIR} \
    --results_out=${RESULTS_OUT}"

# 创建评估启动脚本
EVALUATION_SCRIPT="/tmp/run_evaluation_${DATASET}_$$.sh"
cat > "${EVALUATION_SCRIPT}" << EOF
#!/bin/bash
${EVALUATION_CMD}
EOF
chmod +x "${EVALUATION_SCRIPT}"

print_info "启动评估阶段..." | tee -a "${PIPELINE_LOG}"
echo "=== 阶段2: 快慢思考评估开始 ===" >> "${PIPELINE_LOG}"
nohup bash "${EVALUATION_SCRIPT}" >> "${PIPELINE_LOG}" 2>&1 &
EVALUATION_PID=$!

log_info "评估进程ID: ${EVALUATION_PID}"

# 等待评估完成
if wait_for_process ${EVALUATION_PID} ${EVALUATION_TIMEOUT} "评估阶段"; then
    log_success "评估阶段完成"
    
    # 检查结果文件是否生成
    if [ -f "${RESULTS_OUT}" ]; then
        log_success "结果文件生成成功: ${RESULTS_OUT}"
    else
        log_warning "结果文件未生成，但评估已完成"
    fi
else
    log_error "评估阶段失败，检查日志: ${PIPELINE_LOG}"
    exit 1
fi

# 清理临时文件
rm -f "${EVALUATION_SCRIPT}"

# =============================================================================
# 完成总结
# =============================================================================
# 记录完成信息到日志
echo "=== 全流程运行完成 ===" >> "${PIPELINE_LOG}"
echo "总结信息:" >> "${PIPELINE_LOG}"
echo "- 统一日志文件: ${PIPELINE_LOG}" >> "${PIPELINE_LOG}"
echo "- 知识库目录: ${KNOWLEDGE_BASE_DIR}" >> "${PIPELINE_LOG}"
echo "- 结果文件: ${RESULTS_OUT}" >> "${PIPELINE_LOG}"
echo "- 执行时间: $(date)" >> "${PIPELINE_LOG}"

print_success "=== 全流程运行完成 ==="
print_info "统一日志文件: ${PIPELINE_LOG}"
print_info "查看最终结果: tail -20 ${PIPELINE_LOG}"

print_success "全流程脚本执行完成！"
