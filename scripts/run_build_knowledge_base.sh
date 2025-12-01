#!/bin/bash

# =============================================================================
# 知识库构建脚本 - Knowledge Base Builder
# =============================================================================
# 用法示例：
#   bash run_build_knowledge_base.sh                           # 使用YAML配置
#   bash run_build_knowledge_base.sh dog                       # 指定数据集
#   bash run_build_knowledge_base.sh dog --gpu 0 --kshot 5     # 指定多个参数
#   bash run_build_knowledge_base.sh bird --experience_number 12 --classify_top_k 15  # 指定超参数
#   bash run_build_knowledge_base.sh car --gpu 0 --kshot 5 --experience_number 10      # 完整参数
#   bash run_build_knowledge_base.sh bird --use_experience_base false --gpu 1   # 消融实验：不使用经验库
#   bash run_build_knowledge_base.sh dog --vocabulary_free true --gpu 0   # 消融实验：使用开放词汇
#
# 命令行参数：
#   位置参数1: 数据集名称 (dog, bird, flower, pet, car, aircraft, eurosat, food, dtd)
#   --gpu GPU_ID              GPU编号 (覆盖YAML配置)
#   --kshot NUM               每类样本数 (覆盖YAML配置)
#   --test_suffix NUM         测试数据后缀 (覆盖YAML配置)
#   --use_experience_base BOOL  是否使用经验库 (消融实验用)
#   --vocabulary_free BOOL    是否使用开放词汇 (消融实验用)
#   --conda_env ENV_NAME      Conda环境名 (覆盖YAML配置)
#   --help                    显示帮助信息

# =============================================================================
# 帮助函数
# =============================================================================
show_help() {
    cat << EOF
知识库构建脚本 - Knowledge Base Builder

用法:
    bash run_build_knowledge_base.sh [DATASET] [选项]

位置参数:
    DATASET                  数据集名称 (可选)
                            支持: dog, bird, flower, pet, car, aircraft, eurosat, food, dtd, caltech101, caltech256, deepfashion_multimodal, sun397, imagenet_a, imagenet_r, imagenet_1k, birdsnap, ucf
                            如不指定，使用config.yaml中的配置

选项:
    --gpu GPU_ID            GPU编号，例如: --gpu 0 或 --gpu "0,1"
    --kshot NUM             每个类别的样本数，例如: --kshot 5
    --test_suffix NUM       测试数据后缀，例如: --test_suffix 10
    --experience_number NUM 经验库最大经验条数，例如: --experience_number 12
    --classify_top_k NUM    分类时返回的类别数目，例如: --classify_top_k 10
    --use_experience_base BOOL 是否使用经验库（消融实验用），例如: --use_experience_base false
    --vocabulary_free BOOL  是否使用开放词汇（消融实验用），例如: --vocabulary_free true
    --conda_env ENV_NAME    Conda环境名称
    --help                  显示此帮助信息

示例:
    # 使用YAML配置
    bash run_build_knowledge_base.sh

    # 仅指定数据集
    bash run_build_knowledge_base.sh aircraft

    # 指定数据集和GPU
    bash run_build_knowledge_base.sh eurosat --gpu 1

    # 指定多个参数
    bash run_build_knowledge_base.sh food --gpu 2 --kshot 6 --test_suffix 8

    # 指定超参数
    bash run_build_knowledge_base.sh eurosat --experience_number 15 --classify_top_k 20

    # 消融实验：不使用经验库
    bash run_build_knowledge_base.sh bird --use_experience_base false --gpu 1

    # 消融实验：使用开放词汇
    bash run_build_knowledge_base.sh dog --vocabulary_free true --gpu 0

优先级: 命令行参数 > YAML配置文件

EOF
    exit 0
}

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
    # 使用grep和sed提取值，处理带引号和不带引号的情况
    grep "^[[:space:]]*${key}:" "$file" | sed 's/^[[:space:]]*[^:]*:[[:space:]]*//' | sed 's/[[:space:]]*#.*//' | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/"
}

# =============================================================================
# 命令行参数解析
# =============================================================================

# 首先从YAML读取默认配置
# 优先使用环境变量中的 CUDA_VISIBLE_DEVICES，否则从 YAML 读取
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    CUDA_VISIBLE_DEVICES_VALUE=$(get_yaml_value "cuda_visible_devices" "${CONFIG_FILE}")
else
    CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES}"
fi
DATASET_NAME=$(get_yaml_value "name" "${CONFIG_FILE}")
TEST_DATA_SUFFIX=$(get_yaml_value "test_data_suffix" "${CONFIG_FILE}")
KSHOT_VALUE=$(get_yaml_value "kshot" "${CONFIG_FILE}")
EXPERIENCE_NUMBER_VALUE=$(get_yaml_value "experience_base_max_number" "${CONFIG_FILE}")
CLASSIFY_TOP_K_VALUE=$(get_yaml_value "classify_top_k" "${CONFIG_FILE}")
USE_EXPERIENCE_BASE_VALUE=$(get_yaml_value "use_experience_base" "${CONFIG_FILE}")
VOCABULARY_FREE_VALUE=$(get_yaml_value "vocabulary_free" "${CONFIG_FILE}")
CONDA_ENV_VALUE=$(get_yaml_value "conda_env" "${CONFIG_FILE}")
CONDA_BASE_VALUE=$(get_yaml_value "conda_base" "${CONFIG_FILE}")
PROJECT_ROOT_VALUE=$(get_yaml_value "project_root" "${CONFIG_FILE}")
LOG_BASE_DIR_VALUE=$(get_yaml_value "base_dir" "${CONFIG_FILE}")

# 解析命令行参数
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            ;;
        --gpu)
            CUDA_VISIBLE_DEVICES_VALUE="$2"
            shift 2
            ;;
        --kshot)
            KSHOT_VALUE="$2"
            shift 2
            ;;
        --test_suffix)
            TEST_DATA_SUFFIX="$2"
            shift 2
            ;;
        --conda_env)
            CONDA_ENV_VALUE="$2"
            shift 2
            ;;
        --experience_number)
            EXPERIENCE_NUMBER_VALUE="$2"
            shift 2
            ;;
        --classify_top_k)
            CLASSIFY_TOP_K_VALUE="$2"
            shift 2
            ;;
        --use_experience_base)
            USE_EXPERIENCE_BASE_VALUE="$2"
            shift 2
            ;;
        --vocabulary_free)
            VOCABULARY_FREE_VALUE="$2"
            shift 2
            ;;
        --*)
            echo "错误: 未知选项 $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# 处理位置参数（数据集名称）
if [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    DATASET_NAME="${POSITIONAL_ARGS[0]}"
fi

# =============================================================================
# 应用配置参数
# =============================================================================

# GPU设置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"

# 数据集配置
DATASET="${DATASET_NAME}"
KSHOT="${KSHOT_VALUE}"

# 超参数配置
EXPERIENCE_NUMBER="${EXPERIENCE_NUMBER_VALUE}"
CLASSIFY_TOP_K="${CLASSIFY_TOP_K_VALUE}"
USE_EXPERIENCE_BASE="${USE_EXPERIENCE_BASE_VALUE}"
VOCABULARY_FREE="${VOCABULARY_FREE_VALUE}"

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
        DATASET_DIR="CUB_200_2011/CUB_200_2011"  # 鸟类数据集特殊路径
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
    "aircraft")
        DATASET_NUM="100"
        CONFIG_FILE="aircraft100_all.yml"
        DATASET_DIR="fgvc_aircraft"
        ;;
    "eurosat")
        DATASET_NUM="10"
        CONFIG_FILE="eurosat10_all.yml"
        DATASET_DIR="eurosat"
        ;;
    "food")
        DATASET_NUM="101"
        CONFIG_FILE="food101_all.yml"
        DATASET_DIR="food_101"
        ;;
    "dtd")
        DATASET_NUM="47"
        CONFIG_FILE="dtd47_all.yml"
        DATASET_DIR="dtd"
        ;;
    "caltech101")
        DATASET_NUM="101"
        CONFIG_FILE="caltech101_all.yml"
        DATASET_DIR="caltech101"
        ;;
    "caltech256")
        DATASET_NUM="256"
        CONFIG_FILE="caltech256_all.yml"
        DATASET_DIR="caltech256"
        ;;
    "deepfashion_multimodal")
        DATASET_NUM="23"
        CONFIG_FILE="deepfashion_multimodal23_all.yml"
        DATASET_DIR="DeepFashion"
        ;;
    "sun397")
        DATASET_NUM="397"
        CONFIG_FILE="sun397_all.yml"
        DATASET_DIR="SUN397"
        ;;
    "imagenet_a")
        DATASET_NUM="200"
        CONFIG_FILE="imagenet_a200_all.yml"
        DATASET_DIR="ImageNet_A"
        ;;
    "imagenet_r")
        DATASET_NUM="200"
        CONFIG_FILE="imagenet_r200_all.yml"
        DATASET_DIR="ImageNet_R"
        ;;
    "imagenet_1k")
        DATASET_NUM="1000"
        CONFIG_FILE="imagenet_1k_all.yml"
        DATASET_DIR="ImageNet_1k"
        ;;
    "birdsnap")
        DATASET_NUM="500"
        CONFIG_FILE="birdsnap500_all.yml"
        DATASET_DIR="birdsnap"
        ;;
    "ucf")
        DATASET_NUM="101"
        CONFIG_FILE="ucf101_all.yml"
        DATASET_DIR="ucf101"
        ;;
    "imagenet_sketch")
        DATASET_NUM="1000"
        CONFIG_FILE="imagenet_sketch1000_all.yml"
        DATASET_DIR="ImageNet_Sketch"
        ;;
    "imagenet_v2")
        DATASET_NUM="1000"
        CONFIG_FILE="imagenet_v2_1000_all.yml"
        DATASET_DIR="ImageNet_v2"
        ;;
    *)
        echo "错误: 不支持的数据集 '${DATASET}'"
        echo "支持的数据集: dog, bird, flower, pet, car, aircraft, eurosat, food, dtd, caltech101, caltech256, deepfashion_multimodal, sun397, imagenet_a, imagenet_r, imagenet_1k, birdsnap, ucf, imagenet_sketch, imagenet_v2"
        exit 1
        ;;
esac

# 生成路径
# 对于 caltech101 和 caltech256，DATASET 已经包含编号，不需要再加 DATASET_NUM
# deepfashion_multimodal 使用 deepfashion_multimodal23 作为实验目录名
# sun397 使用 sun397 作为实验目录名（已包含编号）
# imagenet_1k 使用 imagenet_1k 作为实验目录名（已包含编号）
if [ "${DATASET}" = "caltech101" ] || [ "${DATASET}" = "caltech256" ] || [ "${DATASET}" = "sun397" ] || [ "${DATASET}" = "imagenet_1k" ]; then
    EXPERIMENT_DIR="${DATASET}"
elif [ "${DATASET}" = "deepfashion_multimodal" ]; then
    EXPERIMENT_DIR="deepfashion_multimodal23"
elif [ "${DATASET}" = "imagenet_sketch" ]; then
    EXPERIMENT_DIR="ImageNet_Sketch1000"
elif [ "${DATASET}" = "imagenet_v2" ]; then
    EXPERIMENT_DIR="imagenet_v2_1000"
else
    EXPERIMENT_DIR="${DATASET}${DATASET_NUM}"
fi

KNOWLEDGE_BASE_DIR="./experiments/${EXPERIMENT_DIR}/knowledge_base"
LOG_DIR="${LOG_BASE_DIR}/knowledge_base/${EXPERIMENT_DIR}"

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

LOG_FILE=$(generate_log_filename "build_knowledge_base_${DATASET}" "${LOG_DIR}")

# 超参数配置
EXPERIENCE_NUMBER="${EXPERIENCE_NUMBER_VALUE:-8}"
CLASSIFY_TOP_K="${CLASSIFY_TOP_K_VALUE:-10}"

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

# 打印配置信息到终端和日志文件
print_info "=== 运行配置 ==="
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "数据集: ${DATASET}"
echo "配置文件: ${CONFIG_FILE}"
echo "运行模式: build_knowledge_base"
echo "K-shot: ${KSHOT}"
echo "Experience Number: ${EXPERIENCE_NUMBER}  # 经验库最大经验条数"
echo "Classify Top K: ${CLASSIFY_TOP_K}  # 分类时返回的类别数目"
echo "Use Experience Base: ${USE_EXPERIENCE_BASE}  # 是否使用经验库（消融实验用）"
echo "Vocabulary Free: ${VOCABULARY_FREE}  # 是否使用开放词汇（消融实验用）"
echo "知识库目录: ${KNOWLEDGE_BASE_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "虚拟环境: ${CONDA_ENV}"
print_info "================"

# 将配置信息写入临时文件
TEMP_HEADER="/tmp/kb_header_${DATASET}_$$.txt"
cat > "${TEMP_HEADER}" << LOGHEADER
[INFO] === Build Knowledge Base 启动, YAML 配置摘要 ===
GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  # 使用的GPU编号
Dataset: ${DATASET} (num_classes=${NUM_CLASSES})  # 数据集及类别数
K-shot: ${KSHOT}  # 检索库使用每个类别的样本数目
Conda Env: ${CONDA_ENV}  # Conda环境名称, Conda Base: ${CONDA_BASE}  # Conda安装路径
Project Root: ${PROJECT_ROOT}  # 项目根目录
Knowledge Base Dir: ${KNOWLEDGE_BASE_DIR}  # 知识库目录
Config File: ./configs/expts/${CONFIG_FILE}  # 实验配置文件
Log File: ${LOG_FILE}  # 日志文件路径
Run Mode: build_knowledge_base  # 运行模式
[INFO] ========================================================

LOGHEADER

# 激活conda环境并运行
print_info "激活虚拟环境并开始运行..."
print_info "日志将实时写入: ${LOG_FILE}"
print_info "可以使用 'tail -f '${LOG_FILE}'' 查看实时日志"

# 检查conda环境是否存在
if [ ! -d "${CONDA_BASE}/envs/${CONDA_ENV}" ]; then
    print_error "Conda环境不存在: ${CONDA_BASE}/envs/${CONDA_ENV}"
    print_error "请检查CONDA_ENV和CONDA_BASE配置是否正确"
    exit 1
fi

# 构建命令
CMD="source /home/hdl/miniconda3/envs/${CONDA_ENV}/bin/activate && python discovering.py \
    --mode=build_knowledge_base \
    --config_file_env=./configs/env_machine.yml \
    --config_file_expt=./configs/expts/${CONFIG_FILE} \
    --num_per_category=${KSHOT} \
    --knowledge_base_dir=${KNOWLEDGE_BASE_DIR} \
    --experience_number=${EXPERIENCE_NUMBER} \
    --classify_top_k=${CLASSIFY_TOP_K} \
    --use_experience_base=${USE_EXPERIENCE_BASE} \
    --vocabulary_free=${VOCABULARY_FREE}"

# 创建启动脚本（先写配置信息，再运行Python）
TEMP_SCRIPT="/tmp/run_build_knowledge_base_${DATASET}_$$.sh"
cat > "${TEMP_SCRIPT}" << EOF
#!/bin/bash
# 先将配置信息写入日志
cat "${TEMP_HEADER}"
# 然后运行Python程序
${CMD}
EOF
chmod +x "${TEMP_SCRIPT}"

# 后台运行逻辑
print_info "开始后台运行..."
    nohup bash "${TEMP_SCRIPT}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    print_success "任务已启动！"
    print_info "进程ID: ${PID}"
    print_info "日志文件: ${LOG_FILE}"
    print_info "查看实时日志: tail -f '${LOG_FILE}'"
    print_info "停止任务: kill ${PID}"

    # 等待几秒钟检查进程是否正常启动
    sleep 3
    if kill -0 ${PID} 2>/dev/null; then
        print_success "进程运行正常"
    else
        print_error "进程启动失败，请检查日志文件"
        exit 1
    fi

# 清理临时文件
print_info "清理临时文件..."
rm -f "${TEMP_SCRIPT}"

print_info "脚本执行完成"
