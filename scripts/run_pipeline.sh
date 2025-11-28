#!/bin/bash
# =============================================================================
# FGVR Pipeline 脚本 - 完全后台执行，统一日志文件 + 精简 YAML 输出 + 自动递增日志
# 知识库构建 + 快慢思考评估
# =============================================================================
# 用法示例：
#   bash run_pipeline.sh                           # 使用YAML配置
#   bash run_pipeline.sh aircraft                  # 指定数据集
#   bash run_pipeline.sh eurosat --gpu 2 --kshot 5 # 多参数
#
# 命令行参数：
#   位置参数1: 数据集名称 (dog, bird, flower, pet, car, aircraft, eurosat, food, dtd)
#   --gpu GPU_ID              GPU编号
#   --kshot NUM               每类样本数
#   --test_suffix NUM         测试数据后缀
#   --conda_env ENV_NAME      Conda环境名
#   --help                    显示帮助信息

# =============================================================================
# 帮助函数
# =============================================================================
show_help() {
    cat << EOF
FGVR Pipeline 脚本 - 完整流程（知识库构建 + 快慢思考评估）

用法:
    bash run_pipeline.sh [DATASET] [选项]

位置参数:
    DATASET                  数据集名称 (可选)
                            支持: dog, bird, flower, pet, car, aircraft, eurosat, food, dtd, caltech101, caltech256, deepfashion_multimodal, sun397, imagenet_a, imagenet_r, imagenet_1k, birdsnap, ucf

选项:
    --gpu GPU_ID            GPU编号
    --kshot NUM             每个类别的样本数
    --test_suffix NUM       测试数据后缀（使用discovery集时）
    --use_test_data         使用images_test目录进行测试
    --test_percentage NUM   测试集采样百分比 (0-100)
    --conda_env ENV_NAME    Conda环境名称
    --help                  显示此帮助信息

示例:
    # 使用YAML配置
    bash run_pipeline.sh

    # 指定数据集
    bash run_pipeline.sh aircraft

    # 使用discovery集
    bash run_pipeline.sh food --gpu 3 --kshot 6 --test_suffix 8
    
    # 使用测试集
    bash run_pipeline.sh bird --gpu 1 --kshot 5 --use_test_data --test_percentage 30

优先级: 命令行参数 > YAML配置文件

EOF
    exit 0
}

# =============================================================================
# 配置读取
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "[ERROR] 配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi

# 简单 YAML 解析函数
get_yaml_value() {
    local key=$1
    local file=$2
    grep "^[[:space:]]*${key}:" "$file" \
        | sed 's/^[[:space:]]*[^:]*:[[:space:]]*//' \
        | sed 's/[[:space:]]*#.*//' \
        | sed 's/^"\(.*\)"$/\1/' \
        | sed "s/^'\(.*\)'$/\1/"
}

# =============================================================================
# 命令行参数解析
# =============================================================================

# 首先从YAML读取默认配置
CUDA_VISIBLE_DEVICES=$(get_yaml_value "cuda_visible_devices" "${CONFIG_FILE}")
DATASET=$(get_yaml_value "name" "${CONFIG_FILE}")
TEST_DATA_SUFFIX=$(get_yaml_value "test_data_suffix" "${CONFIG_FILE}")
USE_TEST_DATA=$(get_yaml_value "use_test_data" "${CONFIG_FILE}")
TEST_PERCENTAGE=$(get_yaml_value "test_percentage" "${CONFIG_FILE}")
KSHOT=$(get_yaml_value "kshot" "${CONFIG_FILE}")
CONDA_ENV=$(get_yaml_value "conda_env" "${CONFIG_FILE}")
CONDA_BASE=$(get_yaml_value "conda_base" "${CONFIG_FILE}")
PROJECT_ROOT=$(get_yaml_value "project_root" "${CONFIG_FILE}")
LOG_BASE_DIR=$(get_yaml_value "base_dir" "${CONFIG_FILE}")

# 解析命令行参数
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            ;;
        --gpu)
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --kshot)
            KSHOT="$2"
            shift 2
            ;;
        --test_suffix)
            TEST_DATA_SUFFIX="$2"
            shift 2
            ;;
        --use_test_data)
            USE_TEST_DATA="true"
            shift
            ;;
        --test_percentage)
            TEST_PERCENTAGE="$2"
            shift 2
            ;;
        --conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --*)
            echo "[ERROR] 未知选项: $1"
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
    DATASET="${POSITIONAL_ARGS[0]}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

# 数据集映射
case "${DATASET}" in
    "dog")      DATASET_NUM="120"; CONFIG_FILE_DS="dog120_all.yml"; DATASET_DIR="dogs_120" ;;
    "bird")     DATASET_NUM="200"; CONFIG_FILE_DS="bird200_all.yml"; DATASET_DIR="CUB_200_2011/CUB_200_2011" ;;
    "flower")   DATASET_NUM="102"; CONFIG_FILE_DS="flower102_all.yml"; DATASET_DIR="flowers_102" ;;
    "pet")      DATASET_NUM="37"; CONFIG_FILE_DS="pet37_all.yml"; DATASET_DIR="pet_37" ;;
    "car")      DATASET_NUM="196"; CONFIG_FILE_DS="car196_all.yml"; DATASET_DIR="car_196" ;;
    "aircraft") DATASET_NUM="100"; CONFIG_FILE_DS="aircraft100_all.yml"; DATASET_DIR="fgvc_aircraft" ;;
    "eurosat")  DATASET_NUM="10"; CONFIG_FILE_DS="eurosat10_all.yml"; DATASET_DIR="eurosat" ;;
    "food")     DATASET_NUM="101"; CONFIG_FILE_DS="food101_all.yml"; DATASET_DIR="food_101" ;;
    "dtd")      DATASET_NUM="47"; CONFIG_FILE_DS="dtd47_all.yml"; DATASET_DIR="dtd" ;;
    "caltech101") DATASET_NUM="101"; CONFIG_FILE_DS="caltech101_all.yml"; DATASET_DIR="caltech101" ;;
    "caltech256") DATASET_NUM="256"; CONFIG_FILE_DS="caltech256_all.yml"; DATASET_DIR="caltech256" ;;
    "deepfashion_multimodal") DATASET_NUM="23"; CONFIG_FILE_DS="deepfashion_multimodal23_all.yml"; DATASET_DIR="DeepFashion" ;;
    "sun397") DATASET_NUM="397"; CONFIG_FILE_DS="sun397_all.yml"; DATASET_DIR="SUN397" ;;
    "imagenet_a") DATASET_NUM="200"; CONFIG_FILE_DS="imagenet_a200_all.yml"; DATASET_DIR="ImageNet_A" ;;
    "imagenet_r") DATASET_NUM="200"; CONFIG_FILE_DS="imagenet_r200_all.yml"; DATASET_DIR="ImageNet_R" ;;
    "imagenet_1k") DATASET_NUM="1000"; CONFIG_FILE_DS="imagenet_1k_all.yml"; DATASET_DIR="ImageNet_1k" ;;
    "birdsnap") DATASET_NUM="500"; CONFIG_FILE_DS="birdsnap500_all.yml"; DATASET_DIR="birdsnap" ;;
    "ucf") DATASET_NUM="101"; CONFIG_FILE_DS="ucf101_all.yml"; DATASET_DIR="ucf_101" ;;
    *) echo "[ERROR] 不支持的数据集 '${DATASET}'. 支持: dog, bird, flower, pet, car, aircraft, eurosat, food, dtd, caltech101, caltech256, deepfashion_multimodal, sun397, imagenet_a, imagenet_r, imagenet_1k, birdsnap, ucf"; exit 1 ;;
esac

# 对于 caltech101 和 caltech256，DATASET 已经包含编号，不需要再加 DATASET_NUM
# deepfashion_multimodal 使用 deepfashion_multimodal23 作为实验目录名
# sun397 使用 sun397 作为实验目录名（已包含编号）
# imagenet_1k 使用 imagenet_1k 作为实验目录名（已包含编号）
if [ "${DATASET}" = "caltech101" ] || [ "${DATASET}" = "caltech256" ] || [ "${DATASET}" = "sun397" ] || [ "${DATASET}" = "birdsnap" ] || [ "${DATASET}" = "imagenet_1k" ]; then
    EXPERIMENT_DIR="${DATASET}"
elif [ "${DATASET}" = "deepfashion_multimodal" ]; then
    EXPERIMENT_DIR="deepfashion_multimodal23"
else
    EXPERIMENT_DIR="${DATASET}${DATASET_NUM}"
fi

KNOWLEDGE_BASE_DIR="./experiments/${EXPERIMENT_DIR}/knowledge_base"
# 修改：使用JSON文件而不是目录
TEST_DATA_JSON="./experiments/${EXPERIMENT_DIR}/images_split/images_discovery_all_${TEST_DATA_SUFFIX}.json"
RESULTS_OUT="./results/${DATASET}_fast_slow_results.json"
LOG_DIR="${LOG_BASE_DIR}/pipeline/${EXPERIMENT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${RESULTS_OUT}")"

# =============================================================================
# 自动生成唯一日志文件（存在则递增）
# =============================================================================
generate_log_filename() {
    local base_name=$1
    local log_dir=$2
    local file="${log_dir}/${base_name}.log"
    if [ ! -f "$file" ]; then
        echo "$file"
        return
    fi
    local counter=1
    while [ -f "${log_dir}/${base_name}(${counter}).log" ]; do
        counter=$((counter+1))
    done
    echo "${log_dir}/${base_name}(${counter}).log"
}
LOG_FILE=$(generate_log_filename "pipeline_${DATASET}" "${LOG_DIR}")

# =============================================================================
# 颜色输出函数（仅前台打印信息）
# =============================================================================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# 检查项目和环境
# =============================================================================
if [ ! -d "${PROJECT_ROOT}" ]; then
    print_error "项目根目录不存在: ${PROJECT_ROOT}"; exit 1
fi
cd "${PROJECT_ROOT}" || exit 1
if [ ! -d "${CONDA_BASE}/envs/${CONDA_ENV}" ]; then
    print_error "Conda环境不存在: ${CONDA_BASE}/envs/${CONDA_ENV}"; exit 1
fi

# =============================================================================
# 完全后台执行函数（Step1报错则退出）
# =============================================================================
run_pipeline_bg() {
    (
        # 输出 YAML 配置关键内容到日志（带中文）
        echo "[INFO] === Pipeline 启动, YAML 配置摘要 ==="
        echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  # 使用的GPU编号"
        echo "Dataset: ${DATASET} (num_classes=${DATASET_NUM})  # 数据集及类别数"
        if [ "${USE_TEST_DATA}" = "true" ]; then
            echo "Test Mode: images_test  # 使用测试集"
            echo "Test Percentage: ${TEST_PERCENTAGE}%  # 测试集采样百分比"
        else
            echo "Test Mode: discovery_set  # 使用discovery集, test_data_suffix=${TEST_DATA_SUFFIX}"
            echo "Test Data Json: ${TEST_DATA_JSON}  # 测试数据JSON文件"
        fi
        echo "K-shot: ${KSHOT}  # 检索库使用每个类别的样本数目"
        echo "Conda Env: ${CONDA_ENV}  # Conda环境名称, Conda Base: ${CONDA_BASE}  # Conda安装路径"
        echo "Knowledge Base Dir: ${KNOWLEDGE_BASE_DIR}  # 知识库目录"
        echo "Results Out: ${RESULTS_OUT}  # 快慢思考评估结果输出文件"
        echo "---------------------------"
        echo ""

        # 激活环境
        source "${CONDA_BASE}/envs/${CONDA_ENV}/bin/activate"

        # Step1: 构建知识库
        echo "[INFO] === Step1: 构建知识库 ==="
        python discovering.py --mode=build_knowledge_base \
            --config_file_env=./configs/env_machine.yml \
            --config_file_expt=./configs/expts/${CONFIG_FILE_DS} \
            --num_per_category=${KSHOT} \
            --knowledge_base_dir=${KNOWLEDGE_BASE_DIR}
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "[ERROR] Step1: 知识库构建失败，退出 pipeline (exit code=${EXIT_CODE})"
            exit $EXIT_CODE
        fi
        echo "[SUCCESS] Step1: 知识库构建完成"

        # Step2: 快慢思考评估
        echo "[INFO] === Step2: 快慢思考评估 ==="
        if [ "${USE_TEST_DATA}" = "true" ]; then
            python discovering.py --mode=fast_slow \
                --config_file_env=./configs/env_machine.yml \
                --config_file_expt=./configs/expts/${CONFIG_FILE_DS} \
                --use_test_data \
                --test_percentage=${TEST_PERCENTAGE} \
                --knowledge_base_dir=${KNOWLEDGE_BASE_DIR} \
                --results_out=${RESULTS_OUT}
        else
        python discovering.py --mode=fast_slow \
            --config_file_env=./configs/env_machine.yml \
            --config_file_expt=./configs/expts/${CONFIG_FILE_DS} \
            --test_data_dir=${TEST_DATA_JSON} \
            --knowledge_base_dir=${KNOWLEDGE_BASE_DIR} \
            --results_out=${RESULTS_OUT}
        fi

        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "[ERROR] Step2: 快慢思考评估失败 (exit code=${EXIT_CODE})"
            exit $EXIT_CODE
        fi
        echo "[SUCCESS] Step2: 快慢思考评估完成"
    ) >> "${LOG_FILE}" 2>&1 &
    PID_BG=$!
    echo $PID_BG
}

# =============================================================================
# 启动后台 pipeline
# =============================================================================
PID_PIPELINE=$(run_pipeline_bg)
print_success "Pipeline 已启动 (完全后台), PID=${PID_PIPELINE}"
print_info "日志文件: ${LOG_FILE}"
echo "实时查看日志: tail -f '${LOG_FILE}'"
