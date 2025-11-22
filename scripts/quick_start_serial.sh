#!/bin/bash
TARGET_GPUS="all" # 目标GPU列表（空格分隔，脚本会自动分配到这些GPU中空闲的，"all"表示使用所有GPU）
LOG_DIR="/home/hdl/project/fgvr_test/logs/quick_start" # 日志目录
RESERVED_GPUS=1 # 预留给其他用户的GPU数量（0表示不预留，1表示至少留1张卡，以此类推）
MAX_RETRY_TIMES=-1 # 命令失败时的最大重试次数（-1表示无限重试，0表示不重试，>0表示最多重试N次）
RUN_IN_BACKGROUND=true # 是否后台运行（true=完全后台运行，日志实时刷新；false=前台运行）
EXTREME_MODE=false # 极限资源利用模式（true=只要有足够显存就调度，允许多进程共享GPU；false=正常模式）
ALLOW_SHARED_GPU=true # 是否允许与他人共享GPU（true=允许共享，false=独占模式，仅在EXTREME_MODE=false时生效）
DEBUG_MODE=true # 调试模式（true=输出详细调试信息；false=正常输出）

COMMANDS=(
    # ========== 实际运行的命令 ==========
    #"bash /home/hdl/project/fgvr_test/scripts/run_pipeline.sh eurosat --gpu \${available_gpu} | 35 | 1"
    #"bash /home/hdl/project/fgvr_test/scripts/run_pipeline.sh dtd --gpu \${available_gpu} | 35 | 1"
    #"bash /home/hdl/project/fgvr_test/scripts/run_pipeline.sh aircraft --gpu \${available_gpu} | 35 | 1"
    "bash /home/hdl/project/fgvr_test/scripts/fast_slow.sh eurosat --gpu \${available_gpu} --test_suffix 1 | 35 | 1"
    "bash /home/hdl/project/fgvr_test/scripts/fast_slow.sh pet --gpu \${available_gpu} --test_suffix 1 | 35 | 1" 
    "bash /home/hdl/project/fgvr_test/scripts/fast_slow.sh bird --gpu \${available_gpu} --test_suffix 1 | 35 | 1" 
    
    # ========== 示例1：Bash脚本命令 ==========
    
    # 项目脚本（单卡）
    # "bash scripts/run_build_knowledge_base.sh aircraft --gpu \${available_gpu} --kshot 4 | 15 | 1"
    # "bash scripts/run_fast_slow.sh aircraft --gpu \${available_gpu} --test_suffix 10 | 20 | 1"
    # "bash scripts/run_pipeline.sh aircraft --gpu \${available_gpu} --kshot 4 | 25 | 1"
    
    # ========== 示例2：Python脚本命令 ==========
    
    # Python命令（单卡）- 使用CUDA_VISIBLE_DEVICES
    # "CUDA_VISIBLE_DEVICES=\${available_gpu} python train.py --epochs 100 | 18 | 1"
    # "CUDA_VISIBLE_DEVICES=\${available_gpu} python evaluate.py --model best.pth | 10 | 1"
    
    # Python命令（单卡）- 使用--gpu参数
    # "python train.py --gpu \${available_gpu} --epochs 50 --batch_size 32 | 20 | 1"
    
    # Python命令（单卡）- 直接运行
    # "python /path/to/script.py --arg1 value1 --gpu \${available_gpu} | 20 | 1"
    
    # ========== 示例3：多卡命令（优先调度） ==========
    
    # 双卡训练
    # "CUDA_VISIBLE_DEVICES=\${available_gpu_0},\${available_gpu_1} python train_ddp.py --epochs 100 | 36 | 2"
    # "python train.py --gpu \${available_gpu_0} \${available_gpu_1} --distributed | 40 | 2"
    
    # 三卡训练
    # "CUDA_VISIBLE_DEVICES=\${available_gpu_0},\${available_gpu_1},\${available_gpu_2} python train_multi.py | 50 | 3"
    
    # 四卡训练
    # "python train.py --gpus \${available_gpu_0},\${available_gpu_1},\${available_gpu_2},\${available_gpu_3} | 80 | 4"
    
    # ========== 示例4：混合命令 ==========
    
    # 先单卡预处理，再多卡训练
    # "python preprocess.py --input data/ --output processed/ | 0 | 1"
    # "CUDA_VISIBLE_DEVICES=\${available_gpu_0},\${available_gpu_1} python train.py --data processed/ | 40 | 2"
    
    # ========== 示例5：不需要GPU的命令 ==========
    
    # CPU命令（显存需求为0，GPU数量为0）
    # "python analyze_results.py --input results/ --output report.pdf | 0 | 0"
    # "bash scripts/postprocess.sh | 0 | 0"
)

# =============================================================================
# 智能多GPU串行执行脚本 - Smart Multi-GPU Serial Execution
# =============================================================================
# 功能：
#   1. 多GPU动态分配（自动分配到空闲GPU）
#   2. 智能GPU显存监控（每个命令可指定不同的显存需求）
#   3. 支持单卡和多卡命令（通过 available_gpu_N 变量）
#   4. 多卡命令优先调度
#   5. 自动重试机制（显存不足或命令失败时自动重试）
#   6. 真正的串行执行（等待进程完全结束）
#   7. 自动递增日志文件，避免覆盖
#   8. GPU占用追踪（避免同一GPU被分配给多个命令）
#   9. 支持Bash和Python脚本
#   10. 完全后台运行模式
#
# GPU变量使用：
#   - ${available_gpu}     : 单卡命令使用
#   - ${available_gpu_0}   : 多卡命令第1张卡
#   - ${available_gpu_1}   : 多卡命令第2张卡
#   - ${available_gpu_N}   : 多卡命令第N+1张卡 (N=0-29)
#
# 用法：
#   1. 设置 TARGET_GPUS 为可用GPU列表
#   2. 在 COMMANDS 数组中添加命令，格式：
#      "命令内容 | 所需显存(GB) | 需要GPU数量"
#   3. 运行: bash scripts/quick_start_serial.sh
#   4. 后台运行: 设置 RUN_IN_BACKGROUND=true
#
# =============================================================================

# =============================================================================
# 📝 配置区域 - 在这里配置GPU和命令
# =============================================================================

# 显存检查间隔（秒）
CHECK_INTERVAL=3

# 命令数组 - 格式: "命令 | 所需显存(GB) | GPU数量"
# 注意：
#   1. 命令、显存需求、GPU数量用 | 分隔
#   2. GPU数量默认为1（单卡）
#   3. 命令中使用 \${available_gpu} 或 \${available_gpu_0}, \${available_gpu_1} 等变量
#   4. 脚本会自动替换这些变量为实际的GPU编号
#   5. 使用 # 注释掉不需要执行的命令
#   6. 支持bash和python脚本

# =============================================================================
# 初始化
# =============================================================================

mkdir -p "${LOG_DIR}"

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1

# 处理TARGET_GPUS="all"的情况
if [ "$TARGET_GPUS" = "all" ] || [ "$TARGET_GPUS" = "ALL" ]; then
    if command -v nvidia-smi &> /dev/null; then
        # 获取所有GPU的ID
        ALL_GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')
        TARGET_GPUS="${ALL_GPU_IDS% }"  # 去掉末尾空格
        echo "[DEBUG] TARGET_GPUS='all' 检测到 ${TARGET_GPUS}"
    else
        echo "[ERROR] TARGET_GPUS='all' 但 nvidia-smi 不可用"
        exit 1
    fi
fi

# GPU占用追踪（记录本脚本启动的进程占用的GPU及其已用显存）
declare -A SCRIPT_GPU_OCCUPIED  # 0=空闲，1=被本脚本占用
declare -A SCRIPT_GPU_USED_MEM  # 记录本脚本在该GPU上已分配的显存(MB)
for gpu_id in $TARGET_GPUS; do
    SCRIPT_GPU_OCCUPIED[$gpu_id]=0
    SCRIPT_GPU_USED_MEM[$gpu_id]=0
done

# =============================================================================
# 工具函数
# =============================================================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
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

print_step() {
    echo -e "${CYAN}[STEP $1/$2]${NC} $3"
}

print_debug() {
    if [ "$DEBUG_MODE" = true ]; then
        echo -e "${MAGENTA}[DEBUG]${NC} $1"
    fi
}

# 生成唯一日志文件名
generate_log_filename() {
    local base_name="quick_start_serial"
    local file="${LOG_DIR}/${base_name}.log"
    
    if [ ! -f "$file" ]; then
        echo "$file"
        return
    fi
    
    local counter=1
    while [ -f "${LOG_DIR}/${base_name}(${counter}).log" ]; do
        counter=$((counter + 1))
    done
    
    echo "${LOG_DIR}/${base_name}(${counter}).log"
}

# 解析命令行（分离命令、显存需求和GPU数量）
parse_command_line() {
    local line="$1"
    
    # 使用 | 分隔命令、显存需求和GPU数量
    # 格式1: cmd | mem | gpu_count
    if [[ $line =~ ^(.+)[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*$ ]]; then
        local cmd="${BASH_REMATCH[1]}"
        local mem="${BASH_REMATCH[2]}"
        local gpu_count="${BASH_REMATCH[3]}"
        echo "${cmd}|${mem}|${gpu_count}"
    # 格式2: cmd | mem (默认1张GPU)
    elif [[ $line =~ ^(.+)[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*$ ]]; then
        local cmd="${BASH_REMATCH[1]}"
        local mem="${BASH_REMATCH[2]}"
        echo "${cmd}|${mem}|1"
    else
        # 默认：0显存，1张GPU
        echo "${line}|0|1"
    fi
}

# 获取GPU剩余显存（MB）
get_gpu_free_memory() {
    local gpu_id=$1
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "999999"  # 返回一个很大的值
        return
    fi
    
    # 获取指定GPU的剩余显存（MB）
    local free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | head -1)
    
    if [ -z "$free_mem" ]; then
        echo "0"
    else
        echo "$free_mem"
    fi
}

# 获取GPU总显存（MB）
get_gpu_total_memory() {
    local gpu_id=$1
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "999999"
        return
    fi
    
    local total_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | head -1)
    
    if [ -z "$total_mem" ]; then
        echo "0"
    else
        echo "$total_mem"
    fi
}

# 获取GPU利用率（%）
get_gpu_utilization() {
    local gpu_id=$1
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "0"
        return
    fi
    
    local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | head -1)
    
    if [ -z "$util" ]; then
        echo "0"
    else
        echo "$util"
    fi
}

# 检查GPU是否完全空闲（独占模式）
is_gpu_fully_idle() {
    local gpu_id=$1
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "true"
        return
    fi
    
    # 获取利用率
    local util=$(get_gpu_utilization "$gpu_id")
    
    # 获取显存使用率
    local free_mem=$(get_gpu_free_memory "$gpu_id")
    local total_mem=$(get_gpu_total_memory "$gpu_id")
    local used_mem=$((total_mem - free_mem))
    local mem_usage_percent=$(awk "BEGIN {printf \"%.2f\", ($used_mem/$total_mem)*100}")
    
    # 判断条件：利用率0% 且 空闲显存 > 97%
    if [ "$util" -eq 0 ] && [ $(echo "$mem_usage_percent < 3" | bc -l 2>/dev/null || echo "0") -eq 1 ]; then
        echo "true"
    else
        echo "false"
    fi
}

# 获取可用GPU列表（根据不同模式筛选）
# 返回格式：按剩余显存降序排列的GPU列表
get_available_gpus() {
    local required_mem_gb=$1
    local required_mem_mb=$((required_mem_gb * 1024))
    local available=()
    declare -A gpu_free_mem_map
    declare -A gpu_remaining_mem  # 极限模式下的实际可用显存
    
    local current_time=$(date '+%Y-%m-%d %H:%M:%S')
    print_info "[$current_time] 开始扫描可用GPU..."
    print_debug "  模式配置: EXTREME_MODE=$EXTREME_MODE, ALLOW_SHARED_GPU=$ALLOW_SHARED_GPU"
    print_debug "  所需显存: ${required_mem_gb}GB (${required_mem_mb}MB)"
    
    if ! command -v nvidia-smi &> /dev/null; then
        # 如果没有nvidia-smi，返回未被本脚本占用的GPU
        for gpu_id in $TARGET_GPUS; do
            if [ "$EXTREME_MODE" = true ] || [ "${SCRIPT_GPU_OCCUPIED[$gpu_id]}" -eq 0 ]; then
                available+=("$gpu_id")
            fi
        done
        print_info "  └─ nvidia-smi不可用，返回所有未占用GPU"
        echo "${available[@]}"
        return
    fi
    
    # 获取所有满足条件的GPU
    for gpu_id in $TARGET_GPUS; do
        local free_mem=$(get_gpu_free_memory "$gpu_id")
        local total_mem=$(get_gpu_total_memory "$gpu_id")
        local util=$(get_gpu_utilization "$gpu_id")
        local free_gb=$((free_mem / 1024))
        local used_mem=$((total_mem - free_mem))
        local mem_usage_percent=$(awk "BEGIN {printf \"%.1f\", ($used_mem/$total_mem)*100}")
        local script_used_mem=${SCRIPT_GPU_USED_MEM[$gpu_id]:-0}
        local script_used_gb=$((script_used_mem / 1024))
        
        print_debug "  GPU $gpu_id 详细信息:"
        print_debug "    - 总显存: $((total_mem / 1024))GB"
        print_debug "    - 空闲显存: ${free_gb}GB"
        print_debug "    - 利用率: ${util}%"
        print_debug "    - 本脚本已用: ${script_used_gb}GB"
        
        # 极限模式：只要有足够显存就可以使用，允许多进程共享
        if [ "$EXTREME_MODE" = true ]; then
            # 计算实际可用显存（考虑本脚本已分配的显存）
            local actual_free_mem=$((free_mem - script_used_mem))
            
            if [ "$actual_free_mem" -ge "$required_mem_mb" ]; then
                available+=("$gpu_id")
                gpu_free_mem_map[$gpu_id]=$actual_free_mem
                gpu_remaining_mem[$gpu_id]=$actual_free_mem
                print_info "  └─ GPU $gpu_id: 可用 [极限模式] (实际空闲:$((actual_free_mem/1024))GB, 系统空闲:${free_gb}GB, 本脚本占用:${script_used_gb}GB, 利用率:${util}%)"
            else
                print_info "  └─ GPU $gpu_id: 显存不足 [极限模式] (实际空闲:$((actual_free_mem/1024))GB < 需求:${required_mem_gb}GB)"
            fi
            continue
        fi
        
        # 非极限模式：检查是否被本脚本占用
        if [ "${SCRIPT_GPU_OCCUPIED[$gpu_id]}" -eq 1 ]; then
            print_info "  └─ GPU $gpu_id: 已被本脚本占用，跳过"
            print_debug "    - 本脚本已在该GPU上分配了 ${script_used_gb}GB 显存"
            continue
        fi
        
        # 独占模式检查（仅在非极限模式且ALLOW_SHARED_GPU=false时）
        if [ "$ALLOW_SHARED_GPU" = false ]; then
            local is_idle=$(is_gpu_fully_idle "$gpu_id")
            if [ "$is_idle" = false ]; then
                print_info "  └─ GPU $gpu_id: 不满足独占条件 (利用率:${util}%, 显存占用:${mem_usage_percent}%), 跳过"
                continue
            fi
        fi
        
        # 检查显存是否充足
        if [ "$free_mem" -ge "$required_mem_mb" ]; then
            available+=("$gpu_id")
            gpu_free_mem_map[$gpu_id]=$free_mem
            gpu_remaining_mem[$gpu_id]=$free_mem
            print_info "  └─ GPU $gpu_id: 可用 (空闲:${free_gb}GB, 利用率:${util}%, 显存占用:${mem_usage_percent}%)"
        else
            print_info "  └─ GPU $gpu_id: 显存不足 (空闲:${free_gb}GB < 需求:${required_mem_gb}GB)"
        fi
    done
    
    # 如果有可用GPU，按剩余显存降序排序（优先选择显存多的）
    if [ ${#available[@]} -gt 0 ]; then
        # 使用冒泡排序按剩余显存降序排列
        local n=${#available[@]}
        for ((i=0; i<n-1; i++)); do
            for ((j=0; j<n-i-1; j++)); do
                local gpu1=${available[$j]}
                local gpu2=${available[$((j+1))]}
                local mem1=${gpu_free_mem_map[$gpu1]}
                local mem2=${gpu_free_mem_map[$gpu2]}
                if [ "$mem1" -lt "$mem2" ]; then
                    # 交换
                    available[$j]=$gpu2
                    available[$((j+1))]=$gpu1
                fi
            done
        done
        print_success "  └─ 找到 ${#available[@]} 张可用GPU（已按剩余显存降序排列）: ${available[*]}"
    else
        print_warning "  └─ 未找到满足条件的可用GPU"
    fi
    
    # 应用预留GPU策略
    local available_count=${#available[@]}
    
    if [ $available_count -eq 0 ]; then
        # 没有可用GPU
        print_info "  └─ 预留策略: 无可用GPU"
        echo ""
        return
    elif [ $available_count -eq $RESERVED_GPUS ]; then
        # 空闲GPU数 = 预留数，使用一张（剩下k-1张）
        print_info "  └─ 预留策略: 可用数=${available_count} = 预留数=${RESERVED_GPUS}，使用1张"
        echo "${available[0]}"
    elif [ $available_count -gt $RESERVED_GPUS ]; then
        # 空闲GPU数 > 预留数，返回可以使用的GPU（用到只剩k张）
        local usable_count=$((available_count - RESERVED_GPUS))
        print_info "  └─ 预留策略: 可用数=${available_count} > 预留数=${RESERVED_GPUS}，可使用${usable_count}张"
        echo "${available[@]:0:$usable_count}"
    else
        # 空闲GPU数 < 预留数，不返回任何GPU（等待更多GPU释放）
        print_warning "  └─ 预留策略: 可用数=${available_count} < 预留数=${RESERVED_GPUS}，不分配"
        echo ""
    fi
}

# 标记GPU为被本脚本占用
mark_gpus_occupied() {
    local required_mem_mb=$1
    shift
    local gpus=("$@")
    
    for gpu_id in "${gpus[@]}"; do
        if [ "$gpu_id" != "none" ] && [ -n "$gpu_id" ]; then
            if [ "$EXTREME_MODE" = true ]; then
                # 极限模式：累加显存使用
                local current_used=${SCRIPT_GPU_USED_MEM[$gpu_id]:-0}
                SCRIPT_GPU_USED_MEM[$gpu_id]=$((current_used + required_mem_mb))
                local new_used_gb=$((SCRIPT_GPU_USED_MEM[$gpu_id] / 1024))
                print_info "标记 GPU $gpu_id [极限模式] (已分配显存: ${new_used_gb}GB, 本次新增: $((required_mem_mb / 1024))GB)"
                print_debug "  GPU $gpu_id 显存追踪: $current_used MB -> ${SCRIPT_GPU_USED_MEM[$gpu_id]} MB"
            else
                # 正常模式：标记为占用
                SCRIPT_GPU_OCCUPIED[$gpu_id]=1
                SCRIPT_GPU_USED_MEM[$gpu_id]=$required_mem_mb
                print_info "标记 GPU $gpu_id 为占用 (分配显存: $((required_mem_mb / 1024))GB)"
            fi
        fi
    done
}

# 释放GPU
mark_gpus_free() {
    local required_mem_mb=$1
    shift
    local gpus=("$@")
    
    for gpu_id in "${gpus[@]}"; do
        if [ "$gpu_id" != "none" ] && [ -n "$gpu_id" ]; then
            if [ "$EXTREME_MODE" = true ]; then
                # 极限模式：减少显存使用
                local current_used=${SCRIPT_GPU_USED_MEM[$gpu_id]:-0}
                SCRIPT_GPU_USED_MEM[$gpu_id]=$((current_used - required_mem_mb))
                # 确保不会小于0
                if [ ${SCRIPT_GPU_USED_MEM[$gpu_id]} -lt 0 ]; then
                    SCRIPT_GPU_USED_MEM[$gpu_id]=0
                fi
                local remaining_gb=$((SCRIPT_GPU_USED_MEM[$gpu_id] / 1024))
                print_info "释放 GPU $gpu_id [极限模式] (剩余已分配: ${remaining_gb}GB, 本次释放: $((required_mem_mb / 1024))GB)"
                print_debug "  GPU $gpu_id 显存追踪: $current_used MB -> ${SCRIPT_GPU_USED_MEM[$gpu_id]} MB"
            else
                # 正常模式：标记为空闲
                SCRIPT_GPU_OCCUPIED[$gpu_id]=0
                SCRIPT_GPU_USED_MEM[$gpu_id]=0
                print_info "释放 GPU $gpu_id"
            fi
        fi
    done
}

# 分配GPU给命令（支持单卡和多卡）
allocate_gpus() {
    local required_mem_gb=$1
    local gpu_count=$2
    local retry_count=0
    
    # 如果不需要GPU
    if [ "$gpu_count" -eq 0 ]; then
        echo "none"
        return 0
    fi
    
    print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_info "开始GPU分配流程"
    print_info "  需求: ${gpu_count}张GPU，每张需${required_mem_gb}GB显存"
    print_info "  极限模式: $([ "$EXTREME_MODE" = true ] && echo "✓ 启用（允许多进程共享GPU）" || echo "✗ 禁用")"
    if [ "$EXTREME_MODE" = false ]; then
        print_info "  独占模式: $([ "$ALLOW_SHARED_GPU" = false ] && echo "✓ 启用（仅用完全空闲GPU）" || echo "✗ 禁用")"
    fi
    print_info "  预留GPU数: ${RESERVED_GPUS}"
    print_info "  调试模式: $([ "$DEBUG_MODE" = true ] && echo "✓ 启用" || echo "✗ 禁用")"
    print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 等待有足够数量的GPU可用
    while true; do
        retry_count=$((retry_count + 1))
        local current_time=$(date '+%Y-%m-%d %H:%M:%S')
        
        echo ""
        print_info "[$current_time] 第 $retry_count 次尝试分配GPU"
        
        local available_gpus=($(get_available_gpus "$required_mem_gb"))
        local available_count=${#available_gpus[@]}
        
        # 统计各种GPU状态
        local total_gpus=0
        local occupied_by_script=0
        local sufficient_mem=0
        local fully_idle=0
        
        for gpu_id in $TARGET_GPUS; do
            total_gpus=$((total_gpus + 1))
            
            if [ "${SCRIPT_GPU_OCCUPIED[$gpu_id]}" -eq 1 ]; then
                occupied_by_script=$((occupied_by_script + 1))
            fi
            
            local free_mem=$(get_gpu_free_memory "$gpu_id")
            local required_mem_mb=$((required_mem_gb * 1024))
            if [ "$free_mem" -ge "$required_mem_mb" ]; then
                sufficient_mem=$((sufficient_mem + 1))
            fi
            
            if [ "$(is_gpu_fully_idle "$gpu_id")" = "true" ]; then
                fully_idle=$((fully_idle + 1))
            fi
        done
        
        print_info "  GPU统计信息:"
        print_info "    • 总GPU数: $total_gpus"
        print_info "    • 本脚本占用: $occupied_by_script 张"
        print_info "    • 显存充足: $sufficient_mem 张"
        print_info "    • 完全空闲: $fully_idle 张"
        print_info "    • 预留要求: $RESERVED_GPUS 张"
        print_info "    • 最终可分配: $available_count 张"
        
        if [ $available_count -ge $gpu_count ]; then
            # 取前N个GPU（已按剩余显存降序排列）
            local allocated=("${available_gpus[@]:0:$gpu_count}")
            echo ""
            print_success "✓ 调度成功！分配 ${gpu_count} 张GPU: ${allocated[*]}"
            
            # 显示每张分配的GPU详情
            for gpu_id in "${allocated[@]}"; do
                local free_mem=$(get_gpu_free_memory "$gpu_id")
                local free_gb=$((free_mem / 1024))
                local util=$(get_gpu_utilization "$gpu_id")
                print_info "  • GPU $gpu_id: ${free_gb}GB可用, 利用率${util}%"
            done
            
            echo "${allocated[@]}"
            return 0
        else
            echo ""
            print_warning "✗ 调度失败：需要 ${gpu_count} 张，可分配 ${available_count} 张"
            
            # 分析失败原因
            local reason=""
            if [ $occupied_by_script -gt 0 ]; then
                reason="${reason}本脚本占用${occupied_by_script}张; "
            fi
            if [ $sufficient_mem -lt $gpu_count ]; then
                reason="${reason}显存不足(仅${sufficient_mem}张满足); "
            fi
            if [ "$ALLOW_SHARED_GPU" = false ] && [ $fully_idle -lt $gpu_count ]; then
                reason="${reason}独占模式要求不满足(仅${fully_idle}张完全空闲); "
            fi
            if [ $RESERVED_GPUS -gt 0 ]; then
                reason="${reason}预留策略限制; "
            fi
            
            print_info "  失败原因: ${reason}"
            print_info "  将在 ${CHECK_INTERVAL} 秒后重试..."
            
            sleep "$CHECK_INTERVAL"
        fi
    done
}

# 替换命令中的GPU变量
replace_gpu_variables() {
    local cmd="$1"
    shift
    local allocated_gpus=("$@")
    
    # 如果不需要GPU
    if [ "${allocated_gpus[0]}" = "none" ]; then
        echo "$cmd"
        return
    fi
    
    # 替换 ${available_gpu} 为第一张GPU
    cmd="${cmd//\$\{available_gpu\}/${allocated_gpus[0]}}"
    
    # 替换 ${available_gpu_0}, ${available_gpu_1}, ... ${available_gpu_29}
    for i in {0..29}; do
        if [ $i -lt ${#allocated_gpus[@]} ]; then
            cmd="${cmd//\$\{available_gpu_${i}\}/${allocated_gpus[$i]}}"
        fi
    done
    
    echo "$cmd"
}

# 等待进程完成（真正的串行）
wait_for_process_completion() {
    local pid=$1
    local gpu_id=$2
    
    print_info "等待进程 (PID: $pid) 完成..."
    
    # 等待主进程完成
    while kill -0 "$pid" 2>/dev/null; do
        sleep 2
    done
    
    # 等待GPU上的所有进程完成（如果使用了GPU）
    if [ "$gpu_id" != "none" ] && command -v nvidia-smi &> /dev/null; then
        print_info "等待 GPU $gpu_id 上的所有计算进程完成..."
        local max_wait=60  # 最多等待60秒
        local waited=0
        
        while [ $waited -lt $max_wait ]; do
            # 检查是否还有进程在使用该GPU
            local gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu_id" 2>/dev/null | grep -v "No running" || true)
            
            if [ -z "$gpu_pids" ]; then
                print_success "GPU $gpu_id 上的所有计算进程已完成"
                break
            fi
            
            sleep 2
            waited=$((waited + 2))
        done
        
        if [ $waited -ge $max_wait ]; then
            print_warning "等待GPU进程超时，继续执行下一个命令"
        fi
    fi
}

# 检查命令是否成功
check_command_success() {
    local exit_code=$1
    local log_file="$2"
    
    # 检查退出码
    if [ $exit_code -ne 0 ]; then
        return 1
    fi
    
    # 检查日志中是否有常见错误关键词
    if [ -f "$log_file" ]; then
        if grep -qi "out of memory\|cuda error\|runtime error\|segmentation fault" "$log_file" 2>/dev/null; then
            return 1
        fi
    fi
    
    return 0
}

# 识别命令类型（bash或python）
identify_command_type() {
    local cmd="$1"
    
    if [[ $cmd =~ ^python[0-9.]* ]] || [[ $cmd =~ python[[:space:]] ]]; then
        echo "python"
    elif [[ $cmd =~ ^bash[[:space:]] ]] || [[ $cmd =~ \.sh ]]; then
        echo "bash"
    else
        echo "unknown"
    fi
}

# =============================================================================
# 主执行逻辑
# =============================================================================

main() {
    # 生成日志文件
    LOG_FILE=$(generate_log_filename)
    
    # 如果是后台运行模式，重定向输出
    if [ "$RUN_IN_BACKGROUND" = true ]; then
        exec > >(tee -a "$LOG_FILE") 2>&1
        print_info "后台运行模式：日志实时输出到 $LOG_FILE"
    else
        exec > >(tee -a "$LOG_FILE") 2>&1
    fi
    
    # 打印欢迎信息
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo "           智能多GPU串行执行脚本 - Smart Multi-GPU Serial Execution"
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo ""
    print_info "日志文件: $LOG_FILE"
    print_info "GPU资源池: $TARGET_GPUS"
    print_info "预留GPU数: $RESERVED_GPUS"
    print_info "显存检查间隔: ${CHECK_INTERVAL}秒"
    print_info "运行模式: $([ "$RUN_IN_BACKGROUND" = true ] && echo "后台运行" || echo "前台运行")"
    print_info "调试模式: $([ "$DEBUG_MODE" = true ] && echo "✓ 启用（详细输出）" || echo "✗ 禁用")"
    echo ""
    print_info "═══ GPU调度策略 ═══"
    print_info "极限资源利用模式: $([ "$EXTREME_MODE" = true ] && echo "✓ 启用" || echo "✗ 禁用")"
    if [ "$EXTREME_MODE" = true ]; then
        print_warning "  └─ 极限模式：允许多进程共享GPU，只要有足够显存就调度"
        print_warning "  └─ 示例：A800(80GB)被占用16GB，进程需12GB，可再放5个进程"
    else
        print_info "GPU共享模式: $([ "$ALLOW_SHARED_GPU" = true ] && echo "允许共享" || echo "独占模式")"
        if [ "$ALLOW_SHARED_GPU" = false ]; then
            print_warning "  └─ 独占模式：仅使用利用率0%且显存占用<3%的GPU"
        fi
    fi
    if [ $MAX_RETRY_TIMES -eq -1 ]; then
        print_info "重试策略: 无限重试"
    else
        print_info "最大重试次数: $MAX_RETRY_TIMES"
    fi
    print_info "GPU选择策略: 优先选择剩余显存多的GPU"
    echo ""
    
    # 检查命令数组是否为空
    if [ ${#COMMANDS[@]} -eq 0 ]; then
        print_error "没有要执行的命令！"
        print_info "请在脚本开头的 COMMANDS 数组中添加命令"
        exit 1
    fi
    
    # 检查GPU可用性
    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU状态检查："
        for gpu_id in $TARGET_GPUS; do
            local free_mem=$(get_gpu_free_memory "$gpu_id")
            local free_gb=$((free_mem / 1024))
            local total_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | head -1)
            local total_gb=$((total_mem / 1024))
            local occupied="${SCRIPT_GPU_OCCUPIED[$gpu_id]}"
            local status_text="空闲"
            [ "$occupied" -eq 1 ] && status_text="占用"
            echo "  GPU $gpu_id: ${free_gb}GB / ${total_gb}GB 可用 [$status_text]"
        done
        echo ""
    fi
    
    # 解析并排序命令（多卡命令优先）
    declare -a parsed_commands
    declare -a command_indices
    
    for i in "${!COMMANDS[@]}"; do
        local parsed=$(parse_command_line "${COMMANDS[$i]}")
        local gpu_count=$(echo "$parsed" | cut -d'|' -f3)
        parsed_commands[$i]="$parsed"
        command_indices[$i]="$i:$gpu_count"
    done
    
    # 按GPU数量排序（降序，多卡优先）
    IFS=$'\n' sorted_indices=($(printf "%s\n" "${command_indices[@]}" | sort -t':' -k2 -rn | cut -d':' -f1))
    unset IFS
    
    # 显示要执行的命令
    print_info "共有 ${#COMMANDS[@]} 个命令待执行（多卡命令优先）："
    echo ""
    for i in "${!sorted_indices[@]}"; do
        local idx=$((i + 1))
        local orig_idx=${sorted_indices[$i]}
        local parsed="${parsed_commands[$orig_idx]}"
        local cmd=$(echo "$parsed" | cut -d'|' -f1)
        local mem=$(echo "$parsed" | cut -d'|' -f2)
        local gpu_count=$(echo "$parsed" | cut -d'|' -f3)
        local cmd_type=$(identify_command_type "$cmd")
        
        echo "  [$idx] $cmd"
        if [ "$gpu_count" -eq 0 ]; then
            echo "      └─ CPU命令（无需GPU）[类型: $cmd_type]"
        else
            echo "      └─ 所需显存: ${mem}GB × ${gpu_count}卡 [类型: $cmd_type]"
        fi
    done
    echo ""
    
    # 记录开始时间
    START_TIME=$(date +%s)
    print_info "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo ""
    
    # 串行执行每个命令（按多卡优先顺序）
    local total_cmds=${#COMMANDS[@]}
    local success_count=0
    local failed_count=0
    
    for i in "${!sorted_indices[@]}"; do
        local orig_idx=${sorted_indices[$i]}
        local cmd_line="${COMMANDS[$orig_idx]}"
        local step_num=$((i + 1))
        
        # 解析命令、显存需求和GPU数量
        local parsed="${parsed_commands[$orig_idx]}"
        local cmd_template=$(echo "$parsed" | cut -d'|' -f1)
        local required_mem_gb=$(echo "$parsed" | cut -d'|' -f2)
        local gpu_count=$(echo "$parsed" | cut -d'|' -f3)
        local cmd_type=$(identify_command_type "$cmd_template")
        
        echo ""
        print_step "$step_num" "$total_cmds" "准备执行命令"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "命令模板: $cmd_template"
        echo "命令类型: $cmd_type"
        if [ "$gpu_count" -eq 0 ]; then
            echo "GPU需求: CPU命令（无需GPU）"
        else
            echo "GPU需求: ${gpu_count}卡 × ${required_mem_gb}GB显存"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 重试逻辑
        local retry_count=0
        local cmd_success=false
        local allocated_gpus=()
        
        # 无限重试或有限重试
        while true; do
            # 检查是否超过重试次数（-1表示无限重试）
            if [ $MAX_RETRY_TIMES -ne -1 ] && [ $retry_count -gt $MAX_RETRY_TIMES ]; then
                break
            fi
            
            if [ $retry_count -gt 0 ]; then
                if [ $MAX_RETRY_TIMES -eq -1 ]; then
                    print_warning "第 $retry_count 次重试（无限重试模式）..."
                else
                    print_warning "第 $retry_count 次重试（最多 $MAX_RETRY_TIMES 次）..."
                fi
            fi
            
            # 分配GPU
            print_info "正在分配 ${gpu_count} 张GPU（显存需求: ${required_mem_gb}GB/卡）..."
            local allocated_gpus_str=$(allocate_gpus "$required_mem_gb" "$gpu_count")
            allocated_gpus=($allocated_gpus_str)
            
            if [ "${allocated_gpus[0]}" = "none" ]; then
                print_info "此命令不需要GPU"
            else
                print_success "已分配GPU: ${allocated_gpus[*]}"
                # 标记GPU为占用
                mark_gpus_occupied "$required_mem_mb" "${allocated_gpus[@]}"
            fi
            
            # 替换命令中的GPU变量
            local cmd=$(replace_gpu_variables "$cmd_template" "${allocated_gpus[@]}")
            print_info "实际执行命令: $cmd"
            
            # 执行命令
            local cmd_start_time=$(date +%s)
            
            # 创建临时日志文件记录命令输出
            local temp_log="${LOG_DIR}/temp_cmd_${step_num}_${retry_count}.log"
            
            # 在后台执行命令，将输出重定向到临时日志
            print_debug "执行命令: $cmd"
            print_debug "临时日志: $temp_log"
            
            eval "$cmd" > "$temp_log" 2>&1 &
            local pid=$!
            
            if ! kill -0 "$pid" 2>/dev/null; then
                print_error "命令启动失败！进程PID $pid 不存在"
                continue
            fi
            
            print_info "命令已启动 (PID: $pid)"
            print_debug "进程状态: $(ps -p $pid -o state= 2>/dev/null || echo "未知")"
            
            # 等待命令完成（真正的串行）
            # 如果使用了多张GPU，需要等待所有GPU上的进程完成
            if [ "${allocated_gpus[0]}" = "none" ]; then
                wait_for_process_completion "$pid" "none"
            else
                # 等待主进程
                wait_for_process_completion "$pid" "${allocated_gpus[0]}"
                
                # 额外等待所有分配的GPU上的进程完成
                for gpu_id in "${allocated_gpus[@]}"; do
                    print_info "确认 GPU $gpu_id 上的进程已完成..."
                    local max_wait=30
                    local waited=0
                    while [ $waited -lt $max_wait ]; do
                        local gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu_id" 2>/dev/null | grep -v "No running" || true)
                        if [ -z "$gpu_pids" ]; then
                            break
                        fi
                        sleep 2
                        waited=$((waited + 2))
                    done
                done
                print_success "所有分配的GPU (${allocated_gpus[*]}) 已释放"
                # 释放GPU
                mark_gpus_free "$required_mem_mb" "${allocated_gpus[@]}"
            fi
            
            # 检查命令是否成功
            wait $pid
            local exit_code=$?
            
            local cmd_end_time=$(date +%s)
            local cmd_duration=$((cmd_end_time - cmd_start_time))
            local cmd_duration_min=$((cmd_duration / 60))
            local cmd_duration_sec=$((cmd_duration % 60))
            
            # 将临时日志追加到主日志
            if [ -f "$temp_log" ]; then
                echo "" >> "$LOG_FILE"
                echo "=== 命令 $step_num 输出 (重试: $retry_count) ===" >> "$LOG_FILE"
                cat "$temp_log" >> "$LOG_FILE"
                echo "=== 命令 $step_num 输出结束 ===" >> "$LOG_FILE"
                echo "" >> "$LOG_FILE"
            fi
            
            # 检查是否成功
            if check_command_success $exit_code "$temp_log"; then
                print_success "命令执行成功 (耗时: ${cmd_duration_min}分${cmd_duration_sec}秒)"
                cmd_success=true
                rm -f "$temp_log"
                break
            else
                print_error "命令执行失败 (退出码: $exit_code, 耗时: ${cmd_duration_min}分${cmd_duration_sec}秒)"
                print_debug "失败命令: $cmd"
                
                # 释放GPU（如果失败）
                if [ "${allocated_gpus[0]}" != "none" ]; then
                    mark_gpus_free "$required_mem_mb" "${allocated_gpus[@]}"
                fi
                
                # 分析错误类型
                local error_type="未知错误"
                if grep -qi "out of memory" "$temp_log" 2>/dev/null; then
                    error_type="显存不足 (OOM)"
                    print_error "  └─ 错误类型: ${error_type}"
                    print_warning "  └─ 建议: 增加显存需求参数或减少batch size"
                elif grep -qi "cuda.*memory" "$temp_log" 2>/dev/null; then
                    error_type="CUDA显存错误"
                    print_error "  └─ 错误类型: ${error_type}"
                elif grep -qi "cuda error" "$temp_log" 2>/dev/null; then
                    error_type="CUDA运行时错误"
                    print_error "  └─ 错误类型: ${error_type}"
                elif grep -qi "runtime error" "$temp_log" 2>/dev/null; then
                    error_type="运行时错误"
                    print_error "  └─ 错误类型: ${error_type}"
                elif grep -qi "segmentation fault" "$temp_log" 2>/dev/null; then
                    error_type="段错误"
                    print_error "  └─ 错误类型: ${error_type}"
                elif grep -qi "importerror\|modulenotfounderror" "$temp_log" 2>/dev/null; then
                    error_type="模块导入错误"
                    print_error "  └─ 错误类型: ${error_type}"
                    print_warning "  └─ 建议: 检查Python环境和依赖"
                elif [ $exit_code -eq 127 ]; then
                    error_type="命令未找到"
                    print_error "  └─ 错误类型: ${error_type}"
                    print_warning "  └─ 建议: 检查命令路径和环境变量"
                elif [ $exit_code -eq 126 ]; then
                    error_type="权限不足"
                    print_error "  └─ 错误类型: ${error_type}"
                    print_warning "  └─ 建议: 检查文件执行权限"
                else
                    print_error "  └─ 错误类型: ${error_type} (退出码: $exit_code)"
                fi
                
                # 显示最后几行错误日志
                if [ -f "$temp_log" ]; then
                    print_debug "  最后10行输出:"
                    if [ "$DEBUG_MODE" = true ]; then
                        tail -10 "$temp_log" | while IFS= read -r line; do
                            echo "    $line"
                        done
                    fi
                fi
                
                # 检查是否是显存不足错误，等待更长时间
                if [[ "$error_type" == *"显存"* ]] || [[ "$error_type" == *"OOM"* ]]; then
                    print_warning "  └─ 等待显存释放..."
                    sleep 10
                fi
                
                rm -f "$temp_log"
                retry_count=$((retry_count + 1))
                
                # 检查是否还能重试
                if [ $MAX_RETRY_TIMES -eq -1 ]; then
                    print_warning "将在 10 秒后重试（无限重试模式）..."
                    sleep 10
                elif [ $retry_count -le $MAX_RETRY_TIMES ]; then
                    print_warning "将在 10 秒后重试..."
                    sleep 10
                else
                    print_error "已达到最大重试次数 ($MAX_RETRY_TIMES)，放弃此命令"
                fi
            fi
        done
        
        # 统计成功/失败
        if [ "$cmd_success" = true ]; then
            success_count=$((success_count + 1))
        else
            failed_count=$((failed_count + 1))
        fi
        
        # 在命令之间添加缓冲时间
        if [ $step_num -lt $total_cmds ]; then
            print_info "等待 5 秒后执行下一个命令..."
            sleep 5
        fi
    done
    
    # 打印执行总结
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════"
    print_info "所有命令执行完成"
    echo "═══════════════════════════════════════════════════════════════════════════"
    
    END_TIME=$(date +%s)
    TOTAL_DURATION=$((END_TIME - START_TIME))
    TOTAL_DURATION_MIN=$((TOTAL_DURATION / 60))
    TOTAL_DURATION_SEC=$((TOTAL_DURATION % 60))
    
    echo ""
    print_info "执行统计："
    echo "  • GPU资源池: $TARGET_GPUS"
    echo "  • 总命令数: $total_cmds"
    echo "  • 成功: ${GREEN}${success_count}${NC}"
    echo "  • 失败: ${RED}${failed_count}${NC}"
    echo "  • 成功率: $(awk "BEGIN {printf \"%.1f%%\", ($success_count/$total_cmds)*100}")"
    echo "  • 总耗时: ${TOTAL_DURATION_MIN}分${TOTAL_DURATION_SEC}秒"
    echo "  • 平均每命令: $(awk "BEGIN {printf \"%.1f\", $TOTAL_DURATION/$total_cmds}")秒"
    echo "  • 开始时间: $(date -d @$START_TIME '+%Y-%m-%d %H:%M:%S')"
    echo "  • 结束时间: $(date -d @$END_TIME '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 显示最终GPU状态
    if command -v nvidia-smi &> /dev/null; then
        print_info "最终GPU状态："
        for gpu_id in $TARGET_GPUS; do
            local free_mem=$(get_gpu_free_memory "$gpu_id")
            local free_gb=$((free_mem / 1024))
            echo "  GPU $gpu_id: ${free_gb}GB 可用"
        done
        echo ""
    fi
    
    print_info "完整日志已保存至: $LOG_FILE"
    echo "═══════════════════════════════════════════════════════════════════════════"
    
    if [ $failed_count -gt 0 ]; then
        exit 1
    fi
}

# =============================================================================
# 脚本入口
# =============================================================================

# 检查是否在正确的目录
if [ ! -f "discovering.py" ]; then
    print_error "请在项目根目录下运行此脚本"
    exit 1
fi

# 如果是后台运行模式，使用nohup
if [ "$RUN_IN_BACKGROUND" = true ]; then
    # 检查是否已经在后台运行
    if [ -z "$QUICK_START_BACKGROUND" ]; then
        export QUICK_START_BACKGROUND=1
        nohup bash "$0" "$@" > /dev/null 2>&1 &
        echo "脚本已在后台启动 (PID: $!)"
        echo "查看日志: tail -f $(generate_log_filename)"
        exit 0
    fi
fi

# 执行主逻辑
main

exit 0
