#!/usr/bin/env bash
# 在「用户目录的 tmp」下执行 pip 安装，避免默认 /tmp 占满导致 No space left on device
# 用法: bash install_with_user_tmp.sh  或  source install_with_user_tmp.sh 后执行 pip install -r requirements.txt

export TMPDIR="${TMPDIR:-$HOME/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR"
echo "Using TMPDIR=$TMPDIR"

# 若在 fl1 环境中安装 f_learning 依赖：
# pip install -r requirements.txt
# 若需指定环境，可先: conda activate fl1
pip install -r "$(dirname "$0")/requirements.txt"
