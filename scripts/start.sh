#!/usr/bin/env bash
# 启动 Interview 面试系统（开发模式）
set -e
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

exec .venv/bin/python -m web.server
