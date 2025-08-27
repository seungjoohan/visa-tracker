#!/usr/bin/env python3
import asyncio
import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tasks.daily_processor import run_daily_process

if __name__ == "__main__":
    asyncio.run(run_daily_process())