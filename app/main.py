from fastapi import FastAPI, BackgroundTasks
import logging
from datetime import datetime

from app.core.config import settings
from app.tasks.daily_processor import run_daily_process

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Visa News Tracker",
    description="AI-powered visa policy news tracking and analysis system",
    version="1.0.0"
)

# 현재는 API 라우터 없이 시작 (나중에 추가 가능)
# app.include_router(news.router, prefix="/api/v1/news", tags=["news"])
# app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
# app.include_router(email.router, prefix="/api/v1/email", tags=["email"])

@app.get("/")
async def root():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "service": "Visa News Tracker",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """상세 헬스 체크"""
    return {
        "status": "healthy",
        "checks": {
            "api": "ok",
            "config": "loaded" if settings else "error"
        }
    }

@app.post("/process/daily")
async def trigger_daily_process(background_tasks: BackgroundTasks):
    """수동으로 일일 처리 트리거 (테스트/디버깅용)"""
    logger.info("Manual daily process triggered")
    background_tasks.add_task(run_daily_process)
    return {
        "message": "Daily processing started", 
        "timestamp": datetime.now().isoformat()
    }