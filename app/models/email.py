from pydantic import BaseModel
from typing import List
from app.models.news import AnalyzedNews

class EmailContent(BaseModel):
    subject: str
    html_content: str
    recipients: List[str]
    analyzed_news: List[AnalyzedNews]