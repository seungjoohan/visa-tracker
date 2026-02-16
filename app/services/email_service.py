import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jinja2 import Template
from typing import List, Dict
from datetime import datetime
import logging
from pathlib import Path

from app.models.news import NewsArticle, ImportanceLevel, AnalyzedNews
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.smtp_server = settings.smtp_server
        self.smtp_port = settings.smtp_port
        self.email_user = settings.email_user
        self.email_password = settings.email_password
        self.recipients = settings.recipients
        
        # 이메일 템플릿 로드
        template_path = settings.config_dir / "email_template.html"
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                self.email_template = Template(f.read())
        else:
            self.email_template = Template(self._get_default_template())
    
    async def send_daily_summary(self, analyzed_articles: List[NewsArticle]) -> bool:
        """일일 뉴스 요약 이메일 전송"""
        try:
            if not self.recipients:
                logger.error("No email recipients configured")
                return False
            
            if not analyzed_articles:
                logger.info("No articles to send")
                return True
            
            # 기사들을 중요도별로 분류
            categorized_articles = self._categorize_articles(analyzed_articles)
            
            # 이메일 내용 생성
            subject = f"Visa News Summary - {datetime.now().strftime('%Y-%m-%d')}"
            html_content = self._generate_email_content(categorized_articles)
            
            # 이메일 전송
            success = await self._send_email(subject, html_content)
            
            if success:
                logger.info(f"Daily summary email sent successfully to {len(self.recipients)} recipients")
                return True
            else:
                logger.error("Failed to send daily summary email")
                return False
                
        except Exception as e:
            logger.error(f"Error sending daily summary email: {e}")
            return False
    
    def _categorize_articles(self, articles: List[NewsArticle]) -> Dict[ImportanceLevel, List[NewsArticle]]:
        """기사들을 중요도별로 분류"""
        categorized = {
            ImportanceLevel.NEEDS_ATTENTION: [],
            ImportanceLevel.GOOD_TO_KNOW: [],
            ImportanceLevel.NO_ATTENTION_REQUIRED: []
        }
        
        for article in articles:
            if article.importance_level:
                categorized[article.importance_level].append(article)
            else:
                # 중요도가 설정되지 않은 경우 기본값
                categorized[ImportanceLevel.NO_ATTENTION_REQUIRED].append(article)
        
        return categorized
    
    def summarize_news_content():
        # summarize each article
        pass


    def _generate_email_content(self, categorized_articles: Dict[ImportanceLevel, List[NewsArticle]]) -> str:
        """이메일 HTML 콘텐츠 생성"""
        total_articles = sum(len(articles) for articles in categorized_articles.values())

        na = categorized_articles[ImportanceLevel.NEEDS_ATTENTION]
        gtk = categorized_articles[ImportanceLevel.GOOD_TO_KNOW]
        no_att = categorized_articles[ImportanceLevel.NO_ATTENTION_REQUIRED]

        # Build a one-line daily summary
        daily_summary = self._build_daily_summary(na, gtk, no_att)

        context = {
            'date': datetime.now().strftime('%Y/%m/%d'),
            'total_articles': total_articles,
            'daily_summary': daily_summary,
            'needs_attention': na,
            'good_to_know': gtk,
            'no_attention': no_att,
        }

        return self.email_template.render(**context)

    def _build_daily_summary(
        self,
        needs_attention: List[NewsArticle],
        good_to_know: List[NewsArticle],
        no_attention: List[NewsArticle],
    ) -> str:
        """Build a one-line summary across all categories."""
        all_articles = needs_attention + good_to_know + no_attention
        if not all_articles:
            return "No visa-related news today."

        topics = set()
        for a in all_articles:
            if a.analysis_result and hasattr(a.analysis_result, 'affected_visa_types'):
                for vt in a.analysis_result.affected_visa_types:
                    topics.add(vt)
            else:
                for kw in (a.keywords or []):
                    if kw.upper() in ("H1B", "H-1B", "F1", "F-1", "GREEN CARD", "L1", "L-1", "O1", "O-1"):
                        topics.add(kw.upper().replace("1B", "-1B").replace("F1", "F-1"))

        parts = []
        if needs_attention:
            parts.append(f"{len(needs_attention)} urgent")
        if good_to_know:
            parts.append(f"{len(good_to_know)} worth noting")
        if no_attention:
            parts.append(f"{len(no_attention)} general")

        count_str = ", ".join(parts)
        topic_str = f" Topics: {', '.join(sorted(topics))}." if topics else ""
        return f"{len(all_articles)} articles today ({count_str}).{topic_str}"
    
    async def _send_email(self, subject: str, html_content: str) -> bool:
        """실제 이메일 전송"""
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_user
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = subject
            
            # HTML 파트 추가
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            # SMTP 서버 연결 및 전송
            async with aiosmtplib.SMTP(
                hostname=self.smtp_server,
                port=self.smtp_port,
                start_tls=True
            ) as smtp:
                await smtp.login(self.email_user, self.email_password)
                await smtp.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"SMTP send error: {e}")
            return False
    
    def _get_default_template(self) -> str:
        """기본 이메일 템플릿"""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Visa News Summary</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        .header { background-color: #2c3e50; color: white; padding: 20px; text-align: center; }
        .section { margin: 20px 0; }
        .needs-attention { border-left: 5px solid #e74c3c; padding-left: 15px; }
        .good-to-know { border-left: 5px solid #f39c12; padding-left: 15px; }
        .no-attention { border-left: 5px solid #95a5a6; padding-left: 15px; }
        .article { margin: 15px 0; padding: 15px; border: 1px solid #ecf0f1; border-radius: 5px; }
        .article-title { font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
        .article-summary { color: #555; margin-bottom: 10px; }
        .article-link { color: #3498db; text-decoration: none; }
        .article-meta { font-size: 0.9em; color: #7f8c8d; }
        .summary-stats { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📰 비자 뉴스 요약</h1>
            <p>{{ date }}</p>
        </div>
        
        <div class="summary-stats">
            <h3>📊 오늘의 요약</h3>
            <p><strong>총 {{ total_articles }}개</strong>의 비자 관련 뉴스를 분석했습니다.</p>
            <ul>
                <li>🚨 주의 필요: {{ needs_attention|length }}개</li>
                <li>📌 알아두면 좋음: {{ good_to_know|length }}개</li>
                <li>📰 일반 정보: {{ no_attention|length }}개</li>
            </ul>
        </div>
        
        {% if needs_attention %}
        <div class="section needs-attention">
            <h2>🚨 주의 필요 (Needs Attention)</h2>
            {% for article in needs_attention %}
            <div class="article">
                <div class="article-title">{{ article.title }}</div>
                <div class="article-summary">{{ article.summary or article.description or "요약 없음" }}</div>
                <div class="article-meta">
                    <a href="{{ article.url }}" class="article-link" target="_blank">원문 보기</a> | 
                    {{ article.published_at.strftime('%Y-%m-%d %H:%M') }} | 
                    관련도: {{ "%.0f"|format(article.relevance_score * 100) }}%
                </div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if good_to_know %}
        <div class="section good-to-know">
            <h2>📌 알아두면 좋음 (Good to Know)</h2>
            {% for article in good_to_know %}
            <div class="article">
                <div class="article-title">{{ article.title }}</div>
                <div class="article-summary">{{ article.summary or article.description or "요약 없음" }}</div>
                <div class="article-meta">
                    <a href="{{ article.url }}" class="article-link" target="_blank">원문 보기</a> | 
                    {{ article.published_at.strftime('%Y-%m-%d %H:%M') }} | 
                    관련도: {{ "%.0f"|format(article.relevance_score * 100) }}%
                </div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if no_attention %}
        <div class="section no-attention">
            <h2>📰 일반 정보 (No Attention Required)</h2>
            {% for article in no_attention %}
            <div class="article">
                <div class="article-title">{{ article.title }}</div>
                <div class="article-summary">{{ article.summary or article.description or "요약 없음" }}</div>
                <div class="article-meta">
                    <a href="{{ article.url }}" class="article-link" target="_blank">원문 보기</a> | 
                    {{ article.published_at.strftime('%Y-%m-%d %H:%M') }} | 
                    관련도: {{ "%.0f"|format(article.relevance_score * 100) }}%
                </div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <div style="text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 0.9em;">
            <p>이 이메일은 비자 뉴스 트래커에 의해 자동으로 생성되었습니다.</p>
        </div>
    </div>
</body>
</html>
        """
