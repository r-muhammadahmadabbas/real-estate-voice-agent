# Monitoring & Maintenance Plan

## 1. Monitoring Checklist

| Metric | Target | Alert Threshold | Frequency | Action |
|--------|--------|-----------------|-----------|--------|
| API Uptime | 99.9% | < 99.5% | Real-time | Investigate + restart |
| Avg Response Latency | < 2s | > 3s | Real-time | Check LLM + STT |
| STT Accuracy | > 90% | < 85% | Daily | Retrain Whisper |
| TTS Quality | Good | Poor | Weekly | Switch provider |
| Booking Success Rate | > 90% | < 85% | Daily | Check calendar API |
| Email Delivery | > 99% | < 95% | Daily | Check SMTP |
| RAG Accuracy | > 90% | < 85% | Weekly | Update knowledge base |
| Hallucination Rate | < 2% | > 5% | Weekly | Tune prompts |
| Off-Topic Leak Rate | 0% | > 0% | Real-time | Strengthen guardrails |
| Prompt Injection Attempts | 0 | > 0 | Real-time | Alert security |
| Error Rate | < 1% | > 2% | Hourly | Check logs |
| Cost per Call | < $0.05 | > $0.10 | Daily | Optimize |

## 2. Weekly Maintenance Schedule

| Day | Task | Duration |
|-----|------|----------|
| Monday | Review weekend metrics, check alerts | 30 min |
| Tuesday | Update property database (new listings) | 1 hour |
| Wednesday | Update FAQ knowledge base | 30 min |
| Thursday | Review failed conversations, improve prompts | 1 hour |
| Friday | Backup database, review security logs | 30 min |

## 3. Retraining Loop

```
New match results → Update feature table → Retrain model → Evaluate → Deploy
```

| Trigger | Action |
|---------|--------|
| 100 new conversations | Retrain intent classifier |
| 50 new properties | Update vector embeddings |
| Accuracy drops 5% | Full model retrain |
| New FAQs added | Update RAG knowledge base |

## 4. Backup Strategy

| Data | Frequency | Retention | Location |
|------|-----------|-----------|----------|
| SQLite DB | Daily | 30 days | Cloud storage |
| ChromaDB | Weekly | 8 weeks | Cloud storage |
| Call logs | Daily | 90 days | Cloud storage |
| Config/Secrets | On change | Infinite | Vault |
| Model artifacts | On deploy | 10 versions | Cloud storage |

## 5. Security Review

| Task | Frequency |
|------|-----------|
| Prompt injection testing | Weekly |
| API key rotation | Monthly |
| Access log review | Weekly |
| Dependency vulnerability scan | Monthly |
| Penetration test | Quarterly |