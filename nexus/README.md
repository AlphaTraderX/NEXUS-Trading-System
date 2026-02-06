# NEXUS Trading System v2.1

## Multi-Asset Automated Trading | 13 Validated Edges | AI-Powered Signals

---

## Overview

NEXUS is a professional-grade trading system that:
- Scans 34 instruments across stocks, forex, and futures
- Detects 13 academically validated statistical edges
- Generates AI-reasoned trading signals
- Manages risk with institutional-grade controls
- Delivers signals via Discord and Telegram

## Target Performance

| Mode | Monthly Target | Max Drawdown |
|------|---------------|--------------|
| Conservative | 6-10% | 10% |
| Standard | 10-12% | 10% |
| Aggressive | 12-15% | 10% |

## Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/nexus.git
cd nexus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Run in paper trading mode
python main.py
```

## Architecture
```
nexus/
├── config/       # Configuration & settings
├── core/         # Data models & enums
├── data/         # Broker connections & data feeds
├── scanners/     # 13 edge detection scanners
├── intelligence/ # Cost engine, scoring, AI reasoning
├── risk/         # Position sizing, circuit breakers, kill switch
├── execution/    # Signal generation & order management
├── delivery/     # Discord & Telegram alerts
├── storage/      # Database models & repositories
├── monitoring/   # Health checks & metrics
├── api/          # FastAPI endpoints & WebSocket
└── tests/        # Test suites
```

## Validated Edges

| Tier | Edge | Expected Return |
|------|------|-----------------|
| A | Insider Cluster | 0.30-0.50% |
| A | VWAP Deviation | 0.15-0.20% |
| A | Turn of Month | 0.25-0.35% |
| B | Gap Fill | 0.15-0.20% |
| B | RSI Extreme | 0.12-0.18% |
| B | Power Hour | 0.10-0.15% |

## License

Private - All Rights Reserved

## Author

Built with Claude (Architect) + Cursor (Builder)
