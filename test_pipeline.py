from nexus.intelligence import CostEngine, OpportunityScorer, RegimeDetector, ReasoningEngine
from nexus.core.enums import EdgeType, Direction, Market, MarketRegime
from nexus.core.models import Opportunity
from datetime import datetime, timezone
import pandas as pd
import numpy as np

opp = Opportunity(
    id='test', detected_at=datetime.now(timezone.utc), scanner='Test',
    symbol='SPY', market=Market.US_STOCKS, direction=Direction.LONG,
    entry_price=500.0, stop_loss=495.0, take_profit=512.50,
    primary_edge=EdgeType.VWAP_DEVIATION, secondary_edges=[], edge_data={}
)

np.random.seed(42)
bars = pd.DataFrame({
    'open': 500 + np.random.randn(100), 'high': 502 + np.random.randn(100),
    'low': 498 + np.random.randn(100), 'close': 500 + np.cumsum(np.random.randn(100) * 0.5),
    'volume': np.random.randint(1000000, 5000000, 100)
})

regime = RegimeDetector().detect_regime(bars)
costs = CostEngine().calculate_costs('SPY', Market.US_STOCKS, 'ibkr', 5000, 1)
cost_analysis = CostEngine().calculate_net_edge(0.18, costs)
scored = OpportunityScorer().score(opp, {'alignment': 'STRONG_BULLISH'}, 1.8, regime.regime, cost_analysis)

print(f"Regime: {regime.regime.value}")
print(f"Score: {scored.score}/100 | Tier: {scored.tier}")
print(f"Net Edge: {cost_analysis['net_edge']:.4f}%")
print("Pipeline working!")
