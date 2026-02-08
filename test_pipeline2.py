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

detector = RegimeDetector()
regime_result = detector.detect_regime(bars)

# Check what type it is
print(f"Type: {type(regime_result)}")
print(f"Regime result: {regime_result}")

# Try attribute access instead of dict access
if hasattr(regime_result, 'regime'):
    print(f"Regime (attr): {regime_result.regime}")
    regime_value = regime_result.regime
else:
    print(f"Regime (dict): {regime_result['regime']}")
    regime_value = regime_result['regime']

costs = CostEngine().calculate_costs('SPY', Market.US_STOCKS, 'ibkr', 5000, 1)
cost_analysis = CostEngine().calculate_net_edge(0.18, costs)

print(f"Net Edge: {cost_analysis['net_edge']:.4f}%")
print("Pipeline working!")
