"""
investment_advisor — modular stock recommendation engine.

Quick start:
    from investment_advisor import MarketDataAggregator, InvestmentAdvisor

    aggregator = MarketDataAggregator()
    advisor    = InvestmentAdvisor()

    bundle = aggregator.collect_all(["XLK", "XLF", "XLV"])
    recs   = advisor.recommend(bundle)
    for r in recs:
        print(r)
"""

from .aggregator import MarketDataAggregator
from .advisor import InvestmentAdvisor, Recommendation
from .data.finnhub_client import FinnhubClient
from .data.polygon_client import PolygonClient
from .data.fred_client import FredClient

__all__ = [
    "MarketDataAggregator",
    "InvestmentAdvisor",
    "Recommendation",
    "FinnhubClient",
    "PolygonClient",
    "FredClient",
]
