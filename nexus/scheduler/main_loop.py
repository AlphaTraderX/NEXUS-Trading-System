"""
NEXUS Main Scheduler Loop

The foreman of the operation - tells scanners when to run.
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from nexus.config.settings import settings
from nexus.core.models import Opportunity
from nexus.data.massive import MassiveProvider
from nexus.data.oanda import OANDAProvider
from nexus.scanners.orchestrator import ScannerOrchestrator
from nexus.scheduler.market_hours import MarketHours

# Conditional import for Windows compatibility
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

UTC = ZoneInfo("UTC")


class NexusScheduler:
    """
    Main NEXUS scheduler - runs continuously and triggers scans.
    
    Responsibilities:
    1. Connect to data providers when markets open
    2. Run scanners at appropriate intervals
    3. Handle market hours awareness
    4. Graceful shutdown on interrupt
    """
    
    def __init__(
        self,
        scan_interval: int = 60,  # seconds between scans
        verbose: bool = True,
    ):
        self.scan_interval = scan_interval
        self.verbose = verbose
        self.running = False
        
        # Data providers (initialized on start)
        self.polygon: Optional[MassiveProvider] = None
        self.oanda: Optional[OANDAProvider] = None
        
        # Orchestrator (initialized with providers)
        self.orchestrator: Optional[ScannerOrchestrator] = None
        
        # Stats
        self.scan_count = 0
        self.opportunities_found = 0
        self.start_time: Optional[datetime] = None
    
    async def initialize_providers(self) -> bool:
        """Initialize and connect to data providers."""
        try:
            # Initialize Polygon/Massive for US stocks
            self.polygon = MassiveProvider(api_key=settings.polygon_api_key)
            await self.polygon.connect()
            self._log("Connected to Polygon (US Stocks)")
            
            # Initialize OANDA for forex
            self.oanda = OANDAProvider(
                api_key=settings.oanda_api_key,
                account_id=settings.oanda_account_id,
                practice=True  # Use demo account
            )
            await self.oanda.connect()
            self._log("Connected to OANDA (Forex)")
            
            # Initialize orchestrator with providers
            self.orchestrator = ScannerOrchestrator(
                stock_provider=self.polygon,
                forex_provider=self.oanda,
            )
            self._log("Scanner orchestrator initialized")
            
            return True
            
        except Exception as e:
            self._log(f"Failed to initialize providers: {e}", level="error")
            return False
    
    async def shutdown_providers(self):
        """Gracefully disconnect from data providers."""
        try:
            if self.polygon:
                await self.polygon.disconnect()
                self._log("Disconnected from Polygon")
            
            if self.oanda:
                await self.oanda.disconnect()
                self._log("Disconnected from OANDA")
                
        except Exception as e:
            self._log(f"Error during shutdown: {e}", level="error")
    
    async def run_scan_cycle(self) -> list[Opportunity]:
        """
        Run one complete scan cycle.
        
        Checks market hours and runs appropriate scanners.
        """
        opportunities = []
        
        # Get current market status
        status = MarketHours.get_market_status()
        
        # Log scan start
        self._log(f"--- Scan #{self.scan_count} ---")
        
        # Log market status on first scan, then every 10 scans
        if self.scan_count == 1 or self.scan_count % 10 == 0:
            self._log_market_status(status)
        
        # Run orchestrator if any market is open
        any_market_open = any(status["markets"].values())
        
        if not any_market_open:
            self._log("Markets closed - skipping scanners")
            return opportunities
        
        # Log which markets are being scanned
        open_markets = [m for m, is_open in status["markets"].items() if is_open]
        self._log(f"Scanning: {', '.join(open_markets)}")
        
        # Run the scan
        try:
            opportunities = await self.orchestrator.run_scan_cycle()
            
            if opportunities:
                self.opportunities_found += len(opportunities)
                self._log_opportunities(opportunities)
                
        except Exception as e:
            self._log(f"Scan cycle error: {e}", level="error")
        
        # Log scan completion
        if opportunities:
            self._log(f"Found {len(opportunities)} opportunities!")
        else:
            self._log("No opportunities this cycle")
        
        return opportunities
    
    async def start(self):
        """
        Start the scheduler main loop.
        
        Runs continuously until interrupted.
        """
        self.running = True
        self.start_time = datetime.now(UTC)
        
        self._print_banner()
        
        # Initialize providers
        if not await self.initialize_providers():
            self._log("Failed to initialize. Exiting.", level="error")
            return
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self._log(f"Scheduler started. Scanning every {self.scan_interval} seconds.")
        
        # Main loop
        try:
            while self.running:
                self.scan_count += 1
                
                # Run scan cycle
                await self.run_scan_cycle()
                
                # Wait for next cycle
                if self.running:
                    await asyncio.sleep(self.scan_interval)
                    
        except asyncio.CancelledError:
            self._log("Scheduler cancelled")
        except KeyboardInterrupt:
            self._log("Keyboard interrupt received")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the scheduler gracefully."""
        self._log("Shutting down scheduler...")
        self.running = False
        
        await self.shutdown_providers()
        
        self._print_summary()
    
    def _setup_signal_handlers(self):
        """Set up handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            self._log(f"Received signal {signum}")
            self.running = False
        
        # Handle Ctrl+C
        signal.signal(signal.SIGINT, handle_signal)
        
        # Handle termination (not available on Windows)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, handle_signal)
    
    def _print_banner(self):
        """Print startup banner."""
        banner = """
============================================================
    _   _________  ____  _______
   / | / / ____/ |/ / / / / ___/
  /  |/ / __/  |   / / / /\\__ \\
 / /|  / /___ /   / /_/ /___/ /
/_/ |_/_____//_/|_\\____//____/

        AUTOMATED TRADING SYSTEM
============================================================
"""
        print(banner)
        print(f"  Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"  Mode: {'PAPER' if settings.paper_trading else 'LIVE'}")
        print(f"  Scan Interval: {self.scan_interval}s")
        print("============================================================\n")
    
    def _print_summary(self):
        """Print session summary on shutdown."""
        if not self.start_time:
            return
        
        runtime = datetime.now(UTC) - self.start_time
        hours = runtime.total_seconds() / 3600
        
        print("\n============================================================")
        print("                    SESSION SUMMARY")
        print("============================================================")
        print(f"  Runtime: {runtime}")
        print(f"  Scans completed: {self.scan_count}")
        print(f"  Opportunities found: {self.opportunities_found}")
        print(f"  Avg per hour: {self.opportunities_found / max(hours, 0.01):.1f}")
        print("============================================================\n")
    
    def _log_market_status(self, status: dict):
        """Log current market status."""
        markets = status["markets"]
        sessions = status["active_sessions"]
        
        market_str = " | ".join([
            f"Forex: {'OPEN' if markets['forex'] else 'CLOSED'}",
            f"US: {'OPEN' if markets['us_stocks'] else 'CLOSED'}",
            f"UK: {'OPEN' if markets['uk_stocks'] else 'CLOSED'}",
        ])
        
        session_str = ", ".join(sessions) if sessions else "None"
        
        self._log(f"Markets: {market_str}")
        self._log(f"Active sessions: {session_str}")
    
    def _log_opportunities(self, opportunities: list[Opportunity]):
        """Log found opportunities."""
        print("\n" + "=" * 60)
        print(f"  OPPORTUNITIES FOUND: {len(opportunities)}")
        print("=" * 60)
        
        for opp in opportunities:
            direction = "LONG" if opp.direction.value == "long" else "SHORT"
            print(f"\n  {opp.primary_edge.value} | {opp.symbol}")
            print(f"  Direction: {direction}")
            print(f"  Entry: ${opp.entry_price:.2f}")
            print(f"  Stop: ${opp.stop_loss:.2f}")
            print(f"  Target: ${opp.take_profit:.2f}")
            print(f"  R:R: {opp.risk_reward_ratio:.2f}")
        
        print("\n" + "=" * 60 + "\n")
    
    def _log(self, message: str, level: str = "info"):
        """Log a message with timestamp."""
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        prefix = {
            "info": "INFO ",
            "error": "ERROR",
            "warn": "WARN ",
        }.get(level, "INFO ")
        
        if self.verbose:
            print(f"{timestamp} | {prefix} | {message}")


async def run_scheduler(scan_interval: int = 60):
    """Convenience function to run the scheduler."""
    scheduler = NexusScheduler(scan_interval=scan_interval)
    await scheduler.start()


# Allow running directly
if __name__ == "__main__":
    asyncio.run(run_scheduler())
