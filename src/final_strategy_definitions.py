from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Side = Literal["long", "short", "both"]

FilterName = Literal[
    "none",
    "asset_above_ema200",
    "asset_below_ema200",
    "btc_above_ema200",
    "btc_below_ema200",
    "volume_z_gt_0",
    "volume_z_gt_1",
]


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    name: str
    category: str
    side: Side
    lookback_hours: float | None
    horizon_minutes: int | None
    momentum_lower: float | None
    momentum_upper: float | None
    vol_lower: float | None
    vol_upper: float | None
    filter_name: FilterName
    allowed_products: tuple[str, ...] | None = None
    excluded_products: tuple[str, ...] | None = None
    description: str = ""


def get_final_strategies() -> list[StrategyDefinition]:
    """
    Expanded final strategy set.

    Methodology labels:

    - exploratory_competition_candidate:
        Strong but contaminated by earlier iterative competition research.

    - systematic_grid_candidate:
        Candidate discovered through the broader systematic grid research.
        Cleaner than V3 because it was not manually asset-pruned through the competition loop,
        but not as strict as the final development-only rediscovery set.

    - rediscovered_clean_candidate:
        Selected using only pre-holdout development data, then tested on final holdout.

    - ablation_baseline:
        Used to test whether volatility/trend filters added value.

    - benchmark:
        Passive buy-and-hold comparison.
    """

    return [
        # ─────────────────────────────────────────────────────────────────────
        # Exploratory / competition-derived
        # ─────────────────────────────────────────────────────────────────────
        StrategyDefinition(
            strategy_id="A_v3_baseline",
            name="Exploratory V3 Baseline",
            category="exploratory_competition_candidate",
            side="both",
            lookback_hours=24.0,
            horizon_minutes=240,
            momentum_lower=None,
            momentum_upper=None,
            vol_lower=40.0,
            vol_upper=50.0,
            filter_name="none",
            description=(
                "Exploratory competition-derived strategy. "
                "LONG if 24h return < -4%, vol 40-50, assets BTC/ETH/XRP/DOGE. "
                "SHORT if 24h return in [0.25%, 1%), vol 40-50, assets BTC/ETH/SOL/DOGE. "
                "Strong result, but contaminated by iterative earlier research."
            ),
        ),

        # ─────────────────────────────────────────────────────────────────────
        # Earlier systematic grid-derived candidates
        # ─────────────────────────────────────────────────────────────────────
        StrategyDefinition(
            strategy_id="B_extreme_selloff_rebound_grid",
            name="B Grid Extreme Selloff Rebound",
            category="systematic_grid_candidate",
            side="long",
            lookback_hours=24.0,
            horizon_minutes=480,
            momentum_lower=None,
            momentum_upper=-6.0,
            vol_lower=20.0,
            vol_upper=50.0,
            filter_name="none",
            description=(
                "Systematic grid-derived candidate. LONG when 24h return < -6% "
                "and volatility percentile is 20-50. Hold for 8h."
            ),
        ),
        StrategyDefinition(
            strategy_id="C_upside_continuation_below_trend",
            name="C Upside Continuation Below Trend",
            category="systematic_grid_candidate",
            side="long",
            lookback_hours=6.0,
            horizon_minutes=720,
            momentum_lower=4.0,
            momentum_upper=6.0,
            vol_lower=None,
            vol_upper=None,
            filter_name="asset_below_ema200",
            description=(
                "Systematic grid-derived candidate. LONG when 6h return is between +4% and +6% "
                "and the asset is below EMA200. Hold for 12h. "
                "Included because it produced one of the strongest earlier final-year results."
            ),
        ),
        StrategyDefinition(
            strategy_id="D_low_vol_overextension_short_grid",
            name="D Low-Vol Overextension Short",
            category="systematic_grid_candidate",
            side="short",
            lookback_hours=12.0,
            horizon_minutes=720,
            momentum_lower=6.0,
            momentum_upper=None,
            vol_lower=0.0,
            vol_upper=30.0,
            filter_name="asset_above_ema200",
            description=(
                "Systematic grid-derived short candidate. SHORT when 12h return > +6%, "
                "volatility percentile is 0-30, and asset is above EMA200. Hold for 12h."
            ),
        ),

        # ─────────────────────────────────────────────────────────────────────
        # Clean rediscovered candidates
        # ─────────────────────────────────────────────────────────────────────
        StrategyDefinition(
            strategy_id="R1_shortterm_extreme_selloff_rebound",
            name="R1 Short-Term Extreme Selloff Rebound",
            category="rediscovered_clean_candidate",
            side="long",
            lookback_hours=24.0,
            horizon_minutes=720,
            momentum_lower=None,
            momentum_upper=-6.0,
            vol_lower=40.0,
            vol_upper=50.0,
            filter_name="btc_below_ema200",
            description=(
                "Clean rediscovered candidate. LONG when 24h return < -6%, "
                "volatility percentile is 40-50, and BTC is below EMA200. Hold for 12h."
            ),
        ),
        StrategyDefinition(
            strategy_id="R2_highvol_upside_continuation_btc_below",
            name="R2 High-Vol Upside Continuation Below BTC Trend",
            category="rediscovered_clean_candidate",
            side="long",
            lookback_hours=6.0,
            horizon_minutes=720,
            momentum_lower=6.0,
            momentum_upper=None,
            vol_lower=70.0,
            vol_upper=100.0,
            filter_name="btc_below_ema200",
            description=(
                "Clean rediscovered candidate. LONG when 6h return > +6%, "
                "volatility percentile is 70-100, and BTC is below EMA200. Hold for 12h."
            ),
        ),
        StrategyDefinition(
            strategy_id="R4_highvol_selloff_rebound_volume",
            name="R4 High-Vol Selloff Rebound With Volume",
            category="rediscovered_clean_candidate",
            side="long",
            lookback_hours=4.0,
            horizon_minutes=240,
            momentum_lower=None,
            momentum_upper=-6.0,
            vol_lower=70.0,
            vol_upper=100.0,
            filter_name="volume_z_gt_0",
            description=(
                "Clean rediscovered candidate. LONG when 4h return < -6%, "
                "volatility percentile is 70-100, and volume z-score > 0. Hold for 4h."
            ),
        ),

        # ─────────────────────────────────────────────────────────────────────
        # Ablation baselines
        # ─────────────────────────────────────────────────────────────────────
        StrategyDefinition(
            strategy_id="E_momentum_only_baseline_grid",
            name="E Momentum-Only Baseline",
            category="ablation_baseline",
            side="long",
            lookback_hours=24.0,
            horizon_minutes=480,
            momentum_lower=None,
            momentum_upper=-6.0,
            vol_lower=None,
            vol_upper=None,
            filter_name="none",
            description=(
                "Momentum-only baseline. LONG when 24h return < -6%, "
                "no volatility or trend filter. Hold for 8h."
            ),
        ),
        StrategyDefinition(
            strategy_id="R1_baseline_selloff_no_vol_filter",
            name="R1 Baseline Selloff Without Vol Filter",
            category="ablation_baseline",
            side="long",
            lookback_hours=24.0,
            horizon_minutes=720,
            momentum_lower=None,
            momentum_upper=-6.0,
            vol_lower=None,
            vol_upper=None,
            filter_name="btc_below_ema200",
            description=(
                "Ablation baseline for R1. LONG when 24h return < -6% and BTC is below EMA200, "
                "but without the volatility filter. Hold for 12h."
            ),
        ),
        StrategyDefinition(
            strategy_id="R3_upside_continuation_no_vol_btc_below",
            name="R3 Upside Continuation Without Vol Filter",
            category="ablation_baseline",
            side="long",
            lookback_hours=6.0,
            horizon_minutes=720,
            momentum_lower=6.0,
            momentum_upper=None,
            vol_lower=None,
            vol_upper=None,
            filter_name="btc_below_ema200",
            description=(
                "Ablation baseline for R2. LONG when 6h return > +6% and BTC is below EMA200, "
                "but without the volatility filter. Hold for 12h."
            ),
        ),

        # ─────────────────────────────────────────────────────────────────────
        # Passive benchmark
        # ─────────────────────────────────────────────────────────────────────
        StrategyDefinition(
            strategy_id="F_buy_and_hold_equal_weight",
            name="Buy & Hold Equal Weight",
            category="benchmark",
            side="long",
            lookback_hours=None,
            horizon_minutes=None,
            momentum_lower=None,
            momentum_upper=None,
            vol_lower=None,
            vol_upper=None,
            filter_name="none",
            description=(
                "Equal-weight buy-and-hold benchmark across BTC, ETH, SOL, DOGE, and XRP "
                "over the final holdout year."
            ),
        ),
    ]