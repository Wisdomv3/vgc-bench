"""Decision-grade labels for Champions AI.

The labels are intentionally separated from numeric thresholds. Thresholds will
be calibrated later from simulation and benchmark data instead of being chosen
arbitrarily during the heuristic phase.
"""

from enum import Enum


class DecisionGrade(str, Enum):
    STUPENDOUS = "Stupendous"
    AMAZING = "Amazing"
    OUTSTANDING = "Outstanding"
    AWESOME = "Awesome"
    GREAT = "Great"
    GOOD = "Good"
    OK = "Ok"
    MISTAKE = "Mistake"
    MISS = "Miss"
    THROWING = "Throwing"


GRADE_ORDER: tuple[DecisionGrade, ...] = (
    DecisionGrade.STUPENDOUS,
    DecisionGrade.AMAZING,
    DecisionGrade.OUTSTANDING,
    DecisionGrade.AWESOME,
    DecisionGrade.GREAT,
    DecisionGrade.GOOD,
    DecisionGrade.OK,
    DecisionGrade.MISTAKE,
    DecisionGrade.MISS,
    DecisionGrade.THROWING,
)
