import math


class Convergence:
    """Require full data exposure before LR plateau reductions; two passes before stopping."""

    def __init__(
        self,
        min_delta=0.001,
        reduce_patience=3,
        stop_patience=6,
        min_factor=0.0625,
        min_coverage=2,
        regression_tolerance=0.05,
        failure_patience=3,
    ):
        self.min_delta = min_delta
        self.reduce_patience = reduce_patience
        self.stop_patience = stop_patience
        self.min_factor = min_factor
        self.min_coverage = min_coverage
        self.regression_tolerance = regression_tolerance
        self.failure_patience = failure_patience
        self.best = -float("inf")
        self.factor = 1.0
        self.bad = 0
        self.floor_bad = 0
        self.regression_bad = 0
        self.unhealthy = 0

    def update(self, score, coverage, healthy):
        if not math.isfinite(score):
            raise ValueError("Nonfinite validation score")
        self.unhealthy = 0 if healthy else self.unhealthy + 1
        self.regression_bad = self.regression_bad + 1 if score < self.best - self.regression_tolerance else 0
        improved = healthy and score > self.best + self.min_delta
        if improved:
            self.best = score
            self.bad = 0
            self.floor_bad = 0
        elif coverage >= 1:
            self.bad += 1
            if self.factor <= self.min_factor:
                self.floor_bad += 1
            elif self.bad >= self.reduce_patience:
                self.factor = max(self.min_factor, self.factor * 0.5)
                self.bad = 0
                self.floor_bad = 0
        reason = (
            "representation_unhealthy"
            if self.unhealthy >= self.failure_patience
            else (
                "validation_regression"
                if self.regression_bad >= self.failure_patience
                else (
                    "validation_plateau"
                    if healthy
                    and coverage >= self.min_coverage
                    and self.factor <= self.min_factor
                    and self.floor_bad >= self.stop_patience
                    else None
                )
            )
        )
        return {"improved": improved, "stop_reason": reason, "lr_factor": self.factor, "best": self.best}
