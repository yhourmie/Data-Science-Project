"""Compare candidate models with CV, calibrate the winner, pick a profit-optimal threshold.

Every candidate is logged to MLflow as a nested run; the final calibrated model is
registered as `bank-marketing` and also saved to models/bank/model.joblib for serving.
"""

from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from dsp.bank.data import FEATURES, PROCESSED, TARGET
from dsp.bank.pipeline import best_threshold, build, expected_profit, lift_at
from dsp.common.tracking import (
    MODELS_DIR,
    REPORTS_DIR,
    load_params,
    seed_everything,
    setup_mlflow,
    write_json,
)

OUT = REPORTS_DIR / "bank"


def figures(y, proba, threshold, grid, profits, importances) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {}

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(y, proba)
    ax[0].plot(fpr, tpr)
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set(
        title=f"ROC (AUC {roc_auc_score(y, proba):.3f})",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    prec, rec, _ = precision_recall_curve(y, proba)
    ax[1].plot(rec, prec)
    ax[1].axhline(np.mean(y), ls="--", color="grey", label="base rate")
    ax[1].set(
        title=f"Precision-recall (AP {average_precision_score(y, proba):.3f})",
        xlabel="Recall",
        ylabel="Precision",
    )
    ax[1].legend()
    fig.tight_layout()
    paths["roc_pr"] = str(OUT / "roc_pr.png")
    fig.savefig(paths["roc_pr"], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    ax[0].plot(mean_pred, frac_pos, "o-")
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set(
        title="Calibration (test set)", xlabel="Predicted probability", ylabel="Observed rate"
    )
    ax[1].plot(grid, profits)
    ax[1].axvline(threshold, ls="--", color="grey")
    ax[1].set(
        title=f"Expected profit vs threshold (best {threshold:.2f})",
        xlabel="Threshold",
        ylabel="Profit",
    )
    fig.tight_layout()
    paths["calibration_profit"] = str(OUT / "calibration_profit.png")
    fig.savefig(paths["calibration_profit"], dpi=120)
    plt.close(fig)

    top = importances.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top.index, top.values)
    ax.set(title="Permutation importance (drop in PR-AUC)", xlabel="Importance")
    fig.tight_layout()
    paths["importance"] = str(OUT / "importance.png")
    fig.savefig(paths["importance"], dpi=120)
    plt.close(fig)
    return paths


def main() -> None:
    p = load_params("bank")
    seed_everything(p["seed"])
    train = pd.read_csv(PROCESSED / "train.csv")
    test = pd.read_csv(PROCESSED / "test.csv")
    X, y = train[FEATURES], train[TARGET].to_numpy()
    Xt, yt = test[FEATURES], test[TARGET].to_numpy()
    pos_weight = float((y == 0).sum() / (y == 1).sum())

    mlflow = setup_mlflow("bank-marketing")
    cv = StratifiedKFold(p["cv_folds"], shuffle=True, random_state=p["seed"])
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}

    with mlflow.start_run(run_name="model-selection"):
        mlflow.log_params(
            {
                "cost_per_call": p["cost_per_call"],
                "revenue_per_subscription": p["revenue_per_subscription"],
                "dropped": ",".join(p["drop_columns"]),
            }
        )
        cv_results = {}
        for name, hp in p["models"].items():
            with mlflow.start_run(run_name=name, nested=True):
                scores = cross_validate(
                    build(name, hp, p["seed"], pos_weight), X, y, cv=cv, scoring=scoring, n_jobs=1
                )
                res = {k: float(np.mean(scores[f"test_{k}"])) for k in scoring}
                res.update({f"{k}_std": float(np.std(scores[f"test_{k}"])) for k in scoring})
                mlflow.log_params({"model": name, **hp})
                mlflow.log_metrics({f"cv_{k}": v for k, v in res.items()})
                cv_results[name] = res
                print(
                    f"{name:14s} PR-AUC {res['pr_auc']:.4f} ± {res['pr_auc_std']:.4f}"
                    f"  ROC-AUC {res['roc_auc']:.4f}"
                )

        winner = max(cv_results, key=lambda n: cv_results[n]["pr_auc"])
        mlflow.set_tag("winner", winner)

        # Isotonic calibration so predicted probabilities can be used for the cost decision.
        model = CalibratedClassifierCV(
            build(winner, p["models"][winner], p["seed"], pos_weight), method="isotonic", cv=cv
        )
        model.fit(X, y)

        # Threshold chosen on out-of-fold train predictions, never on the test set.
        from sklearn.model_selection import cross_val_predict

        oof = cross_val_predict(
            CalibratedClassifierCV(
                build(winner, p["models"][winner], p["seed"], pos_weight), method="isotonic", cv=3
            ),
            X,
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        thr, _, _, _ = best_threshold(y, oof, p["cost_per_call"], p["revenue_per_subscription"])

        proba = model.predict_proba(Xt)[:, 1]
        _, _, grid, profits = best_threshold(
            yt, proba, p["cost_per_call"], p["revenue_per_subscription"]
        )
        call_everyone = expected_profit(
            yt, proba, 0.0, p["cost_per_call"], p["revenue_per_subscription"]
        )
        test_metrics = {
            "test_roc_auc": roc_auc_score(yt, proba),
            "test_pr_auc": average_precision_score(yt, proba),
            "test_brier": brier_score_loss(yt, proba),
            "test_f1_at_threshold": f1_score(yt, proba >= thr),
            "test_lift_top10": lift_at(yt, proba, 0.10),
            "threshold": thr,
            "test_profit_model": expected_profit(
                yt, proba, thr, p["cost_per_call"], p["revenue_per_subscription"]
            ),
            "test_profit_call_everyone": call_everyone,
            "test_share_called": float((proba >= thr).mean()),
        }
        test_metrics = {k: float(v) for k, v in test_metrics.items()}
        mlflow.log_metrics(test_metrics)

        imp = permutation_importance(
            model,
            Xt,
            yt,
            scoring="average_precision",
            n_repeats=5,
            random_state=p["seed"],
            n_jobs=1,
        )
        importances = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)

        for path in figures(yt, proba, thr, grid, profits, importances).values():
            mlflow.log_artifact(path)

        MODELS_DIR.joinpath("bank").mkdir(parents=True, exist_ok=True)
        bundle = {"model": model, "threshold": thr, "features": FEATURES, "winner": winner}
        joblib.dump(bundle, MODELS_DIR / "bank" / "model.joblib")
        # MLflow saves sklearn models with skops, which refuses unknown types by default.
        # This model was trained in this very run, so its own types are trusted explicitly.
        import skops.io as sio

        trusted = sio.get_untrusted_types(data=sio.dumps(model))
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="bank-marketing",
            input_example=Xt.head(3),
            skops_trusted_types=trusted,
        )

        write_json(
            {
                "winner": winner,
                "cv": cv_results,
                "test": test_metrics,
                "importance": importances.round(5).to_dict(),
            },
            OUT / "metrics.json",
        )
        print(f"winner={winner}", {k: round(v, 4) for k, v in test_metrics.items()})


if __name__ == "__main__":
    main()
