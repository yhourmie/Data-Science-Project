"""Compare candidate models with CV, tune the decision threshold, evaluate on the test set.

Every candidate is logged to MLflow as a nested run; the winner is registered as
`ev-phev` and also saved to models/ev/model.joblib for serving.
"""

from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from dsp.common.tracking import (
    MODELS_DIR,
    REPORTS_DIR,
    load_params,
    seed_everything,
    setup_mlflow,
    write_json,
)
from dsp.ev.data import FEATURES, PROCESSED, TARGET
from dsp.ev.pipeline import best_f1_threshold, build

OUT = REPORTS_DIR / "ev"


def figures(test: pd.DataFrame, proba, importances: pd.Series) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    y = test[TARGET].to_numpy()
    paths = {}

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
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
    ax[1].axhline(y.mean(), ls="--", color="grey", label="base rate")
    ax[1].set(
        title=f"Precision-recall (AP {average_precision_score(y, proba):.3f})",
        xlabel="Recall",
        ylabel="Precision",
    )
    ax[1].legend()
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    ax[2].plot(mean_pred, frac_pos, "o-")
    ax[2].plot([0, 1], [0, 1], "--", color="grey")
    ax[2].set(title="Calibration (test set)", xlabel="Predicted P(PHEV)", ylabel="Observed share")
    fig.tight_layout()
    paths["roc_pr_calibration"] = str(OUT / "roc_pr_calibration.png")
    fig.savefig(paths["roc_pr_calibration"], dpi=120)
    plt.close(fig)

    # Observed vs predicted PHEV share per county: does the model get the geography right?
    by_county = (
        test.assign(pred=proba)
        .groupby("county")
        .agg(observed=(TARGET, "mean"), predicted=("pred", "mean"), n=(TARGET, "size"))
        .query("n >= 30")
    )
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(by_county["observed"], by_county["predicted"], s=np.sqrt(by_county["n"]) * 2)
    lim = [0, max(by_county[["observed", "predicted"]].max()) * 1.1]
    ax.plot(lim, lim, "--", color="grey")
    ax.set(
        title="PHEV share by county (test set, n >= 30)",
        xlabel="Observed",
        ylabel="Predicted (mean probability)",
        xlim=lim,
        ylim=lim,
    )
    fig.tight_layout()
    paths["county_share"] = str(OUT / "county_share.png")
    fig.savefig(paths["county_share"], dpi=120)
    plt.close(fig)

    top = importances[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top.index, top.values)
    ax.set(title="Permutation importance (drop in PR-AUC)", xlabel="Importance")
    fig.tight_layout()
    paths["importance"] = str(OUT / "importance.png")
    fig.savefig(paths["importance"], dpi=120)
    plt.close(fig)
    return paths


def main() -> None:
    p = load_params("ev")
    seed_everything(p["seed"])
    train = pd.read_csv(
        PROCESSED / "train.csv", dtype={"postal_code": str, "legislative_district": str}
    )
    test = pd.read_csv(
        PROCESSED / "test.csv", dtype={"postal_code": str, "legislative_district": str}
    )
    X, y = train[FEATURES], train[TARGET].to_numpy()
    Xt, yt = test[FEATURES], test[TARGET].to_numpy()

    def make(name):
        return build(name, p["models"][name], p["seed"], p["min_category_count"])

    mlflow = setup_mlflow("ev-phev")
    cv = StratifiedKFold(p["cv_folds"], shuffle=True, random_state=p["seed"])
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "log_loss": "neg_log_loss"}

    with mlflow.start_run(run_name="model-selection"):
        mlflow.log_params({"rows_train": len(train), "phev_rate": round(float(y.mean()), 4)})
        cv_results = {}
        for name, hp in p["models"].items():
            with mlflow.start_run(run_name=name, nested=True):
                scores = cross_validate(make(name), X, y, cv=cv, scoring=scoring, n_jobs=1)
                res = {k: float(np.mean(scores[f"test_{k}"])) for k in scoring}
                res["log_loss"] = -res["log_loss"]
                res["pr_auc_std"] = float(np.std(scores["test_pr_auc"]))
                mlflow.log_params({"model": name, **hp})
                mlflow.log_metrics({f"cv_{k}": v for k, v in res.items()})
                cv_results[name] = res
                print(
                    f"{name:10s} PR-AUC {res['pr_auc']:.4f} ± {res['pr_auc_std']:.4f}"
                    f"  ROC-AUC {res['roc_auc']:.4f}  log-loss {res['log_loss']:.4f}"
                )

        winner = max(cv_results, key=lambda n: cv_results[n]["pr_auc"])
        mlflow.set_tag("winner", winner)

        # Threshold chosen on out-of-fold train predictions, never on the test set.
        oof = cross_val_predict(make(winner), X, y, cv=cv, method="predict_proba")[:, 1]
        thr, _ = best_f1_threshold(y, oof)

        model = make(winner).fit(X, y)
        proba = model.predict_proba(Xt)[:, 1]
        test_metrics = {
            "test_roc_auc": roc_auc_score(yt, proba),
            "test_pr_auc": average_precision_score(yt, proba),
            "test_brier": brier_score_loss(yt, proba),
            "test_brier_base_rate": brier_score_loss(yt, np.full_like(proba, y.mean())),
            "test_f1_at_threshold": f1_score(yt, proba >= thr),
            "threshold": thr,
        }
        test_metrics = {k: float(v) for k, v in test_metrics.items()}
        mlflow.log_metrics(test_metrics)

        imp = permutation_importance(
            model,
            Xt,
            yt,
            scoring="average_precision",
            n_repeats=3,
            max_samples=20_000,
            random_state=p["seed"],
            n_jobs=1,
        )
        importances = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)

        for path in figures(test, proba, importances).values():
            mlflow.log_artifact(path)

        MODELS_DIR.joinpath("ev").mkdir(parents=True, exist_ok=True)
        bundle = {"model": model, "threshold": thr, "features": FEATURES, "winner": winner}
        joblib.dump(bundle, MODELS_DIR / "ev" / "model.joblib")
        # MLflow saves sklearn models with skops, which refuses unknown types by default.
        # This model was trained in this very run, so its own types are trusted explicitly.
        import skops.io as sio

        trusted = sio.get_untrusted_types(data=sio.dumps(model))
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="ev-phev",
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
