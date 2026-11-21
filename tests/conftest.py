import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from powerplant import config
from powerplant.artifact import Bundle, ModelPack
from powerplant.data import drop_exact_duplicates, load_raw
from powerplant.intervals import conformal_quantile
from powerplant.novelty import fit_novelty


def _linear_pack(df: pd.DataFrame, features: tuple[str, ...]) -> ModelPack:
    x, y = df[list(features)], df[config.TARGET]
    model = LinearRegression().fit(x, y)
    residuals = y.to_numpy() - model.predict(x)
    return ModelPack(
        name="Линейная",
        features=features,
        estimator=model,
        q=conformal_quantile(residuals),
        residuals=residuals,
        novelty=fit_novelty(x),
        background=x.sample(50, random_state=0).reset_index(drop=True),
        physics=False,
    )


@pytest.fixture(scope="session")
def small_bundle() -> Bundle:
    """Быстрый бандл на линейных моделях: тестам API не нужно долгое обучение."""
    df = drop_exact_duplicates(load_raw())
    example = df.sample(72, random_state=0).reset_index(drop=True)
    degraded = example.index >= 36
    example = example.assign(PE=example["PE"].where(~degraded, example["PE"] * 0.98))
    typical = {col: round(float(df[col].median()), 1) for col in config.FEATURES_ALL}
    return Bundle(
        version="test",
        created_at="test",
        dataset_sha256="test",
        level=config.INTERVAL_LEVEL,
        plan=_linear_pack(df, config.FEATURES_PLAN),
        reference=_linear_pack(df, config.FEATURES_ALL),
        detector_window=24,
        detector_threshold=-2.5,
        presets={"Типичный час": typical},
        importance={"plan": [], "ref": []},
        metrics={},
        mean_power=float(df[config.TARGET].mean()),
        price_default=config.PRICE_RUB_PER_MWH,
        example_batch=example,
    )
