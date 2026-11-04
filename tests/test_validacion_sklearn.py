import inspect

import numpy as np
import pytest
from sklearn.preprocessing import KBinsDiscretizer

from src.cuantizacion import ajustar_cuantizador, transformar

PARAMS = inspect.signature(KBinsDiscretizer.__init__).parameters


def _sk(x, strategy, k):
    kw = dict(n_bins=k, encode="ordinal", strategy=strategy)
    if "subsample" in PARAMS:
        kw["subsample"] = None
    if strategy == "quantile" and "quantile_method" in PARAMS:
        kw["quantile_method"] = "linear"
    est = KBinsDiscretizer(**kw).fit(x.reshape(-1, 1))
    cod = est.transform(x.reshape(-1, 1)).ravel().astype(np.int64)
    return est.bin_edges_[0], cod


@pytest.mark.parametrize("k", [3, 10, 25, 50])
@pytest.mark.parametrize("estrategia, strategy", [("uniforme", "uniform"), ("cuantiles", "quantile")])
def test_igual_a_kbins_en_datos_continuos(estrategia, strategy, k):
    x = np.random.default_rng(0).normal(size=20_000)
    lim_sk, cod_sk = _sk(x, strategy, k)
    m = ajustar_cuantizador(x, estrategia, k)
    assert np.allclose(lim_sk, m.limites)
    assert np.array_equal(cod_sk, transformar(x, m).astype(np.int64))