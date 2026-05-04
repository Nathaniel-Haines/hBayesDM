from typing import Dict, List, Sequence, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hbayesdm.base import TaskModel

__all__ = ['rhat', 'print_fit', 'hdi', 'plot_hdi', 'extract_ic']


def rhat(model_data: TaskModel,
         less: float = None) -> Dict[str, Union[List, bool]]:
    """Extract Rhat values from hbayesdm output.

    If `less` is given, returns whether each parameter's max Rhat is <= less.
    """
    rhat_data = az.rhat(model_data.idata)
    if less is None:
        return {v.name: v.values.tolist()
                for v in rhat_data.data_vars.values()}
    return {v.name: v.values.item()
            for v in (rhat_data.max() <= less).data_vars.values()}


def print_fit(*args: TaskModel) -> pd.DataFrame:
    """Compare hbayesdm models by ELPD-LOO."""
    return az.compare({m.model: m.idata for m in args})


def hdi(x: np.ndarray, prob: float = 0.94) -> np.ndarray:
    """Compute the highest density interval. Alias for `arviz.hdi`."""
    return az.hdi(x, prob=prob)


def plot_hdi(x: np.ndarray,
             prob: float = 0.94,
             title: str = None,
             xlabel: str = 'Value',
             ylabel: str = 'Density',
             point_estimate: str = None,
             **kwargs):
    """Plot a posterior distribution with HDI shading."""
    idata = az.from_dict({'posterior': {'x': np.asarray(x)[None, ...]}})
    az.plot_dist(idata,
                 var_names=['x'],
                 ci_prob=prob,
                 point_estimate=point_estimate,
                 **kwargs)
    ax = plt.gca()
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def extract_ic(model_data: TaskModel) -> Dict:
    """Extract LOO information criterion estimates.

    Returns ``{'looic': float, 'loo': arviz.ELPDData}``. Call ``arviz.loo``
    directly on ``model_data.idata`` for pointwise values, Pareto k, etc.
    """
    loo_result = az.loo(model_data.idata)
    return {'looic': float(loo_result.elpd) * -2,
            'loo': loo_result}
