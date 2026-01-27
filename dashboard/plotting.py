from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def plot_series(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    fig = plt.figure()
    plt.plot(df[x], df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    st.pyplot(fig)
    plt.close(fig)


def plot_overlay(series: List[Tuple[str, pd.DataFrame]], x: str, y: str, title: str) -> None:
    fig = plt.figure()
    for label, df in series:
        if x in df.columns and y in df.columns:
            plt.plot(df[x], df[y], label=label)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    st.pyplot(fig)
    plt.close(fig)
