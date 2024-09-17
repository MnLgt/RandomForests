from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from dtreeviz.trees import *
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import talib.abstract as ta
import warnings

pd.options.display.max_rows = 20
pd.options.display.max_columns = 8


warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def get_indicators(df):
    df_ind = df.copy()
    # to add all technical indicators from the talib library to the dataframe
    indicators = []
    for func in ta.__TA_FUNCTION_NAMES__:
        try:
            ind = getattr(ta, func)(df_ind)
            if isinstance(ind, pd.DataFrame):
                for c in ind.columns:
                    df_ind[c] = ind[c]
                    indicators.append(c)
            else:
                df_ind[func] = ind
                indicators.append(func)
        except Exception as e:
            print(f"{func} not added.")
            print(f"Error: {e}")
    return df_ind


class RandomForest:
    """
    This class is used to create a random forest model

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that contains the data
    x_cols : list
        The list of columns to use as features
    y_col : str
        The name of the target column

    Returns
    ----------
    rf : RandomForestRegressor
        The fitted random forest model
    y_pred : numpy.ndarray
        The array of predictions
    fi : pandas.DataFrame
        The dataframe of feature importances
    imp_cols : list
        The list of important columns

    Examples
    --------
    >>> from monty_python.utils import RandomForest
    >>> rf = RandomForest(df, x_cols, y_col)
    >>> rf.rf
    RandomForestRegressor(random_state=42)
    >>> rf.y_pred
    >>> rf.corr_matrix(target ='next_week_mean_price',thresh=0.06)
    >>> rf.plot_feature_importance(thresh=0.01)
    """

    def __init__(self, df, x_cols, y_col):
        self.df = df
        self.x_cols = x_cols
        self.y_col = y_col
        self.X = self.df[self.x_cols]
        self.y = self.df[self.y_col]
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(
            self.X, self.y
        )
        self.rf = self.fit()
        self.y_pred = self.predict()
        self.fi = self.feature_importance()
        self.imp_cols = self.fi["cols"].tolist()
        self.cont_cols = None

    def split_data(self, X, y):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def fit(self):
        # Initialize the model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)

        # Fit the model
        rf.fit(self.X_train, self.y_train)
        return rf

    def predict(self):
        # Predict
        y_pred = self.rf.predict(self.X_test)
        return y_pred

    def feature_importance(self, thresh=0.0001):
        fi = pd.DataFrame(
            {"cols": self.X.columns, "imp": self.rf.feature_importances_}
        ).sort_values("imp", ascending=False)
        return fi.query("imp > @thresh").copy()

    def get_feature_cols(self, thresh=0.001, lookback=7):
        # get the feature columns that have a similarity greater than the threshold
        feature_cols = self.fi.query("imp > @thresh").cols.tolist()

        # get the lookback features
        df, self.cont_cols = lookback_data(
            self.df, feature_cols, self.y_col, lookback)

        # remove null values
        df.dropna(axis=0, inplace=True)

        # reset the index
        df.reset_index(drop=True)
        return df

    def plot_feature_importance(self, thresh=0.01):
        # Define a color map with muted colors
        color_discrete_map = {
            "imp": ["#4C4C6D", "#5E5E8D", "#6F6FA1", "#8080B4", "#9191C8"]
        }
        title = f"Feature Importance for {self.y_col}"
        # Create a bar chart with the muted color map
        fi = self.fi.query("imp > @thresh")
        fig = px.bar(fi, x="imp", y="cols", color="cols",
                     orientation="h", title=title)

        # Show the chart
        fig.show()

    def corr_m(self, target=None):
        corr = self.df.select_dtypes(("float", "object", "int")).corr()
        if not target:
            return corr
        else:
            return corr[[target]].sort_values(target, ascending=False)

    def corr_matrix(self, target="close", thresh=0.001):
        plt.figure(figsize=(15, 15))
        sns.set_theme(style="white")
        corr = self.corr_m().copy()
        corr = corr.sort_values(target, ascending=False).query(
            f"abs({target}) > @thresh"
        )
        heatmap = sns.heatmap(
            corr[[target]],
            annot=True,
            cmap="coolwarm",
            center=0,
            linewidths=3,
            linecolor="white",
        )
        heatmap.set_title(
            f"Correlation Heatmap for {target}", fontdict={"fontsize": 18}, pad=12
        )


def lookback_data(df, feature_cols, target, lookback=7):
    """
    This function is used to add lookback columns to a dataframe
    Lookback columns are previous values of the features and target
    The purpose of this is to use the previous values as features to predict the target

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that contains the data
    feature_cols : list
        The list of columns to use as features
    target : str
        The name of the target column
    lookback : int
        The number of lookback periods to use
    """
    cont_names = []
    df = df.copy()
    for i in range(lookback):
        for col in feature_cols:
            cont_name = f"prev_{col}_{i}"
            df[cont_name] = df[col].shift(i)
            cont_names.append(cont_name)
    return df, cont_names


def get_rmse(df, pred_col, actual_col):
    return np.sqrt(((df[pred_col] - df[actual_col]) ** 2).mean())


def render_all(self, title=None):
    """
    This function is used to render the stocks env results
    for gym anytrading

    Examples
    --------
    render_all(env.unwrapped)
    """
    window_ticks = np.arange(len(self._position_history))

    # Prices trace
    prices_trace = go.Scatter(
        x=window_ticks,
        y=self.prices,
        mode='lines',
        name='Prices'
    )

    short_ticks = []
    long_ticks = []
    for i, tick in enumerate(window_ticks):
        if self._position_history[i] == Positions.Short:
            short_ticks.append(tick)
        elif self._position_history[i] == Positions.Long:
            long_ticks.append(tick)

    # Short positions trace
    short_trace = go.Scatter(
        x=short_ticks,
        y=np.array(self.prices)[short_ticks],
        mode='markers',
        marker=dict(color='red'),
        name='Short'
    )

    # Long positions trace
    long_trace = go.Scatter(
        x=long_ticks,
        y=np.array(self.prices)[long_ticks],
        mode='markers',
        marker=dict(color='green'),
        name='Long'
    )

    layout = go.Layout(
        title=f"{title} - Total Reward: {self._total_reward:.6f} ~ Total Profit: {self._total_profit:.6f}",
        xaxis_title="Ticks",
        yaxis_title="Price"
    )

    fig = go.Figure(data=[prices_trace, short_trace,
                    long_trace], layout=layout)
    fig.show()
