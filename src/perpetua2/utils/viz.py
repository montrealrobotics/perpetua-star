from typing import List, Tuple, Dict

from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines  # For line-based legend handles
import matplotlib.dates as mdates
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns

from datetime import timedelta
import numpy as np
import os
import datetime

import scienceplots  # noqa: F401

from perpetua2.utils.logger import logger

plt.style.use(["science", "no-latex", "sans"])

# This is the default number of colors we'll use
N_COLORS = 10

color_palette = [(74, 35, 119), (140, 197, 227), (245, 95, 116), (89, 168, 156), (240, 197, 113)]
color_palette = [(r / 255, g / 255, b / 255) for r, g, b in color_palette]
color_palette_str = [
    "purple",
    "light_blue",
    "pink",
    "teal",
    "mustard",
]
color_palette_from_str = {k: v for k, v in zip(color_palette_str, color_palette)}
color_palette_from_method = {
    "persistence": color_palette_from_str["mustard"],
    "emergence": color_palette_from_str["teal"],
    "perpetua": color_palette_from_str["mustard"],
    "perpetua-star": color_palette_from_str["purple"],
    "observation": color_palette_from_str["pink"],
    "ground_truth": color_palette_from_str["light_blue"],
    "component": color_palette_from_str["mustard"],
    "loss": color_palette_from_str["pink"],
    "fremen": color_palette_from_str["light_blue"],
}

grey_palette = sns.color_palette("Greys", n_colors=N_COLORS + 2)

# This is a grey palette: 0 means white, 100 means black
# The line below helps indexing "10" as '10% black', while "80" means '80% black
grey_palette_from_str = {f"{p}0": v for p, v in enumerate(grey_palette)}

# The stuff below is just to facilitate semantics
grey_palette_from_str["black"] = grey_palette_from_str["100"]
grey_palette_from_str["white"] = grey_palette_from_str["00"]

# These are just some helpers to index the colors 'semantically' when making plots


def minutes_to_datetime(minutes: np.ndarray, start_day: str = "monday") -> np.ndarray:
    """
    Convert minutes (from Monday 00:00) into datetime objects.
    """
    days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    start_idx = days_of_week.index(start_day.lower())
    start_date = datetime.datetime(2023, 1, 2 + start_idx)  # pick arbitrary Monday
    return np.array([start_date + timedelta(minutes=int(m)) for m in minutes])


def plots_config():
    # To set the font size
    plt.rcParams["font.size"] = 8


def save_fig(
    fig: matplotlib.figure.Figure,
    fig_name: str,
    fig_dir: str,
    fig_fmt: str,
    fig_size: Tuple[float, float] = [6.4, 4],
    save: bool = True,
    dpi: int = 300,
    transparent_png=True,
):
    """
    Code adapted from https://zhauniarovich.com/post/2022/2022-09-matplotlib-graphs-in-research-papers/
    This procedure stores the generated matplotlib figure to the specified
    directory with the specified name and format.

    Parameters
    ----------
    fig : [type]
        Matplotlib figure instance
    fig_name : str
        File name where the figure is saved
    fig_dir : str
        Path to the directory where the figure is saved
    fig_fmt : str
        Format of the figure, the format should be supported by matplotlib
        (additional logic only for pdf and png formats)
    fig_size : Tuple[float, float]
        Size of the figure in inches, by default [6.4, 4]
    save : bool, optional
        If the figure should be saved, by default True. Set it to False if you
        do not want to override already produced figures.
    dpi : int, optional
        Dots per inch - the density for rasterized format (png), by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if not save:
        return

    fig.set_size_inches(fig_size, forward=False)
    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pth = os.path.join(fig_dir, "{}.{}".format(fig_name, fig_fmt.lower()))
    if fig_fmt == "pdf":
        metadata = {"Creator": "", "Producer": "", "CreationDate": None}
        fig.savefig(pth, bbox_inches="tight", metadata=metadata)
    elif fig_fmt == "png":
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(
            pth,
            bbox_inches="tight",
            dpi=dpi,
        )
    else:
        try:
            fig.savefig(pth, bbox_inches="tight")
        except Exception as e:
            print("Cannot save figure: {}".format(e))


class Plotter:
    def __init__(self, exp_name: str, viz: bool = True, res_path: str = None, fig_fmt="pdf") -> None:
        self.viz = viz

        # Create a directory to store the results if viz is False
        script_directory = os.path.abspath(os.curdir)
        self.res_path = res_path
        if self.res_path is None:
            self.res_path = os.path.join(script_directory, "assets", datetime.datetime.now().isoformat(), exp_name)
        logger.info(f"Saving results to {self.res_path}")
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)
        # Savefig partial
        self.savefig = partial(save_fig, fig_dir=self.res_path, fig_fmt=fig_fmt, transparent_png=False)
        # Conversion utils
        self.mm = 1 / 25.4  # conversion from millimeters to inches
        self.cm = 1 / 2.54  # conversion from centimeters to inches
        self.pt = 1 / 72.27  # conversion from points to inches

        # The values below are obtained from LaTeX, using \the\columnwidth and \the\textwidth
        self.PAPER_COLUMN = 252.0  # points
        self.PAPER_TEXTWIDTH = 516.0  # points
        plots_config()

    def plot_observations(
        self,
        sampling_times: List[float],
        true_state: List[float],
        t: List[float],
        obs: List[int],
        location: str = "lower left",
        figure_size: Tuple[int, int] = (6.4, 4.0),
        f_name: str = "observations",
    ) -> None:
        """
        Plot binary observations against the true state over time.

        Args:
            sampling_times (List[float]): Time points for the true state samples.
            true_state (List[float]): True state values at sampling times.
            t (List[float]): Time points for the observations.
            obs (List[int]): Binary observations at times `t`.
            location (str, optional): Legend location in the plot. Default is "lower left".
            figure_size (Tuple[int, int]): Figure size in inches. Defaults to `self.figure_size`.
            f_name (str, optional): File name for saving the plot. Default is "observations".

        Returns:
            None
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figure_size, dpi=300, constrained_layout=True)

        # Plot true state
        ax.plot(
            sampling_times,
            true_state,
            color=color_palette_from_method["ground_truth"],
            ls="--",
            linewidth=1,
            label="True state",
        )

        # Plot observations
        ax.plot(
            t,
            obs,
            color=color_palette_from_method["observation"],
            linestyle="None",
            marker=".",
            markersize=6,
            label="Observation",
        )

        # Labels and legend
        ax.set_xlabel("Time (t)")
        ax.set_title("Sampled Observations")
        ax.legend(loc=location, numpoints=1, frameon=False)

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False, fig_size=figure_size)
            plt.show()
        else:
            self.savefig(fig, fig_name=f_name, save=True, fig_size=figure_size)

        plt.close()

    def plot_loss_curve(
        self,
        iterations: List[int],
        loglikelihoods: List[float],
        title: str = "Loss Curve",
        f_name: str = "loss_curve.png",
    ) -> None:
        """
        Plot the log-likelihood values against the number of iterations.

        Args:
            iterations (List[int]): The number of iterations.
            loglikelihoods (List[float]): The log-likelihood values.
            title (str, optional): The title of the plot. Default is "Loss Curve".
            f_name (str, optional): File name for saving the plot. Default is "loss_curve.png".

        Returns:
            None
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300, constrained_layout=True)

        # Plot log-likelihood values
        ax.plot(iterations, loglikelihoods, lw=2, color=color_palette_from_method["loss"])

        # Labels and title
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Log-Likelihood")
        ax.set_title(title)

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False, fig_size=(6.4, 4.0))
            plt.show()
        else:
            self.savefig(fig, fig_name=f_name, save=True, fig_size=(6.4, 4.0))

        plt.close()

    def plot_mixture_filter(
        self,
        sampling_times: np.ndarray,
        true_state: np.ndarray,
        t: np.ndarray,
        obs: np.ndarray,
        query_times: np.ndarray,
        belief: np.ndarray,
        location: str = "lower left",
        figure_size: Tuple[int, int] = (6.4, 4),
        f_name: str = "mixture_death_filters",
    ) -> None:
        """
        Plot the results of a mixture filter, showing the true state, observations, and filter beliefs over time.

        Args:
            sampling_times (np.ndarray): The times at which the true state was sampled.
            true_state (np.ndarray): The true state values corresponding to the sampling times.
            t (np.ndarray): The times at which observations were made.
            obs (np.ndarray): The binary observations made at times `t`.
            query_times (np.ndarray): The times at which beliefs were queried.
            belief (List[np.ndarray]): The belief values from the mixture filter for each component and the marginal posterior.
            location (str): The location of the legend in the plot.
            figure_size (Tuple[int, int]): The size of the figure in inches. Defaults to (6.4, 4).
            f_name (str): The name of the file to save the plot.

        Returns:
            None
        """
        # Number of belief components
        n_components = len(belief)

        # Create figure and axes
        fig, axes = plt.subplots(
            n_components + 1, 1, figsize=figure_size, sharex=True, dpi=300, constrained_layout=True
        )

        # Plot true state and observations
        axes[0].plot(
            sampling_times,
            true_state,
            color=color_palette_from_method["ground_truth"],
            ls="--",
            linewidth=1,
            label="True state",
        )
        axes[0].plot(
            t,
            obs,
            color=color_palette_from_method["observation"],
            linestyle="None",
            marker=".",
            markersize=6,
            label="Observation",
        )

        axes[0].legend(loc=location, numpoints=1, frameon=False)
        axes[0].set_ylabel("State")

        # Plot conditional posterior beliefs
        for i in range(n_components - 1):
            axes[i + 1].plot(
                query_times,
                belief[i],
                color=color_palette_from_method["component"],
                ls="--",
                lw=2,
                label=rf"$p(X_t = 1 \mid Y_{{1:N}}, C = {i})$",
            )
            axes[i + 1].set_ylabel("Belief")
            axes[i + 1].legend(loc=location, numpoints=1, frameon=False)

        # Plot the marginal posterior belief
        axes[-1].plot(
            query_times,
            belief[-1],
            color=color_palette_from_method["perpetua"],
            lw=2,
            label=r"$p(X_t = 1 \vert C_{K^*}, Y_{1:N})$",
        )

        axes[-1].set_ylabel("Belief")
        axes[-1].set_xlabel("Time (t)")
        axes[-1].legend(loc=location, numpoints=1, frameon=False)

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False, fig_size=figure_size)
            plt.show()
        else:
            self.savefig(fig, fig_name=f_name, save=True, fig_size=figure_size)

        plt.close()

    def plot_perpetua(
        self,
        sampling_times: np.ndarray,
        true_state: np.ndarray,
        t: np.ndarray,
        obs: np.ndarray,
        query_times: np.ndarray,
        belief: np.ndarray,
        weights: np.ndarray,
        location: str = "lower left",
        interval: Tuple[int, int] = (0, np.inf),
        figure_size: Tuple[int, int] = (6.4, 4),
        f_name: str = "perpetua_results",
    ) -> None:
        """
        Create a figure with multiple vertically arranged subplots to visualize the results of the Perpetua model.

        Args:
            sampling_times (np.ndarray): The times at which the true state was sampled.
            true_state (np.ndarray): The true state values corresponding to the sampling times.
            t (np.ndarray): The times at which observations were made.
            obs (np.ndarray): The binary observations made at times `t`.
            query_times (np.ndarray): The times at which beliefs were queried.
            belief (List[np.ndarray]): The belief values from the Perpetua model for each component and the heaviest-weighted component.
            weights (np.ndarray): The weights of each component over time.
            location (str): The location of the legend in the plot.
            interval (Tuple[int, int]): The time interval to plot.
            figure_size (Optional[Tuple[int, int]]): The size of the figure in inches. If None, defaults to the instance's figure size.
            f_name (str): The name of the file to save the plot.

        Returns:
            None
        """
        # Number of belief components + final combined plot
        n_components = len(belief)
        n_axes = n_components + 1

        # Create figure and axes
        fig, axes = plt.subplots(n_axes, 1, figsize=figure_size, sharex=True, dpi=300, constrained_layout=True)

        # Select indices within the given time interval
        indices_gt = np.logical_and(sampling_times > interval[0], sampling_times <= interval[1])
        indices_obs = np.logical_and(t > interval[0], t <= interval[1])
        indices_query = np.logical_and(query_times > interval[0], query_times <= interval[1])

        # Plot true state and observations on the first subplot
        axes[0].plot(
            sampling_times[indices_gt],
            true_state[indices_gt],
            color=color_palette_from_method["ground_truth"],
            ls="--",
            linewidth=1,
            label="True state",
        )
        axes[0].plot(
            t[indices_obs],
            obs[indices_obs],
            color=color_palette_from_method["observation"],
            linestyle="None",
            marker=".",
            markersize=6,
            label="Observation",
        )
        axes[0].legend(loc=location, numpoints=1, frameon=False)
        axes[0].set_ylim([0, 1.05])
        axes[0].set_ylabel("State")

        # Plot conditional posterior beliefs and weights
        for i in range(n_components - 1):
            ax_idx = i + 1
            axes[ax_idx].plot(
                query_times[indices_query],
                belief[i][indices_query],
                color=color_palette_from_str["light_blue"],
                ls="-",
                lw=1,
                label=f"Combination {i}",
            )
            axes[ax_idx].plot(
                query_times[indices_query],
                weights[i][indices_query],
                color=color_palette_from_str["pink"],
                ls="--",
                label=rf"$w_{i}$",
            )
            axes[ax_idx].legend(loc=location, numpoints=1, frameon=False)
            axes[ax_idx].set_ylim([0, 1.05])
            axes[ax_idx].set_ylabel("Belief")

        # Plot the heaviest-weighted component
        axes[-1].plot(
            sampling_times[indices_gt],
            true_state[indices_gt],
            color=color_palette_from_method["ground_truth"],
            ls="--",
            linewidth=1,
            label="True state",
        )
        axes[-1].plot(
            query_times[indices_query],
            belief[-1][indices_query],
            color=color_palette_from_method["perpetua"],
            lw=1,
            label="Heaviest-weighted component",
        )
        axes[-1].legend(loc=location, numpoints=1, frameon=False)
        axes[-1].set_ylim([0, 1.05])
        axes[-1].set_xlabel("Time (t)")
        axes[-1].set_ylabel("Belief")

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False, fig_size=figure_size)
            plt.show()
        else:
            self.savefig(fig, fig_name=f_name, save=True, fig_size=figure_size)

        plt.close()

    def plot_mixtures_evidence(
        self,
        sampling_times: np.ndarray,
        true_state: np.ndarray,
        t: np.ndarray,
        obs: np.ndarray,
        query_times: np.ndarray,
        belief: np.ndarray,
        persistence_belief: np.ndarray,
        emergence_belief: np.ndarray,
        persistence_evidence: np.ndarray,
        emergence_evidence: np.ndarray,
        location: str = "lower left",
        figure_size: Tuple[int, int] = (6.4, 4),
        f_name: str = "mixtures_evidence",
    ) -> None:
        """
        Plot the results of a mixture filter, showing the true state, observations, and filter beliefs over time.

        Args:
            sampling_times (np.ndarray): The times at which the true state was sampled.
            true_state (np.ndarray): The true state values corresponding to the sampling times.
            t (np.ndarray): The times at which observations were made.
            obs (np.ndarray): The binary observations made at times `t`.
            query_times (np.ndarray): The times at which beliefs were queried.
            belief (List[np.ndarray]): The belief values from the mixture filter for each component and the marginal posterior.
            location (str): The location of the legend in the plot.
            figure_size (Tuple[int, int]): The size of the figure in inches. Defaults to (6.4, 4).
            f_name (str): The name of the file to save the plot.

        Returns:
            None
        """
        # Create figure and axes
        fig, axes = plt.subplots(4, 1, figsize=figure_size, sharex=True, dpi=300, constrained_layout=True)

        # Plot true state and observations
        axes[0].plot(
            sampling_times,
            true_state,
            color=color_palette_from_method["ground_truth"],
            ls="--",
            linewidth=1,
            label="True state",
        )
        axes[0].plot(
            t,
            obs,
            color=color_palette_from_method["observation"],
            linestyle="None",
            marker=".",
            markersize=6,
            label="Observation",
        )

        axes[0].legend(loc=location, numpoints=1, frameon=False)
        axes[0].set_ylabel("State")
        axes[0].set_ylim(0, 1.05)

        # Plot persistence and emergence belief
        axes[1].plot(
            query_times,
            persistence_belief,
            color=color_palette_from_method["persistence"],
            lw=1.5,
            label="Persistence Belief",
        )
        axes[1].plot(
            query_times,
            emergence_belief,
            color=color_palette_from_method["emergence"],
            lw=1.5,
            label="Emergence Belief",
        )
        axes[1].set_ylabel("Belief")
        axes[1].legend(loc=location, numpoints=1, frameon=False)
        axes[1].set_ylim(0, 1.05)

        axes[2].plot(
            query_times,
            belief,
            color=color_palette_from_method["perpetua"],
            lw=1.5,
            label="Final Belief",
        )
        axes[2].set_ylabel("Belief")
        axes[2].legend(loc=location, numpoints=1, frameon=False)
        axes[2].set_ylim(0, 1.05)

        # Plot persistence and emergence evidence
        axes[3].plot(
            query_times,
            persistence_evidence,
            color=color_palette_from_method["persistence"],
            lw=1.5,
            label="Persistence Evidence",
        )
        axes[3].plot(
            query_times,
            emergence_evidence,
            color=color_palette_from_method["emergence"],
            lw=1.5,
            label="Emergence Evidence",
        )
        axes[3].set_ylabel("Evidence")

        axes[3].set_xlabel("Time (t)")
        axes[3].legend(loc="upper right", numpoints=1, frameon=False)

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False, fig_size=figure_size)
            plt.show()
        else:
            self.savefig(fig, fig_name=f_name, save=True, fig_size=figure_size)

        plt.close()

    def plot_perpetua_star(
        self,
        sampling_times: np.ndarray,
        true_state: np.ndarray,
        t: np.ndarray,
        obs: np.ndarray,
        query_times: np.ndarray,
        belief: np.ndarray,
        mixture_preds: np.ndarray,
        log_evidence: np.ndarray,
        log_prior: np.ndarray,
        log_model_posterior: np.ndarray,
        location: str = "lower left",
        figure_size: Tuple[int, int] = (8, 4),
        f_name: str = "mixtures_evidence",
    ) -> None:
        """
        Plot the results of a mixture filter, showing the true state, observations, and filter beliefs over time.

        Args:
            sampling_times (np.ndarray): The times at which the true state was sampled.
            true_state (np.ndarray): The true state values corresponding to the sampling times.
            t (np.ndarray): The times at which observations were made.
            obs (np.ndarray): The binary observations made at times `t`.
            query_times (np.ndarray): The times at which beliefs were queried.
            belief (List[np.ndarray]): The belief values from the mixture filter for each component and the marginal posterior.
            location (str): The location of the legend in the plot.
            figure_size (Tuple[int, int]): The size of the figure in inches. Defaults to (6.4, 4).
            f_name (str): The name of the file to save the plot.

        Returns:
            None
        """
        fig = plt.figure(figsize=figure_size, dpi=300, constrained_layout=True)
        gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[2, 2], height_ratios=[1, 1, 1], figure=fig)

        # Left column (smaller plots)
        ax_left_0 = fig.add_subplot(gs[0, 0])
        ax_left_1 = fig.add_subplot(gs[1, 0], sharex=ax_left_0)
        ax_left_2 = fig.add_subplot(gs[2, 0], sharex=ax_left_0)

        # Right column (larger plots, shared x-axis)
        ax_right_0 = fig.add_subplot(gs[0, 1])
        ax_right_1 = fig.add_subplot(gs[1, 1], sharex=ax_right_0)
        ax_right_2 = fig.add_subplot(gs[2, 1], sharex=ax_right_0)

        for ax in [ax_right_0, ax_right_1, ax_right_2]:
            ax.set_ylim(-0.05, 1.05)
        for ax in [ax_right_0, ax_right_1, ax_left_0, ax_left_1]:
            ax.tick_params(labelbottom=False)

        # Plot true state and observations
        ax_right_0.plot(
            sampling_times, true_state, color=color_palette_from_method["ground_truth"], ls="--", linewidth=0.5
        )
        ax_right_0.plot(
            t, obs, color=color_palette_from_method["observation"], linestyle="None", marker=".", markersize=2
        )
        ax_right_0.set_ylabel("State")
        ax_right_0.set_title("Observations")

        for ax in [ax_right_1, ax_right_2, ax_left_1]:
            ax.plot(sampling_times, true_state, color=color_palette_from_method["ground_truth"], ls="--", linewidth=0.5)

        names = ["emergence", "persistence"]
        # Plot persistence and emergence belief
        for i in range(mixture_preds.shape[1]):
            ax_right_1.plot(
                query_times,
                mixture_preds[:, i],
                color=color_palette_from_method[names[i]],
                lw=1,
            )
        ax_right_1.set_ylabel("Belief")
        ax_right_1.set_title("Mixtures")

        ax_right_2.plot(
            query_times,
            belief,
            color=color_palette_from_method["perpetua"],
            lw=1,
        )
        ax_right_2.set_ylabel("Belief")
        ax_right_2.set_title("Perpetua")
        ax_right_2.set_xlabel("Time (t)")

        # Plot log evidence
        for i in range(log_evidence.shape[1]):
            ax_left_0.plot(
                query_times,
                log_evidence[:, i],
                color=color_palette_from_method[names[i]],
                lw=1,
            )
        ax_left_0.set_ylabel(r"$\log p(\mathcal{M} \mid \mathcal{Y}_{1:N})$")
        ax_left_0.set_title("Log Evidence")

        # Plot prior
        ax_left_1.plot(
            query_times,
            np.exp(log_prior),
            color=color_palette_from_method["emergence"],
            lw=1,
        )
        ax_left_1.set_ylim(-0.05, 1.05)
        ax_left_1.set_ylabel(r"$p(\mathcal{M} = 1)$")
        ax_left_1.set_title("Model Prior")

        # Plot model posterior
        ax_left_2.plot(
            query_times,
            np.exp(log_model_posterior[:, 0]),
            color=color_palette_from_method[names[0]],
            lw=1,
        )
        ax_left_2.set_ylabel(r"$p(\mathcal{M} = 1 \mid \mathcal{Y}_{1:N})$")
        ax_left_2.set_title("Model Posterior")
        ax_left_2.set_xlabel("Time (t)")

        # Define legend items with colors, linestyles, markers, and line widths
        legend_items = [
            ("True State", color_palette_from_method["ground_truth"], "--", None, 1),
            ("Observations", color_palette_from_method["observation"], "None", ".", 1),
            ("Persistence", color_palette_from_method["persistence"], "-", None, 1),
            ("Emergence", color_palette_from_method["emergence"], "-", None, 1),
            ("Perpetua", color_palette_from_method["perpetua"], "-", None, 1),
        ]

        # Create legend handles manually
        handles = [
            mlines.Line2D(
                [],
                [],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=6 if marker else 0,
                linewidth=linewidth,
                label=label,
            )
            for label, color, linestyle, marker, linewidth in legend_items
        ]

        # Create a standalone figure for the legend
        fig.legend(
            handles=handles,
            loc="lower center",
            frameon=True,
            ncol=len(handles),
            bbox_to_anchor=(0.5, -0.075),
        )
        fig.canvas.draw()

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False, fig_size=figure_size)
            plt.show()
        else:
            self.savefig(fig, fig_name=f_name, save=True, fig_size=figure_size)

        plt.close()


class WeeklyPlotter(Plotter):
    def plot_environment(
        self,
        scene,
        figure_size: Tuple[int, int] = (8, 8),
        interval: Tuple[int, int] = (0, -1),
        f_name: str = "all_mixture_death_filters",
    ) -> None:
        # Create a figure with three vertically arranged subplots
        fig, axes = plt.subplots(
            scene.n_pickupables, 1, figsize=figure_size, sharex=True, dpi=300, constrained_layout=True
        )
        titles = [
            "TV Remote",
            "Static 1",
            "Car Keys",
            "Static 2",
            "Coffee Mug",
            "Static 5",
            "Guitar Case",
            "Static 7",
        ]

        r_id = 0  # This data has only one receptacle
        for l, pickupable in enumerate(scene):
            indices_gt = np.logical_and(
                pickupable[r_id].gt_timestamps >= interval[0], pickupable[r_id].gt_timestamps <= interval[1]
            )
            indices_obs = np.logical_and(
                pickupable[r_id].timestamps >= interval[0], pickupable[r_id].timestamps <= interval[1]
            )
            # Convert minutes → datetime for proper weekday ticks
            gt_t = minutes_to_datetime(pickupable[r_id].gt_timestamps, "monday")
            obs_t = minutes_to_datetime(pickupable[r_id].timestamps, "monday")
            axes[l].plot(
                obs_t[indices_obs],
                pickupable[r_id].states[indices_obs],
                color=color_palette_from_method["observation"],
                linestyle="None",
                marker=".",
                markersize=1,
            )
            axes[l].plot(
                gt_t[indices_gt],
                pickupable[r_id].gt_states[indices_gt],
                color=color_palette_from_method["ground_truth"],
                ls="--",
                linewidth=0.5,
            )
            axes[l].set_title(titles[l], fontsize=8)
            # Format x-axis as days of the week
            axes[l].xaxis.set_major_locator(mdates.DayLocator())  # ticks per day
            axes[l].xaxis.set_major_formatter(mdates.DateFormatter("%a"))  # Mon, Tue, ...

        # Make x label bigger
        axes[-1].set_xlabel("Day of Week")

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False, fig_size=figure_size)
            plt.show()
        else:
            # Store the figure with a large DPI
            self.savefig(fig, fig_name=f_name, save=True, fig_size=figure_size)
        plt.close()

    def plot_schedule(
        self,
        scene: Dict[int, Dict[str, np.ndarray]],
        plt_gt: bool = False,
        title: str = "Schedule",
        f_name: str = "mixtures_evidence",
    ):
        im_width, im_height = self.PAPER_COLUMN * self.pt, 10 * self.cm
        figure_size = (im_width, im_height)
        # Create a figure with three vertically arranged subplots
        fig, ax = plt.subplots(figsize=figure_size, dpi=300, constrained_layout=True)
        # Define a colormap from red (0) → yellow (0.5) → green (1)
        cmap = LinearSegmentedColormap.from_list(
            "red_yellow_green",
            [
                (0, color_palette_from_str["pink"]),
                (0.5, color_palette_from_str["mustard"]),
                (1, color_palette_from_str["teal"]),
            ],
        )
        # Normalize values 0-1
        norm = colors.Normalize(vmin=0, vmax=1)
        titles = [
            "TV\nRemote\n(In Drawer)",
            "Book\nShelf",
            "Car\nKeys\n(On Holder)",
            "Family\nPhoto",
            "Coffee\nMug\n(On Desk)",
            "TV",
            "Guitar\n(In Case)",
            "Couch",
        ][: scene.n_pickupables]
        # This data has only one receptacle
        r_id = 0
        if plt_gt:
            all_obs = np.array([scene[l][r_id].gt_states for l in range(scene.n_pickupables)])
            all_times = [
                minutes_to_datetime(scene[l][r_id].gt_timestamps, "monday") for l in range(scene.n_pickupables)
            ]
        else:
            all_obs = np.array([scene[l][r_id].states for l in range(scene.n_pickupables)])
            all_times = [minutes_to_datetime(scene[l][r_id].timestamps, "monday") for l in range(scene.n_pickupables)]
        # We'll assume all sequences have the same length
        time_nums = mdates.date2num(all_times[0])  # shape (n_times,)
        ax.imshow(
            all_obs,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            extent=[time_nums[0], time_nums[-1], 0, all_obs.shape[0]],
            origin="upper",
        )
        # Draw horizontal lines to separate rows
        for y in range(1, scene.n_pickupables):
            ax.hlines(y, xmin=time_nums[0], xmax=time_nums[-1], color="black", linewidth=0.5)

        # Set vertical labels
        ax.set_yticks(np.arange(scene.n_pickupables) + 0.5)  # center of each row
        ax.set_yticklabels(titles[::-1], va="center")
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a"))
        ax.set_xlabel("Day of Week")

        legend_elements = [
            mlines.Line2D([0], [0], color=color_palette_from_str["teal"], label="Present"),
            mlines.Line2D([0], [0], color=color_palette_from_str["pink"], label="Absent"),
        ]

        lgnd = ax.legend(
            handles=legend_elements,
            edgecolor=(1, 1, 1, 0),
            framealpha=0.9,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.075),
            ncol=2,
        )

        # This changes the size of the line in the legend
        for handle in lgnd.get_lines():
            handle.set_linewidth(2)

        # Update marker size in legend
        for handle in lgnd.legend_handles:  # note: legendHandles for Line2D objects
            handle.set_markersize(6)  # adjust marker size in legend

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False, fig_size=figure_size)
            plt.show()
        else:
            # Store the figure with a large DPI
            self.savefig(fig, fig_name=f_name, save=True, fig_size=figure_size)
        plt.close()

    def plot_results(
        self,
        scene,
        predictions_list: List[np.ndarray],
        figure_size: Tuple[int, int] = (8, 8),
        interval: Tuple[int, int] = (0, -1),
        f_name: str = "pred_test_weekly",
    ) -> None:
        # Create a figure with three vertically arranged subplots
        fig, axes = plt.subplots(
            scene.n_pickupables, 1, figsize=figure_size, sharex=True, dpi=300, constrained_layout=True
        )
        titles = [
            "TV Remote",
            "Static 1",
            "Car Keys",
            "Static 2",
            "Coffee Mug",
            "Static 5",
            "Guitar Case",
            "Static 7",
        ]

        r_id = 0  # This data has only one receptacle
        for p_id in range(scene.n_pickupables):
            indices_gt = np.logical_and(
                scene[p_id][r_id].gt_timestamps > interval[0], scene[p_id][r_id].gt_timestamps <= interval[1]
            )
            # Convert minutes → datetime for proper weekday ticks
            gt_t = minutes_to_datetime(scene[p_id][r_id].gt_timestamps, "monday")
            axes[p_id].plot(
                gt_t[indices_gt],
                predictions_list[p_id][indices_gt],
                color=color_palette_from_method["perpetua"],
                ls="-",
                linewidth=1.0,
            )
            axes[p_id].plot(
                gt_t[indices_gt],
                scene[p_id][r_id].gt_states[indices_gt],
                color=color_palette_from_method["ground_truth"],
                ls="--",
                linewidth=1,
            )
            axes[p_id].set_title(titles[p_id], fontsize=8, y=0.85)
            # Format x-axis as days of the week
            axes[p_id].xaxis.set_major_locator(mdates.DayLocator())  # ticks per day
            axes[p_id].xaxis.set_major_formatter(mdates.DateFormatter("%a"))  # Mon, Tue, ...

        # Make x label bigger
        axes[-1].set_xlabel("Day of Week")

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False, fig_size=figure_size)
            plt.show()
        else:
            # Store the figure with a large DPI
            self.savefig(fig, fig_name=f_name, save=True, fig_size=figure_size)
        plt.close()


class ProcthorPlotter(Plotter):
    def plot_environment(
        self,
        pickupable,
        figure_size: Tuple[int, int] = (8, 8),
        interval: Tuple[int, int] = (0, -1),
        f_name: str = "all_mixture_death_filters",
    ) -> None:
        # Create a figure with three vertically arranged subplots
        fig, axes = plt.subplots(
            pickupable.n_receptacles, 1, figsize=figure_size, sharex=True, dpi=300, constrained_layout=True
        )
        if pickupable.n_receptacles == 1:
            axes = [axes]  # Make sure axes is always a list even

        for l, receptacle in enumerate(pickupable):
            indices_gt = np.logical_and(
                receptacle.gt_timestamps >= interval[0], receptacle.gt_timestamps <= interval[1]
            )
            indices_obs = np.logical_and(receptacle.timestamps >= interval[0], receptacle.timestamps <= interval[1])
            # Convert minutes → datetime for proper weekday ticks
            obs_t = minutes_to_datetime(receptacle.timestamps * 60, "monday")
            gt_t = minutes_to_datetime(receptacle.gt_timestamps * 60, "monday")
            axes[l].plot(
                obs_t[indices_obs],
                receptacle.states[indices_obs],
                color=color_palette_from_method["observation"],
                linestyle="None",
                marker=".",
                markersize=1,
            )
            axes[l].plot(
                gt_t[indices_gt],
                receptacle.gt_states[indices_gt],
                color=color_palette_from_method["ground_truth"],
                ls="--",
                linewidth=0.5,
            )
            # Format x-axis as days of the week
            axes[l].set_title(f"{pickupable.name} @ {receptacle.name}", fontsize=8)
            axes[l].xaxis.set_major_locator(mdates.DayLocator())  # ticks per day
            axes[l].xaxis.set_major_formatter(mdates.DateFormatter("%a"))  # Mon, Tue, ...

        # Make x label bigger
        axes[-1].set_xlabel("Day of Week")

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False, fig_size=figure_size)
            plt.show()
        else:
            # Store the figure with a large DPI
            self.savefig(fig, fig_name=f_name, save=True, fig_size=figure_size)
        plt.close()

    def plot_pickupable(
        self,
        pickupable,
        pred_dict: dict,
        figure_size: Tuple[int, int] = (8, 8),
        interval: Tuple[int, int] = (0, -1),
        f_name: str = "pred_test_weekly",
    ) -> None:
        # Create a figure with three vertically arranged subplots
        fig, axes = plt.subplots(
            pickupable.n_receptacles, 1, figsize=figure_size, sharex=True, dpi=300, constrained_layout=True
        )
        if pickupable.n_receptacles == 1:
            axes = [axes]  # Make sure axes is always a list even

        p_name = pickupable.name

        for i in range(pickupable.n_receptacles):
            query_times = pickupable[i].gt_timestamps
            gt_states = pickupable[i].gt_states
            r_id = pickupable[i].id
            r_name = pickupable[i].name

            # Get predictions
            pred = pred_dict["receptacles"][r_id]["prediction"]
            # Convert minutes → datetime for proper weekday ticks
            query_times = minutes_to_datetime(query_times * 60, "monday")
            axes[i].plot(
                query_times,
                pred,
                color=color_palette_from_method["perpetua"],
                ls="-",
                linewidth=1.0,
            )
            axes[i].plot(
                query_times,
                gt_states,
                color=color_palette_from_method["ground_truth"],
                ls="--",
                linewidth=1,
            )
            axes[i].set_title(f"{p_name} @ {r_name }", fontsize=8)
            axes[i].set_ylim(-0.1, 1.1)
            # Format x-axis as days of the week
            axes[i].xaxis.set_major_locator(mdates.DayLocator())  # ticks per day
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%a"))  # Mon, Tue, ...

        # Make x label bigger
        axes[-1].set_xlabel("Day of Week")

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False, fig_size=figure_size)
            plt.show()
        else:
            # Store the figure with a large DPI
            self.savefig(fig, fig_name=f_name, save=True, fig_size=figure_size)
        plt.close()


class PlotterPaperResults(Plotter):
    def plot_weekly_results(
        self,
        query_times: np.ndarray,
        true_state: np.ndarray,
        observations: np.ndarray,
        obs_times: np.ndarray,
        predictions: List[np.ndarray],
        methods: List[str],
        interval: Tuple[int, int] = (0, -1),
        f_name: str = "pred_test_weekly",
    ) -> None:
        im_width, im_height = self.PAPER_COLUMN * self.pt, 5.2 * self.cm
        figure_size = (im_width, im_height)
        n_cols = 2
        n_rows = int(np.ceil((len(methods) + 1) / n_cols))
        # Create a grid with the observations first and one subplot per method
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figure_size, sharex=True, dpi=300, constrained_layout=True)
        axes = axes.ravel()

        # Filter desired range
        indices_gt = np.logical_and(query_times > interval[0], query_times <= interval[1])
        indices_t = np.logical_and(obs_times > interval[0], obs_times <= interval[1])
        # Convert minutes → datetime for proper weekday ticks
        gt_t = minutes_to_datetime(query_times, "monday")
        t = minutes_to_datetime(obs_times, "monday")

        # Plot true state
        axes[0].plot(
            gt_t[indices_gt],
            true_state[indices_gt],
            color=grey_palette_from_str["60"],
            ls="-",
            linewidth=0.5,
            label="GT",
        )

        # Plot observations
        axes[0].plot(
            t[indices_t],
            observations[indices_t],
            color=color_palette_from_method["observation"],
            linestyle="None",
            marker=".",
            markersize=1,
            label="Observation",
        )
        # Format x-axis as days of the week
        axes[0].xaxis.set_major_locator(mdates.DayLocator())  # ticks per day
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%a"))  # Mon, Tue, ...
        axes[0].legend(loc="upper right", numpoints=1, frameon=False)

        # FreMEn, Perpetua and Perpetua* predictions
        linestyles = ["-", "-", "-", ":", "-.", "--"]
        for i, (method, prediction) in enumerate(zip(methods, predictions)):
            if method == "Perpetua":
                color = color_palette_from_method["perpetua"]
            elif method == "FreMEn":
                color = color_palette_from_method["fremen"]
            else:  # "Perpetua*"
                color = color_palette_from_method["perpetua-star"]
            axes[i + 1].plot(
                gt_t[indices_gt],
                prediction[indices_gt],
                color=color,
                ls=linestyles[i],
                linewidth=1.0,
                label=f"{method}",
                zorder=2,
            )
            # Plot true state
            axes[i + 1].plot(
                gt_t[indices_gt],
                true_state[indices_gt],
                color=grey_palette_from_str["60"],
                ls="-",
                linewidth=0.5,
                zorder=1,
            )
            # Format x-axis as days of the week
            axes[i + 1].xaxis.set_major_locator(mdates.DayLocator())  # ticks per day
            axes[i + 1].xaxis.set_major_formatter(mdates.DateFormatter("%a"))  # Mon, Tue, ...
            axes[i + 1].legend(loc="upper right", numpoints=1, frameon=False)

        for ax in axes:
            ax.tick_params(labelsize=5.5)
            lgnd = ax.legend(
                edgecolor=(1, 1, 1, 0),
                framealpha=0.9,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=2,
                fontsize=7,
            )

            # This changes the size of the line in the legend
            for handle in lgnd.get_lines():
                handle.set_linewidth(2)

            # Update marker size in legend
            for handle in lgnd.legend_handles:  # note: legendHandles for Line2D objects
                handle.set_markersize(6)  # adjust marker size in legend

        for ax in axes[-n_cols:]:
            ax.set_xlabel("Day of Week")

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False, fig_size=figure_size)
            plt.show()
        else:
            # Store the figure with a large DPI
            self.savefig(fig, fig_name=f_name, save=True, fig_size=figure_size)
        plt.close()
