"""WMEL diagrams."""


# --- import --------------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt


# --- define --------------------------------------------------------------------------------------


# --- subplot -------------------------------------------------------------------------------------


class Subplot:
    """Subplot containing WMEL."""

    def __init__(
        self,
        ax,
        energies,
        number_of_interactions=4,
        title="",
        title_font_size=16,
        state_names=None,
        virtual=[None],
        state_font_size=14,
        state_text_buffer=0.5,
        label_side="left",
    ):
        """Subplot.

        Parameters
        ----------
        ax : matplotlib axis
            The axis.
        energies : 1D array-like
            Energies (scaled between 0 and 1)
        number_of_interactions : integer
            Number of interactions in diagram.
        title : string (optional)
            Title of subplot. Default is empty string.
        state_names: list of str (optional)
            list of the names of the states
        virtual: list of ints (optional)
            list of indexes of any vitual energy states
        state_font_size: numtype (optional)
            font size for the state lables
        state_text_buffer: numtype (optional)
            space between the energy level bars and the state labels
        """
        self.ax = ax
        self.energies = energies
        self.interactions = number_of_interactions
        self.state_names = state_names

        # Plot Energy Levels
        for i in range(len(self.energies)):
            if i in virtual:
                linestyle = "--"
            else:
                linestyle = "-"
            self.ax.axhline(self.energies[i], color="k", linewidth=2, ls=linestyle, zorder=5)

        # add state names
        if isinstance(state_names, list):
            for i in range(len(self.energies)):
                if label_side == "left":
                    ax.text(
                        -state_text_buffer,
                        energies[i],
                        state_names[i],
                        fontsize=state_font_size,
                        verticalalignment="center",
                        horizontalalignment="center",
                    )
                elif label_side == "right":
                    ax.text(
                        1 + state_text_buffer,
                        energies[i],
                        state_names[i],
                        fontsize=state_font_size,
                        verticalalignment="center",
                        horizontalalignment="center",
                    )
        # calculate interaction_positons
        self.x_pos = np.linspace(0, 1, number_of_interactions)
        # set limits
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.01, 1.01)
        # remove guff
        self.ax.axis("off")
        # title
        self.ax.set_title(title, fontsize=title_font_size)

    def add_arrow(
        self,
        index,
        between,
        kind,
        label="",
        head_length=0.1,
        head_aspect=2,
        font_size=14,
        color="k",
    ):
        """Add an arrow to the WMEL diagram.

        Parameters
        ----------
        index : integer
            The interaction, or start and stop interaction for the arrow.
        between : 2-element iterable of integers
            The inital and final state of the arrow
        kind : {'ket', 'bra'}
            The kind of interaction.
        label : string (optional)
            Interaction label. Default is empty string.
        head_length: number (optional)
            size of arrow head
        font_size : number (optional)
            Label font size. Default is 14.
        color : matplotlib color (optional)
            Arrow color. Default is black.

        Returns
        -------
        [line,arrow_head,text]
        """
        if hasattr(index, "index"):
            x_pos = list(index)
        else:
            x_pos = [index] * 2
        x_pos = [np.linspace(0, 1, self.interactions)[i] for i in x_pos]

        # calculate arrow length
        arrow_length = self.energies[between[1]] - self.energies[between[0]]
        arrow_end = self.energies[between[1]]
        if arrow_length > 0:
            direction = 1
            y_poss = [self.energies[between[0]], self.energies[between[1]] - head_length]
        elif arrow_length < 0:
            direction = -1
            y_poss = [self.energies[between[0]], self.energies[between[1]] + head_length]
        else:
            raise ValueError("between invalid!")

        length = abs(y_poss[0] - y_poss[1])
        if kind == "ket":
            line = self.ax.plot(x_pos, y_poss, linestyle="-", color=color, linewidth=2, zorder=9)
        elif kind == "bra":
            line = self.ax.plot(x_pos, y_poss, linestyle="--", color=color, linewidth=2, zorder=9)
        elif kind == "out":
            yi = np.linspace(y_poss[0], y_poss[1], 100)
            xi = (
                np.sin((yi - y_poss[0]) * int((1 / length) * 20) * 2 * np.pi * length) / 40
                + x_pos[0]
            )
            line = self.ax.plot(
                xi, yi, linestyle="-", color=color, linewidth=2, solid_capstyle="butt", zorder=9
            )
        else:
            raise ValueError("kind is not 'ket', 'out', or 'bra'.")
        # add arrow head
        arrow_head = self.ax.arrow(
            x_pos[1],
            arrow_end - head_length * direction,
            0,
            0.0001 * direction,
            head_width=head_length * head_aspect,
            head_length=head_length,
            fc=color,
            ec=color,
            linestyle="solid",
            linewidth=0,
            zorder=10,
        )
        # add text
        text = self.ax.text(
            np.mean(x_pos), -0.15, label, fontsize=font_size, horizontalalignment="center"
        )
        return line, arrow_head, text


# --- artist --------------------------------------------------------------------------------------


class Artist:
    """Dedicated WMEL figure artist."""

    def __init__(
        self,
        size,
        energies,
        state_names=None,
        number_of_interactions=4,
        virtual=[None],
        state_font_size=8,
        state_text_buffer=0.5,
    ):
        """Initialize.

        Parameters
        ----------
        size : [rows, collumns]
            Layout.
        energies : list of numbers
            State energies.
        state_names : list of strings (optional)
            State names. Default is None.
        number_of_interactions : integer (optional)
            Number of interactions. Default is 4.
        virtual : list of integers (optional)
            Indices of states which are virtual. Default is [None].
        state_font_size : number (optional)
            State font size. Default is 8.
        state_text_buffer : number (optional)
            Size of buffer around state text. Default is 0.5.
        """
        # create figure
        figsize = [int(size[0] * ((number_of_interactions + 1.) / 6.)), size[1] * 2.5]
        fig, (subplots) = plt.subplots(size[1], size[0], figsize=figsize)
        self.fig = fig
        # wrap subplots if need be
        if size == [1, 1]:
            self.subplots = np.array([[subplots]])
            plt.subplots_adjust(left=0.3)
        elif size[1] == 1:
            self.subplots = np.array([subplots])
        else:
            self.subplots = subplots
        # add energy levels
        self.energies = energies
        for plot in self.subplots.flatten():
            for i in range(len(self.energies)):
                if i in virtual:
                    linestyle = "--"
                else:
                    linestyle = "-"
                plot.axhline(energies[i], color="k", linewidth=2, linestyle=linestyle)
        # add state names to leftmost plots
        if state_names:
            for i in range(size[1]):
                plot = self.subplots[i][0]
                for j in range(len(self.energies)):
                    plot.text(
                        -state_text_buffer,
                        energies[j],
                        state_names[j],
                        fontsize=state_font_size,
                        verticalalignment="center",
                        horizontalalignment="center",
                    )
        # calculate interaction_positons
        self.x_pos = np.linspace(0, 1, number_of_interactions)
        # plot cleans up a bunch - call it now as well as later
        self.plot()

    def label_rows(self, labels, font_size=15, text_buffer=1.5):
        """Label rows.

        Parameters
        ----------
        labels : list of strings
            Labels.
        font_size : number (optional)
            Font size. Default is 15.
        text_buffer : number
            Buffer around text. Default is 1.5.
        """
        for i in range(len(self.subplots)):
            plot = self.subplots[i][-1]
            plot.text(
                text_buffer,
                0.5,
                labels[i],
                fontsize=font_size,
                verticalalignment="center",
                horizontalalignment="center",
            )

    def label_columns(self, labels, font_size=15, text_buffer=1.15):
        """Label columns.

        Parameters
        ----------
        labels : list of strings
            Labels.
        font_size : number (optional)
            Font size. Default is 15.
        text_buffer : number
            Buffer around text. Default is 1.5.
        """
        for i in range(len(labels)):
            plot = self.subplots[0][i]
            plot.text(
                0.5,
                text_buffer,
                labels[i],
                fontsize=font_size,
                verticalalignment="center",
                horizontalalignment="center",
            )

    def clear_diagram(self, diagram):
        """Clear diagram.

        Parameters
        ----------
        diagram : [column, row]
            Diagram to clear.
        """
        plot = self.subplots[diagram[1]][diagram[0]]
        plot.cla()

    def add_arrow(
        self, diagram, number, between, kind, label="", head_length=0.075, font_size=7, color="k"
    ):
        """Add arrow.

        Parameters
        ----------
        diagram : [column, row]
            Diagram position.
        number : integer
            Arrow position.
        between : [start, stop]
            Arrow span.
        kind : {'ket', 'bra', 'out'}
            Arrow style.
        label : string (optional)
            Arrow label. Default is ''.
        head_length : number (optional)
            Arrow head length. Default 0.075.
        font_size : number (optional)
            Font size. Default is 7.
        color : matplotlib color
            Arrow color. Default is 'k'.

        Returns
        -------
        list
            [line, arrow_head, text]
        """
        column, row = diagram
        x_pos = self.x_pos[number]
        # calculate arrow length
        arrow_length = self.energies[between[1]] - self.energies[between[0]]
        arrow_end = self.energies[between[1]]
        if arrow_length > 0:
            direction = 1
            y_poss = [self.energies[between[0]], self.energies[between[1]] - head_length]
        elif arrow_length < 0:
            direction = -1
            y_poss = [self.energies[between[0]], self.energies[between[1]] + head_length]
        else:
            raise ValueError("Variable between invalid")
        subplot = self.subplots[row][column]
        # add line
        length = abs(y_poss[0] - y_poss[1])
        if kind == "ket":
            line = subplot.plot([x_pos, x_pos], y_poss, linestyle="-", color=color, linewidth=2)
        elif kind == "bra":
            line = subplot.plot([x_pos, x_pos], y_poss, linestyle="--", color=color, linewidth=2)
        elif kind == "out":
            yi = np.linspace(y_poss[0], y_poss[1], 100)
            xi = (
                np.sin((yi - y_poss[0]) * int((1 / length) * 20) * 2 * np.pi * length) / 40 + x_pos
            )
            line = subplot.plot(
                xi, yi, linestyle="-", color=color, linewidth=2, solid_capstyle="butt"
            )
        # add arrow head
        arrow_head = subplot.arrow(
            self.x_pos[number],
            arrow_end - head_length * direction,
            0,
            0.0001 * direction,
            head_width=head_length * 2,
            head_length=head_length,
            fc=color,
            ec=color,
            linestyle="solid",
            linewidth=0,
        )
        # add text
        text = subplot.text(
            self.x_pos[number], -0.1, label, fontsize=font_size, horizontalalignment="center"
        )
        return line, arrow_head, text

    def plot(self, save_path=None, close=False, bbox_inches="tight", pad_inches=1):
        """Plot figure.

        Parameters
        ----------
        save_path : string (optional)
            Save path. Default is None.
        close : boolean (optional)
            Toggle automatic figure closure after plotting.
            Default is False.
        bbox_inches : number (optional)
            Bounding box size, in inches. Default is 'tight'.
        pad_inches : number (optional)
            Pad inches. Default is 1.
        """
        # final manipulations
        for plot in self.subplots.flatten():
            # set limits
            plot.set_xlim(-0.1, 1.1)
            plot.set_ylim(-0.1, 1.1)
            # remove guff
            plot.axis("off")
        # save
        if save_path:
            plt.savefig(
                save_path,
                transparent=True,
                dpi=300,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
            )
        # close
        if close:
            plt.close()
