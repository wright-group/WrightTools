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
        head_length=10,
        head_aspect=1,
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
        kind : {'ket', 'bra', 'outbra', 'outket'}
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
        y_pos = [self.energies[between[0]], self.energies[between[1]]]

        # calculate arrow length
        arrow_length = self.energies[between[1]] - self.energies[between[0]]
        arrow_end = self.energies[between[1]]
        if arrow_length > 0:
            direction = 1
        elif arrow_length < 0:
            direction = -1
        else:
            raise ValueError("between invalid!")

        length = abs(y_pos[0] - y_pos[1])
        if kind == "ket":
            line = self.ax.plot(x_pos, y_pos, linestyle="-", color=color, linewidth=2, zorder=9)
        elif kind == "bra":
            line = self.ax.plot(x_pos, y_pos, linestyle="--", color=color, linewidth=2, zorder=9)
        elif kind == "out":
            yi = np.linspace(y_pos[0], y_pos[1], 100)
            xi = (
                np.sin((yi - y_pos[0]) * int((1 / length) * 20) * 2 * np.pi * length) / 40
                + x_pos[0]
            )
            line = self.ax.plot(
                xi[:-5],
                yi[:-5],
                linestyle="-",
                color=color,
                linewidth=2,
                solid_capstyle="butt",
                zorder=9,
            )  # the yi[:-5] simply cuts off 5 line points to not show past the arrowhead
        elif kind == "outbra":
            yi = np.linspace(y_pos[0], y_pos[1], 100)
            xi = (
                np.sin((yi - y_pos[0]) * int((1 / length) * 20) * 2 * np.pi * length) / 40
                + x_pos[0]
            )
            counter = 0
            while counter - 13 <= len(yi):
                subyi = yi[counter : counter + 15]
                subxi = xi[counter : counter + 15]
                line = self.ax.plot(
                    subxi[:-5],
                    subyi[:-5],
                    linestyle="-",
                    color=color,
                    linewidth=2,
                    solid_capstyle="butt",
                    zorder=9,
                )
                counter += 13  # increment must be equal to the while condition
        else:
            raise ValueError("kind is not 'ket', 'bra', 'out' or 'outbra'.")
        # add arrow head
        dx = x_pos[1] - x_pos[0]
        dy = y_pos[1] - y_pos[0]

        xytext = (x_pos[1] - dx * 1e-2, y_pos[1] - dy * 1e-2)
        annotation = self.ax.annotate(
            "",
            xy=(x_pos[1], y_pos[1]),
            xytext=xytext,
            arrowprops=dict(
                fc=color,
                ec=color,
                shrink=0,
                headwidth=head_length * head_aspect,
                headlength=head_length,
                linewidth=0,
                zorder=10,
            ),
            size=25,
        )
        # add text
        text = self.ax.text(
            np.mean(x_pos), -0.15, label, fontsize=font_size, horizontalalignment="center"
        )
        return line, annotation.arrow_patch, text


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
        size : [columns, rows]
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
        self.cascades = {}  # stores grid address of cascades (if any) for referencing

        # create figure
        figsize = [int(size[0] * ((number_of_interactions + 1.0) / 6)), size[1] * 2.5]
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
        else:
            raise ValueError("unexpected value for 'kind': {kind}, expected ('ket', 'bra', 'out')")
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

    def add_cascade(
        self,
        diagram,
        number,
        number_of_interactions=4,
        titles=[None],
        title_font_size=16,
        state_names=None,
        virtual=[None],
        state_font_size=14,
        state_text_buffer=0.5,
        label_side="left",
        bbox_adjust=[0.2, 0.9, 0.2, 0.78],
        transfer_arrow_linewidth=1.5,
        transfer_arrow_offset=[0.5, 5],
    ):
        """Add cascading process as plot.

        Parameters
        ---------
        diagram : [row,column]
            row and column positions to place the cascade in the artist grid.
        number : int
            number of cascading processes to add in series
        number_of_interactions : int (optional)
            number of interactions for each sub process.
            Equal for all subprocesses. Default is 4.
        titles : list of str (optional)
            list of title for each process. Default is [None]
        title_font_size : int (optional)
            set the title font size. Default is 16
        state_names : list of str (optional)
            list of state names. Default is None
        virtal : list of int (optional)
            list of indices of virtual states. Default is [None]
        state_font_size : int (optional)
            set the font size for the labels. Default is 14
        state_text_buffer : float (optional)
            set the state text buffer. Default is 0.5
        label_side : str (optional)
            set the side on which the state label will appear. Either 'left' or 'right'. Default is 'left'
        bbox_adjust : list of floats (optional)
            adjusts the left, right, bottom, and top bounds of the cascade diagram. Default is bbox_adjust=[0.2, 0.9, 0.2, 0.78]
        transfer_arrow_linewidth = float (ooptional)
            adjusts the thickness of the arrow between cascading WMEL diagrams
        transfer_arrow_offset = [x_offset, y_offset] (optional)
            adjusts the label of the energy transfer arrow along the x and y axis
        """

        x, y = diagram[0], diagram[1]
        self.clear_diagram([x, y])  # clear prev. added single process in grid
        self.cascades[f"[{x},{y}]"] = []  # store list of cascades in grid for iteration

        gridspec = (
            self.subplots[x, y].get_subplotspec().get_gridspec()
        )  # get grid spec to confine cascade
        sfig = self.fig.add_subfigure(gridspec[x, y])  # create subfigure

        bbox = self.subplots[x - 1][y].get_position()  # get grid box position for cascade
        spec = {
            "width_ratios": [1 + 0 ** (i % 2) for i in range(3 * number - 3)]
        }  # define width ratios for subplots
        casc_subplot = sfig.subplots(
            1, 3 * number - 3, gridspec_kw=spec, subplot_kw={"position": bbox}
        )  # make subplots

        # empirical adjustment to subfigure bbox to align levels with other row plots
        sfig.subplots_adjust(
            left=bbox_adjust[0], bottom=bbox_adjust[2], right=bbox_adjust[1], top=bbox_adjust[3]
        )

        # plot the arrows between cascading processes as a sine arrow
        for arrowindex, arrowplot in enumerate(casc_subplot.flatten()[:-1]):
            if arrowindex % 2 == 1:
                xout = np.linspace(0, 4, 100)
                yout = [np.sin(x * 2 * np.pi) for x in xout]
                arrowplot.plot(xout, yout, color="k", linewidth=transfer_arrow_linewidth)
                arrowplot.set_ylim(-20, 20)  # sets the visible amplitude of the sine wave
                arr_lbl_x, arr_lbl_y = transfer_arrow_offset[0], transfer_arrow_offset[1]
                arrowplot.annotate(
                    r"$\mathrm{\hbar}$$\mathrm{\omega}$",
                    [xout[len(xout) // 4] - arr_lbl_x, yout[len(yout) // 2] + arr_lbl_y],
                )  # adds hv label
                arrowplot.arrow(
                    xout[-2], yout[-2], 0.0001, 0, head_width=2, head_length=1, fc="k", ec="k"
                )  # add arrowhead

        # plot the cascade's processes' diagrams
        for ind, plot in enumerate(casc_subplot.flatten()):
            plot.axis("off")
            if ind % 2 == 0:
                if titles != [None]:
                    subtitles = []
                    for t in titles:
                        subtitles.append(t)
                        subtitles.append("")
                elif titles == [None]:
                    subtitles = [""] * (2 * number_of_interactions)
                if ind == 0 and label_side == "left":
                    casc_wmel = Subplot(
                        plot,
                        energies=self.energies,
                        number_of_interactions=number_of_interactions,
                        title=subtitles[ind],
                        title_font_size=title_font_size,
                        state_names=state_names,
                        virtual=virtual,
                        state_font_size=state_font_size,
                        state_text_buffer=state_text_buffer,
                        label_side=label_side,
                    )  # Uses the Subplot class to make the each process diagram

                elif ind == len(subtitles) - 2 and label_side == "right":
                    casc_wmel = Subplot(
                        plot,
                        energies=self.energies,
                        number_of_interactions=number_of_interactions,
                        title=subtitles[ind],
                        title_font_size=title_font_size,
                        state_names=state_names,
                        virtual=virtual,
                        state_font_size=state_font_size,
                        state_text_buffer=state_text_buffer,
                        label_side=label_side,
                    )  # Uses the Subplot class to make the each process diagram

                else:
                    casc_wmel = Subplot(
                        plot,
                        energies=self.energies,
                        number_of_interactions=number_of_interactions,
                        title=subtitles[ind],
                        title_font_size=title_font_size,
                        virtual=virtual,
                    )  # Uses the Subplot class to make the each process diagram
                self.cascades[f"[{x},{y}]"].append(casc_wmel)

    def add_cascade_arrow(
        self,
        diagram,
        cascade_number,
        index,
        between,
        kind,
        label="",
        head_length=10,
        head_aspect=1,
        font_size=14,
        color="k",
    ):
        """Add arrow to cascade subprocess.

        Parameters
        ----------

        diagram : [row,column]
            row and column indices location of the cascade in the figure.
        cascade_number : int
            index of subprocess to add the arrow
        index : int
            interaction index in the subprocess to which the arrow will be added
        between : [start,stop]
            start and stop energy level indeces from which the arrow will point to
        kind : {'bra', 'ket', 'out'}
            the kind of interaction to add
        label : str (optional)
            interaction label. Default is ""
        head_length : int (optional)
            interaction arrow head length. Default is 10
        head_aspect : int (optional)
            set arrow head aspect ratio. Default is 1
        font_size : int (optional)
            set label font size. Default is 14
        color : str (optional)
            set arrow color. Default is black
        """

        x, y = diagram[0], diagram[1]
        self.cascades[f"[{x},{y}]"][cascade_number].add_arrow(
            index,
            between,
            kind,
            label=label,
            head_length=head_length,
            head_aspect=head_aspect,
            font_size=font_size,
            color=color,
        )
