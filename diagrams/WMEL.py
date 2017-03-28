### import ####################################################################


import numpy as np
import matplotlib.pyplot as plt


### define ####################################################################


### subplot ###################################################################


class Subplot:
    
    def __init__(self, ax, energies, interactions, title='', state_linestyles=None):
        '''
        Subplot.
        
        Parameters
        ----------
        ax : matplotlib axis
            The axis.
        energies : 1D array-like
            Energies (scaled between 0 and 1)
        interactions : integer
            
        '''
        self.ax = ax
        self.interactions = interactions
        # plot energies
        if state_linestyles is None:
            state_linestyles = ['-'] * interactions
        self.energies = energies
        for energy, linestyle in zip(self.energies, state_linestyles):
            self.ax.axhline(energy, color='k', linewidth=2, ls=linestyle)
        # set limits
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        # remove guff
        self.ax.axis('off')
        # title
        self.ax.set_title(title, fontsize=16)
        
    def add_arrow(self, index, between, kind, label, 
                  head_size=0.1, font_size=14, color='k'):
        x_pos = np.linspace(0, 1, self.interactions)[index]
        # calculate arrow length
        arrow_length = self.energies[between[1]] - self.energies[between[0]]
        arrow_end = self.energies[between[1]]
        if arrow_length > 0:
            direction = 1
            y_poss = [self.energies[between[0]], self.energies[between[1]] - head_size]
        elif arrow_length < 0:
            direction = -1
            y_poss = [self.energies[between[0]], self.energies[between[1]] + head_size]
        else:
            print('between invalid!')
            return
        # add line
        length = abs(y_poss[0] - y_poss[1])
        if kind == 'ket':
            line = self.ax.plot([x_pos, x_pos], y_poss, linestyle = '-', color = color, linewidth = 2)
        elif kind == 'bra':
            line = self.ax.plot([x_pos, x_pos], y_poss, linestyle = '--', color = color, linewidth = 2)
        elif kind == 'out':
            yi = np.linspace(y_poss[0], y_poss[1], 100)
            xi = np.sin((yi - y_poss[0])*int((1/length)*20)*2*np.pi*length)/40 + x_pos
            line = self.ax.plot(xi, yi, linestyle = '-', color = color, linewidth = 2, solid_capstyle='butt')
        # add arrow head
        arrow_head = self.ax.arrow(x_pos, arrow_end - head_size * direction, 
                                   0, 0.0001*direction,
                                   head_width=head_size*2, 
                                   head_length=head_size,
                                   fc=color, ec=color, linestyle='solid', linewidth=0)
        # add text
        text = self.ax.text(x_pos, -0.2, label, fontsize=font_size, horizontalalignment='center')


### artist ####################################################################


class Artist:

    def __init__(self, size, energies, state_names=None,
                 number_of_interactions=4, virtual=[None],
                 state_font_size=8, state_text_buffer=0.5):
        '''
        virtual a list of indicies
        '''
        # create figure
        figsize = [int(size[0]*((number_of_interactions+1.)/6.)), size[1]*2.5]
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
                if i in virtual: linestyle = '--'
                else: linestyle = '-'
                plot.axhline(energies[i], color = 'k', linewidth = 2, linestyle = linestyle)      
        # add state names to leftmost plots
        if state_names:
            for i in range(size[1]):
                plot = self.subplots[i][0]
                for i in range(len(self.energies)):
                    plot.text(-state_text_buffer, energies[i], state_names[i], fontsize = state_font_size, verticalalignment ='center', horizontalalignment ='center')
        # calculate interaction_positons 
        self.x_pos = np.linspace(0, 1, number_of_interactions)
        # plot cleans up a bunch - call it now as well as later
        self.plot()
    
    def label_rows(self, labels, font_size=15, text_buffer=1.5):
        for i in range(len(self.subplots)):
            plot = self.subplots[i][-1]
            plot.text(text_buffer, 0.5, labels[i], fontsize = font_size, verticalalignment ='center', horizontalalignment ='center')        

    def label_columns(self, labels, font_size=15, text_buffer=1.15):
        for i in range(len(labels)):
            plot = self.subplots[0][i]
            plot.text(0.5, text_buffer, labels[i], fontsize = font_size, verticalalignment ='center', horizontalalignment ='center')        
    
    def clear_diagram(self, diagram):
        plot = self.subplots[diagram[1]][diagram[0]]
        plot.cla()
        
    def add_arrow(self, diagram, number, between, kind, label = '', head_size = 0.075, font_size = 7, color = 'k'):
        '''
        kind one in [ket, bra, out] \n
        returns [line, arrow_head, text]
        '''
        column, row = diagram
        x_pos = self.x_pos[number]
        # calculate arrow length
        arrow_length = self.energies[between[1]] - self.energies[between[0]]
        arrow_end = self.energies[between[1]]
        if arrow_length > 0:
            direction = 1
            y_poss = [self.energies[between[0]], self.energies[between[1]] - head_size]
        elif arrow_length < 0:
            direction = -1
            y_poss = [self.energies[between[0]], self.energies[between[1]] + head_size]
        else:
            print('between invalid!')
            return
        subplot = self.subplots[row][column]
        # add line
        length = abs(y_poss[0] - y_poss[1])
        if kind == 'ket':
            line = subplot.plot([x_pos, x_pos], y_poss, linestyle = '-', color = color, linewidth = 2)
        elif kind == 'bra':
            line = subplot.plot([x_pos, x_pos], y_poss, linestyle = '--', color = color, linewidth = 2)
        elif kind == 'out':
            yi = np.linspace(y_poss[0], y_poss[1], 100)
            xi = np.sin((yi - y_poss[0])*int((1/length)*20)*2*np.pi*length)/40 + x_pos
            line = subplot.plot(xi, yi, linestyle = '-', color = color, linewidth = 2, solid_capstyle='butt')
        # add arrow head
        arrow_head = subplot.arrow(self.x_pos[number], arrow_end - head_size * direction, 
                                   0, 0.0001*direction,
                                   head_width=head_size*2, 
                                   head_length=head_size,
                                   fc=color, ec=color, linestyle='solid', linewidth=0)
        # add text
        text = subplot.text(self.x_pos[number], -0.1, label, fontsize=font_size, horizontalalignment='center')
        return line, arrow_head, text

    def plot(self, save_path=None, close=False, bbox_inches='tight', pad_inches=1):
        # final manipulations
        for plot in self.subplots.flatten():
            # set limits
            plot.set_xlim(-0.1, 1.1)
            plot.set_ylim(-0.1, 1.1)
            # remove guff
            plot.axis('off')
        # save
        if save_path:
            plt.savefig(save_path, transparent=True, dpi=300, bbox_inches=bbox_inches, pad_inches=pad_inches)
        # close
        if close:
            plt.close()


### testing ###################################################################


if __name__ == '__main__':
    # testing code

    plt.close('all')
    
    diagram = Artist(size = [6, 3],
                     energies = [0., 0.4, 0.6, 1.],
                     state_names = ['g', 'a', 'b', 'a+b'])
                     
    diagram.label_rows([r'$\mathrm{\alpha}$', r'$\mathrm{\beta}$', r'$\mathrm{\gamma}$'])
    diagram.label_columns(['I', 'II', 'III', 'IV', 'V', 'VI'])
    
    #pw1 alpha
    diagram.add_arrow([0, 0], 0, [0, 2], 'ket', '1')
    diagram.add_arrow([0, 0], 1, [0, 1], 'bra', '-2')
    diagram.add_arrow([0, 0], 2, [1, 0], 'bra', '2\'')
    diagram.add_arrow([0, 0], 3, [3, 0], 'out')
    
    diagram.add_arrow([1, 0], 3, [2, 0], 'out')
    
    diagram.clear_diagram([2, 1])
    
    diagram.plot('WMEL_out.png')
