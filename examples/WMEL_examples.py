import matplotlib.pyplot as plt
import WrightTools.diagrams.WMEL as WMEL

### off-diagonal TRIEE ########################################################

off_diagonal = WMEL.Artist(size = [6, 2],
                           energies = [0., 0.43, 0.57, 1.],
                           state_names = ['g', 'a', 'b', 'a+b'])
               
off_diagonal.label_rows([r'$\mathrm{\alpha}$', r'$\mathrm{\beta}$', r'$\mathrm{\gamma}$'])
off_diagonal.label_columns(['I', 'II', 'III', 'IV', 'V', 'VI'])

# pw1 alpha
off_diagonal.add_arrow([0, 0], 0, [0, 1], 'ket', '1')
off_diagonal.add_arrow([0, 0], 1, [0, 2], 'bra', '-2')
off_diagonal.add_arrow([0, 0], 2, [2, 0], 'bra', '2\'')
off_diagonal.add_arrow([0, 0], 3, [1, 0], 'out')

# pw1 beta
off_diagonal.add_arrow([0, 1], 0, [0, 1], 'ket', '1')
off_diagonal.add_arrow([0, 1], 1, [0, 2], 'bra', '-2')
off_diagonal.add_arrow([0, 1], 2, [1, 3], 'ket', '2\'')
off_diagonal.add_arrow([0, 1], 3, [3, 2], 'out')

# pw2 alpha
off_diagonal.add_arrow([1, 0], 0, [0, 1], 'ket', '1')
off_diagonal.add_arrow([1, 0], 1, [1, 3], 'ket', '2\'')
off_diagonal.add_arrow([1, 0], 2, [3, 1], 'ket', '-2')
off_diagonal.add_arrow([1, 0], 3, [1, 0], 'out')

# pw2 beta
off_diagonal.add_arrow([1, 1], 0, [0, 1], 'ket', '1')
off_diagonal.add_arrow([1, 1], 1, [1, 3], 'ket', '2\'')
off_diagonal.add_arrow([1, 1], 2, [0, 2], 'bra', '-2')
off_diagonal.add_arrow([1, 1], 3, [3, 2], 'out')

# pw3 alpha
off_diagonal.add_arrow([2, 0], 0, [0, 2], 'bra', '-2')
off_diagonal.add_arrow([2, 0], 1, [0, 1], 'ket', '1')
off_diagonal.add_arrow([2, 0], 2, [2, 0], 'bra', '2\'')
off_diagonal.add_arrow([2, 0], 3, [1, 0], 'out')

# pw3 beta
off_diagonal.add_arrow([2, 1], 0, [0, 2], 'ket', '-2')
off_diagonal.add_arrow([2, 1], 1, [0, 1], 'ket', '1')
off_diagonal.add_arrow([2, 1], 2, [1, 3], 'bra', '2\'')
off_diagonal.add_arrow([2, 1], 3, [3, 2], 'out')

# pw4 alpha
off_diagonal.add_arrow([3, 0], 0, [0, 2], 'ket', '2\'')
off_diagonal.add_arrow([3, 0], 1, [2, 3], 'ket', '1')
off_diagonal.add_arrow([3, 0], 2, [3, 1], 'ket', '-2')
off_diagonal.add_arrow([3, 0], 3, [1, 0], 'out')

# pw4 beta
off_diagonal.add_arrow([3, 1], 0, [0, 2], 'ket', '2\'')
off_diagonal.add_arrow([3, 1], 1, [2, 3], 'ket', '1')
off_diagonal.add_arrow([3, 1], 2, [0, 2], 'bra', '-2')
off_diagonal.add_arrow([3, 1], 3, [3, 2], 'out')

# pw5 alpha
off_diagonal.add_arrow([4, 0], 0, [0, 2], 'bra', '-2')
off_diagonal.add_arrow([4, 0], 1, [2, 0], 'bra', '2\'')
off_diagonal.add_arrow([4, 0], 2, [0, 1], 'ket', '1')
off_diagonal.add_arrow([4, 0], 3, [1, 0], 'out')

# pw5 beta
off_diagonal.add_arrow([4, 1], 0, [0, 2], 'bra', '-2')
off_diagonal.add_arrow([4, 1], 1, [0, 2], 'ket', '2\'')
off_diagonal.add_arrow([4, 1], 2, [2, 3], 'ket', '1')
off_diagonal.add_arrow([4, 1], 3, [3, 2], 'out')

# pw6 alpha
off_diagonal.add_arrow([5, 0], 0, [0, 2], 'ket', '2\'')
off_diagonal.add_arrow([5, 0], 1, [2, 0], 'ket', '-2')
off_diagonal.add_arrow([5, 0], 2, [0, 1], 'ket', '1')
off_diagonal.add_arrow([5, 0], 3, [1, 0], 'out')

# pw6 beta
off_diagonal.add_arrow([5, 1], 0, [0, 2], 'ket', '2\'')
off_diagonal.add_arrow([5, 1], 1, [0, 2], 'bra', '-2')
off_diagonal.add_arrow([5, 1], 2, [2, 3], 'ket', '1')
off_diagonal.add_arrow([5, 1], 3, [3, 2], 'out')

off_diagonal.plot('WMEL_off_diagonal.png')
plt.close()

### on-diagonal TRIEE #########################################################

on_diagonal = WMEL.Artist(size = [6, 3],
                          energies = [0., .5, 1.],
                          state_names = ['g', 'a', 'b', 'a+b'])
               
on_diagonal.label_rows([r'$\mathrm{\alpha}$', r'$\mathrm{\beta}$', r'$\mathrm{\gamma}$'])
on_diagonal.label_columns(['I', 'II', 'III', 'IV', 'V', 'VI'])

on_diagonal.clear_diagram([1, 2])
on_diagonal.clear_diagram([3, 2])

# pw1 alpha
on_diagonal.add_arrow([0, 0], 0, [0, 1], 'ket', '1')
on_diagonal.add_arrow([0, 0], 1, [0, 1], 'bra', '-2')
on_diagonal.add_arrow([0, 0], 2, [1, 0], 'bra', '2\'')
on_diagonal.add_arrow([0, 0], 3, [1, 0], 'out')

# pw1 beta
on_diagonal.add_arrow([0, 1], 0, [0, 1], 'ket', '1')
on_diagonal.add_arrow([0, 1], 1, [0, 1], 'bra', '-2')
on_diagonal.add_arrow([0, 1], 2, [1, 2], 'ket', '2\'')
on_diagonal.add_arrow([0, 1], 3, [2, 1], 'out')

# pw1 gamma
on_diagonal.add_arrow([0, 2], 0, [0, 1], 'ket', '1')
on_diagonal.add_arrow([0, 2], 1, [1, 0], 'ket', '-2')
on_diagonal.add_arrow([0, 2], 2, [0, 1], 'ket', '2\'')
on_diagonal.add_arrow([0, 2], 3, [1, 0], 'out')

# pw2 alpha
on_diagonal.add_arrow([1, 0], 0, [0, 1], 'ket', '1')
on_diagonal.add_arrow([1, 0], 1, [1, 2], 'ket', '2\'')
on_diagonal.add_arrow([1, 0], 2, [2, 1], 'ket', '-2')
on_diagonal.add_arrow([1, 0], 3, [1, 0], 'out')

# pw2 beta
on_diagonal.add_arrow([1, 1], 0, [0, 1], 'ket', '1')
on_diagonal.add_arrow([1, 1], 1, [1, 2], 'ket', '2\'')
on_diagonal.add_arrow([1, 1], 2, [0, 1], 'bra', '-2')
on_diagonal.add_arrow([1, 1], 3, [2, 1], 'out')

# pw3 alpha
on_diagonal.add_arrow([2, 0], 0, [0, 1], 'bra', '-2')
on_diagonal.add_arrow([2, 0], 1, [0, 1], 'ket', '1')
on_diagonal.add_arrow([2, 0], 2, [1, 0], 'bra', '2\'')
on_diagonal.add_arrow([2, 0], 3, [1, 0], 'out')

# pw3 beta
on_diagonal.add_arrow([2, 1], 0, [0, 1], 'bra', '-2')
on_diagonal.add_arrow([2, 1], 1, [0, 1], 'ket', '1')
on_diagonal.add_arrow([2, 1], 2, [1, 2], 'ket', '2\'')
on_diagonal.add_arrow([2, 1], 3, [2, 1], 'out')

# pw3 gamma
on_diagonal.add_arrow([2, 2], 0, [0, 1], 'bra', '-2')
on_diagonal.add_arrow([2, 2], 1, [1, 0], 'bra', '1')
on_diagonal.add_arrow([2, 2], 2, [0, 1], 'ket', '2\'')
on_diagonal.add_arrow([2, 2], 3, [1, 0], 'out')

# pw4 alpha
on_diagonal.add_arrow([3, 0], 0, [0, 1], 'ket', '2\'')
on_diagonal.add_arrow([3, 0], 1, [1, 2], 'ket', '1')
on_diagonal.add_arrow([3, 0], 2, [2, 1], 'ket', '-2')
on_diagonal.add_arrow([3, 0], 3, [1, 0], 'out')

# pw4 beta
on_diagonal.add_arrow([3, 1], 0, [0, 1], 'ket', '2\'')
on_diagonal.add_arrow([3, 1], 1, [1, 2], 'ket', '1')
on_diagonal.add_arrow([3, 1], 2, [0, 1], 'bra', '-2')
on_diagonal.add_arrow([3, 1], 3, [2, 1], 'out')

# pw5 alpha
on_diagonal.add_arrow([4, 0], 0, [0, 1], 'bra', '-2')
on_diagonal.add_arrow([4, 0], 1, [1, 0], 'bra', '2\'')
on_diagonal.add_arrow([4, 0], 2, [0, 1], 'ket', '1')
on_diagonal.add_arrow([4, 0], 3, [1, 0], 'out')

# pw5 beta
on_diagonal.add_arrow([4, 1], 0, [0, 1], 'bra', '-2')
on_diagonal.add_arrow([4, 1], 1, [0, 1], 'ket', '2\'')
on_diagonal.add_arrow([4, 1], 2, [1, 2], 'ket', '1')
on_diagonal.add_arrow([4, 1], 3, [2, 1], 'out')

# pw5 gamma
on_diagonal.add_arrow([4, 2], 0, [0, 1], 'bra', '-2')
on_diagonal.add_arrow([4, 2], 1, [0, 1], 'ket', '2\'')
on_diagonal.add_arrow([4, 2], 2, [1, 0], 'bra', '1')
on_diagonal.add_arrow([4, 2], 3, [1, 0], 'out')

# pw6 alpha
on_diagonal.add_arrow([5, 0], 0, [0, 1], 'ket', '2\'')
on_diagonal.add_arrow([5, 0], 1, [1, 0], 'ket', '-2')
on_diagonal.add_arrow([5, 0], 2, [0, 1], 'ket', '1')
on_diagonal.add_arrow([5, 0], 3, [1, 0], 'out')

# pw6 beta
on_diagonal.add_arrow([5, 1], 0, [0, 1], 'ket', '2\'')
on_diagonal.add_arrow([5, 1], 1, [0, 1], 'bra', '-2')
on_diagonal.add_arrow([5, 1], 2, [1, 2], 'ket', '1')
on_diagonal.add_arrow([5, 1], 3, [2, 1], 'out')

# pw6 beta
on_diagonal.add_arrow([5, 2], 0, [0, 1], 'ket', '2\'')
on_diagonal.add_arrow([5, 2], 1, [0, 1], 'bra', '-2')
on_diagonal.add_arrow([5, 2], 2, [1, 0], 'bra', '1')
on_diagonal.add_arrow([5, 2], 3, [1, 0], 'out')

on_diagonal.plot('WMEL_on_diagonal.png')
plt.close()

### TSF #######################################################################

tsf = WMEL.Artist(size = [1, 1],
                  energies = [0., 0.15, 0.25, 1.],
                  state_names = ['g', 'v', 'v+v\'', 'virt'],
                  virtual = [3.])
# pw1 alpha
tsf.add_arrow([0, 0], 0, [0, 1], 'ket', '1')
tsf.add_arrow([0, 0], 1, [1, 2], 'ket', '2')
tsf.add_arrow([0, 0], 2, [2, 3], 'ket', '800')
tsf.add_arrow([0, 0], 3, [3, 0], 'out')

tsf.plot('TSF.png')
plt.close()

### population transfer #######################################################

pop_transfer = WMEL.Artist(size = [4, 3],
                           energies = [0., 0.4, 0.5, 0.8, 0.9, 1.],
                           number_of_interactions = 6,
                           state_names = ['g', '1S', '1P', '2x 1S', '1S+1P', '2x 1P'])
               
pop_transfer.label_rows([r'$\mathrm{\alpha}$', r'$\mathrm{\beta}$', r'$\mathrm{\gamma}$'])
pop_transfer.label_columns(['diag before', 'cross before', 'diag after', 'cross after'], font_size = 8)

pop_transfer.clear_diagram([1, 2])
pop_transfer.clear_diagram([2, 2])

# diag before alpha
pop_transfer.add_arrow([0, 0], 0, [0, 2], 'ket', '-2')
pop_transfer.add_arrow([0, 0], 1, [2, 0], 'ket', '2\'')
pop_transfer.add_arrow([0, 0], 2, [0, 2], 'ket', '1')
pop_transfer.add_arrow([0, 0], 3, [2, 0], 'out')

# diag before beta
pop_transfer.add_arrow([0, 1], 0, [0, 2], 'ket', '-2')
pop_transfer.add_arrow([0, 1], 1, [0, 2], 'bra', '2\'')
pop_transfer.add_arrow([0, 1], 2, [2, 5], 'ket', '1')
pop_transfer.add_arrow([0, 1], 3, [5, 2], 'out')

# diag before gamma
pop_transfer.add_arrow([0, 2], 0, [0, 2], 'ket', '-2')
pop_transfer.add_arrow([0, 2], 1, [0, 2], 'bra', '2\'')
pop_transfer.add_arrow([0, 2], 2, [2, 0], 'bra', '1')
pop_transfer.add_arrow([0, 2], 3, [2, 0], 'out')

# cross before alpha
pop_transfer.add_arrow([1, 0], 0, [0, 2], 'ket', '-2')
pop_transfer.add_arrow([1, 0], 1, [2, 0], 'ket', '2\'')
pop_transfer.add_arrow([1, 0], 2, [0, 1], 'ket', '1')
pop_transfer.add_arrow([1, 0], 3, [1, 0], 'out')

# cross before beta
pop_transfer.add_arrow([1, 1], 0, [0, 2], 'ket', '-2')
pop_transfer.add_arrow([1, 1], 1, [0, 2], 'bra', '2\'')
pop_transfer.add_arrow([1, 1], 2, [2, 4], 'ket', '1')
pop_transfer.add_arrow([1, 1], 3, [4, 2], 'out')

# diag after alpha
pop_transfer.add_arrow([2, 0], 0, [0, 2], 'ket', '-2')
pop_transfer.add_arrow([2, 0], 1, [2, 0], 'ket', '2\'')
pop_transfer.add_arrow([2, 0], 4, [0, 2], 'ket', '1')
pop_transfer.add_arrow([2, 0], 5, [2, 0], 'out')

# diag after beta
pop_transfer.add_arrow([2, 1], 0, [0, 2], 'ket', '-2')
pop_transfer.add_arrow([2, 1], 1, [0, 2], 'bra', '2\'')
pop_transfer.add_arrow([2, 1], 2, [2, 1], 'ket')
pop_transfer.add_arrow([2, 1], 3, [2, 1], 'bra')
pop_transfer.add_arrow([2, 1], 4, [1, 4], 'ket', '1')
pop_transfer.add_arrow([2, 1], 5, [4, 1], 'out')

# cross after alpha
pop_transfer.add_arrow([3, 0], 0, [0, 2], 'ket', '-2')
pop_transfer.add_arrow([3, 0], 1, [2, 0], 'ket', '2\'')
pop_transfer.add_arrow([3, 0], 4, [0, 1], 'ket', '1')
pop_transfer.add_arrow([3, 0], 5, [1, 0], 'out')

# cross after beta
pop_transfer.add_arrow([3, 1], 0, [0, 2], 'ket', '-2')
pop_transfer.add_arrow([3, 1], 1, [0, 2], 'bra', '2\'')
pop_transfer.add_arrow([3, 1], 2, [2, 1], 'ket')
pop_transfer.add_arrow([3, 1], 3, [2, 1], 'bra')
pop_transfer.add_arrow([3, 1], 4, [1, 3], 'ket', '1')
pop_transfer.add_arrow([3, 1], 5, [3, 1], 'out')

# cross after gamma
pop_transfer.add_arrow([3, 2], 0, [0, 2], 'ket', '-2')
pop_transfer.add_arrow([3, 2], 1, [0, 2], 'bra', '2\'')
pop_transfer.add_arrow([3, 2], 2, [2, 1], 'ket')
pop_transfer.add_arrow([3, 2], 3, [2, 1], 'bra')
pop_transfer.add_arrow([3, 2], 4, [1, 0], 'bra', '1')
pop_transfer.add_arrow([3, 2], 5, [1, 0], 'out')

pop_transfer.plot('pop_transfer.png')
plt.close()
