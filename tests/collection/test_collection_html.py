import WrightTools as wt
import numpy as np

def my_resonance(xi, yi, intensity=1, FWHM=500, x0=7000):
    def single(arr, intensity=intensity, FWHM=FWHM, x0=x0):
        return intensity*(0.5*FWHM)**2/((xi-x0)**2+(0.5*FWHM)**2)
    return single(xi) * single(yi)
xi = np.linspace(6000, 8000, 75)[:, None]
yi = np.linspace(6000, 8000, 75)[None, :]
zi = my_resonance(xi, yi)


results = wt.Collection(name='results')
results.create_data(name='neat')
results.neat.create_variable(name='w1', units='wn', values=xi)
results.neat.create_variable(name='w2', units='wn', values=yi)
results.neat.create_channel(name='signal', values=zi)
results.neat.transform('w1', 'w2')

results.create_data(name='messy')
results.messy.create_variable(name='w1', units='wn', values=xi)
results.messy.create_variable(name='w2', units='wn', values=yi)
results.messy.create_channel(name='signal', values=zi)
results.messy.transform('w1', 'w2')

results.create_data(name='confusing')
results.confusing.create_variable(name='w1', units='wn', values=xi)
results.confusing.create_variable(name='w2', units='wn', values=yi)
results.confusing.create_channel(name='signal', values=zi)
results.confusing.transform('w1', 'w2')

calibration = results.create_collection(name='calibration')
calibration.create_data(name='OPA1_tune_test')
calibration.OPA1_tune_test.create_variable(name='w1', units='wn', values=xi)
calibration.OPA1_tune_test.create_variable(name='w2', units='wn', values=yi)
calibration.OPA1_tune_test.create_channel(name='signal', values=zi)
calibration.OPA1_tune_test.transform('w1', 'w2')

calibration.create_data(name='OPA2_tune_test')
calibration.OPA2_tune_test.create_variable(name='w1', units='wn', values=xi)
calibration.OPA2_tune_test.create_variable(name='w2', units='wn', values=yi)
calibration.OPA2_tune_test.create_channel(name='signal', values=zi)
calibration.OPA2_tune_test.transform('w1', 'w2')

displays= results.create_collection(name='displays')
displays.create_data(name='figure1')
displays.figure1.create_variable(name='w1', units='wn', values=xi)
displays.figure1.create_variable(name='w2', units='wn', values=yi)
displays.figure1.create_channel(name='signal', values=zi)
displays.figure1.transform('w1', 'w2')

displayfig2 = displays.create_collection(name='figure2')
displayfig2.create_data(name='main')
displayfig2.main.create_variable(name='w1', units='wn', values=xi)
displayfig2.main.create_variable(name='w2', units='wn', values=yi)
displayfig2.main.create_channel(name='signal', values=zi)
displayfig2.main.transform('w1', 'w2')

displayfig2.create_data(name='inset')
displayfig2.inset.create_variable(name='w1', units='wn', values=xi)
displayfig2.inset.create_variable(name='w2', units='wn', values=yi)
displayfig2.inset.create_channel(name='signal', values=zi)
displayfig2.inset.transform('w1', 'w2')

wt.Collection.convert(results,units='eV')

#alternating indices should be enough to prove all are eV
print(results.neat.units[0]=='eV')
print(results.messy.units[1]=='eV')
print(results.confusing.units[0]=='eV')
print(results.calibration.OPA1_tune_test.units[1]=='eV')
print(results.calibration.OPA2_tune_test.units[0]=='eV')
print(results.displays.figure1.units[1]== 'eV')
print(results.displays.figure2.main.units[0]== 'eV')
print(results.displays.figure2.inset.units[1]== 'eV')
