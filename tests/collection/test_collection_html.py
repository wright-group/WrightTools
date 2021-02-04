import WrightTools as wt

results = wt.Collection(name='results')
results.create_data(name='neat')
results.create_data(name='messy')
results.create_data(name='confusing')
calibration = results.create_collection(name='calibration')
calibration.create_data(name='OPA1_tune_test')
calibration.create_data(name='OPA2_tune_test')

for name in results.item_names:
    item = results[name]
    print(isinstance(item,wt.Data))
    print(item)

