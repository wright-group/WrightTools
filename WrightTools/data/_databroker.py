from . import Data


def from_databroker(run, dataset="primary"):
    describe = run.describe()
    md = describe["metadata"]
    start = md["start"]
    ds = run[dataset].read()
    shape = start.get("shape", (len(ds.time),))

    detectors = start.get("detectors", [])

    data = Data()
    print(md)
    # data.attrs.update(md)
    for var in ds:
        if var == "uid":
            continue
        if var.endswith("_busy"):
            continue
        if any(var.startswith(d) for d in detectors):
            data.create_channel(var, values=ds[var].data.reshape(shape))
        else:
            # TODO units, once they are in the dataset metadata
            data.create_variable(var, values=ds[var].data.reshape(shape))

    transform = [x[0] for x, ds_name in start["hints"]["dimensions"] if ds_name == dataset]
    data.transform(*transform)
    return data
