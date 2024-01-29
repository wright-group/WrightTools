# --- import --------------------------------------------------------------------------------------


import click
import WrightTools as wt


# --- define --------------------------------------------------------------------------------------


@click.group()
@click.version_option(wt.__version__)
def cli():
    pass


@cli.command(name="tree", help="Print a given data tree.")
@click.argument("path", nargs=1)
@click.option("--internal_path", default="/", help="specify a path internal to the file.  Defaults to root")
@click.option("--depth", "-d", "-L", type=int, default=9, help="Depth to print.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Print a more detailed tree.")
def tree(path, internal_path, depth=9, verbose=False):
    # Create the data/collection object
    obj = wt.open(path)[internal_path]

    # If the object is a data object, it doesn't take depth as a parameter
    if isinstance(obj, wt.Data):
        obj.print_tree(verbose=verbose)
    else:
        obj.print_tree(verbose=verbose, depth=depth)


@cli.command(name="crawl", help="Crawl a directory and report the wt5 objects found.")
@click.argument("directory", default=".")
def crawl(directory=None):
    ...

@cli.command(name="convert", help="Convert numbers to different units.")
@click.argument("number", type=float, nargs=1)
@click.argument("unit", nargs=1)
@click.argument("destination_unit", default=None, nargs=-1)
def convert(number, unit, destination_unit=None):

    def gen_fmt(ref):
        sig_figs = len(str(ref))
        print(str(ref))
        print(sig_figs)

        def fmt(new):
            exponent = int(f"{new:e}".split("e")[1])
            if exponent > 6 or exponent < -3:
                return f"{new:{sig_figs}e}"
            else: # a "normal" size number
                return f"{round(new, sig_figs-exponent)}"

        return fmt

    fmt = gen_fmt(number)

    if len(destination_unit):  # units provided
        if not wt.units.is_valid_conversion(unit, destination_unit):
            raise wt.exceptions.UnitsError(
                wt.units.get_valid_conversions(unit), destination_unit
            )
        new = wt.units.convert(number, unit, destination_unit)
        print(f"{fmt(new)} {destination_unit}")
    else:
        valid_units = wt.units.get_valid_conversions(unit)
        print(valid_units)
        for d_unit in valid_units:
            new = wt.units.convert(number, unit, d_unit)
            print(f"{fmt(new)} {d_unit}")


if __name__ == "__main__":
    cli()

