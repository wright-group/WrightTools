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
@click.option(
    "--internal_path", default="/", help="specify a path internal to the file.  Defaults to root"
)
@click.option("--depth", "-d", "-L", type=int, default=9, help="Depth to print.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Print a more detailed tree.")
def tree(path, internal_path, depth=9, verbose=False):
    # open the object
    obj = wt.open(path)[internal_path]

    if isinstance(obj, wt.Data):
        obj.print_tree(verbose=verbose)
    else:
        obj.print_tree(verbose=verbose, depth=depth)


@cli.command(name="load", help="Open a python cli with the object pre-loaded.")
@click.argument("path")
def load(path):
    import code

    d = wt.open(d)
    ...


@cli.command(name="crawl", help="Crawl a directory and survey the wt5 objects found.")
@click.option("--directory", "-d", default=None, help="Directory to crawl.  Defaults to current directory.")
@click.option("--recursive", "-r", is_flag=True, help="Explore all levels of the directory")
@click.option(
    "--pausing",
    "-p",
    is_flag=True,
    help="pause at each file readout. Interaction with data is possible.",
)
@click.option("--format", "-f", default=None, help="Formatting keys (default only atm)")
# TODO: write output as an option
def crawl(directory=None, recursive=False, pausing=False, format=None):
    import glob, os

    if directory is None:
        directory = os.getcwd()

    if pausing:
        import code

        def raise_sys_exit():
            raise SystemExit

        shell = code.InteractiveConsole(locals={"exit": raise_sys_exit, "quit": raise_sys_exit})

    paths = glob.glob("**/*.wt5", root_dir=directory, recursive=recursive)
    print(f"{len(paths)} wt5 file{'s' if len(paths) != 1 else None} found in {directory}")

    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title=directory)
    table.add_column("", justify="right")  # index
    table.add_column("path", max_width=60)
    table.add_column("size (MB)", justify="center")
    table.add_column("created", max_width=30)
    table.add_column("name")
    table.add_column("shape")
    table.add_column("axes")
    table.add_column("variables")
    table.add_column("channels")


    for i, pathname in enumerate(paths):
        path = os.path.join(directory, pathname)
        d = wt.open(path)
        infos = [str(i), pathname, str(int(os.path.getsize(path)/1e6)), *_format_entry(d)]
        table.add_row(*infos)

        if pausing:
            msg = shell.raw_input("Interact ([n=no]/y=yes/q=quit)? ")
            if msg.lower() in ["q", "quit"]:
                break
            elif msg.lower() in ["y", "yes"]:
                _interact(shell, os.path.join(directory, pathname), isinstance(d, wt.Collection))
                print("-" * 100)
            else:
                continue
        d.close()


from typing import Tuple, Optional
class TableEntry:
    created: str = ""
    name: str = ""
    shape: Optional(Tuple(int))
    axes: Optional(Tuple(str))
    nvars: Optional(int)
    nchan: Optional(int)
    def __init__(self, wt5):
        self.name = wt5.natural_name
        self.created = wt5.attrs["created"].date
        if isinstance(wt5, wt.Data):
            self.shape = wt5.shape
            self.axes = wt5.axis_expressions
            self.nvars = len(wt5.variables)
            self.nchan = len(wt5.channels)

    




def _format_entry(wt5, format=None):
    if isinstance(wt5, wt.Data):
        values = [f"{wt5.attrs['created']}", wt5.natural_name, f"{wt5.shape}", f"{wt5.axis_expressions}", str(len(wt5.variables)), str(len(wt5.channels))]
    elif isinstance(wt5, wt.Collection):
        values = ["---"] + [wt5.natural_name] + ["---"] * 4
    else:
        values = ["---"] + ["<unknown>"] + ["---"] * 4
    return values


def _interact(shell, path, is_collection):
    lines = [
        "import WrightTools as wt",
        "import matplotlib.pyplot as plt",
        f"{'c' if is_collection else 'd'} = wt.open(r'{path}')",
    ]

    [shell.push(line) for line in lines]
    banner = "--- INTERACTING --- (to continue, call exit() or quit())\n"
    banner += "\n".join([">>> " + line for line in lines])

    try:
        shell.interact(banner=banner)
    except SystemExit:
        pass


@cli.command(name="convert")
@click.argument("number", type=float, nargs=1)
@click.argument("unit", nargs=1)
@click.argument("destination_unit", default=None, nargs=-1)
def convert(number, unit, destination_unit=None):
    """Convert numbers to different units."""

    if int(number) == number:
        number = int(number)
    sig_figs = len(str(number))
    sig_figs -= 1 if "." in str(number) else 0

    def fmt(new):
        exponent = int(f"{new:e}".split("e")[1])
        if exponent > 6 or exponent < -3:
            return f"{new:{sig_figs}e}"
        else:  # if a "normal" size number
            if sig_figs - exponent <= 0:
                return f"{int(round(new, sig_figs-exponent))}"
            else:
                return f"{round(new, sig_figs-exponent)}"

    if len(destination_unit):  # units provided
        destination_unit = destination_unit[0]
        if not wt.units.is_valid_conversion(unit, destination_unit):
            raise wt.exceptions.UnitsError(wt.units.get_valid_conversions(unit), destination_unit)
        new = wt.units.convert(number, unit, destination_unit)
        print(f"{number} {unit} = {fmt(new)} {destination_unit}")
    else:
        valid_units = wt.units.get_valid_conversions(unit)
        for d_unit in valid_units:
            new = wt.units.convert(number, unit, d_unit)
            print(f"{fmt(new)} {d_unit}")


if __name__ == "__main__":
    cli()
