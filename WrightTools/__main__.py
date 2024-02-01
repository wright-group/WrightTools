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
    # def raise_sys_exit():
    #     raise SystemExit
    shell = code.InteractiveConsole() # locals={"exit": raise_sys_exit, "quit": raise_sys_exit})
    _interact(shell, path)


@cli.command(name="crawl", help="Crawl a directory and survey the wt5 objects found.")
@click.option("--directory", "-d", default=None, help="Directory to crawl.  Defaults to current directory.")
@click.option("--recursive", "-r", is_flag=True, help="Explore all levels of the directory")
@click.option("--format", "-f", default=None, help="Formatting keys (default only atm)")
# TODO: write output as an option; format kwarg?
def crawl(directory=None, recursive=False, format=None):
    import glob, os, code
    # from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    if directory is None:
        directory = os.getcwd()

    paths = glob.glob("**/*.wt5", root_dir=directory, recursive=recursive)

    def _parse_entry(i, relpath):
        size = os.path.getsize(os.path.join(directory, relpath)) / 1e6
        wt5 = wt.open(os.path.join(directory, relpath))
        name = wt5.natural_name
        try:
            created = wt5.created.human
        except:  # likely an old timestamp that cannot be parsed
            created = wt5.attrs["created"]

        if isinstance(wt5, wt.Data):
            shape = wt5.shape
            axes = wt5.axis_expressions
            nvars = len(wt5.variables)
            nchan = len(wt5.channels)
        elif isinstance(wt5, wt.Collection):
            shape = axes = nvars = nchan = "---"            
        return [
            str(i), relpath, f"{size:0.1f}", created, name, str(shape), str(axes), str(nvars), str(nchan) 
        ]

    table = Table(title=directory + f" ({len(paths)} wt5 file{'s' if len(paths) != 1 else None} found)")
    table.add_column("", justify="right")  # index
    table.add_column("path", max_width=60, no_wrap=True)
    table.add_column("size (MB)", justify="center")
    table.add_column("created", max_width=30)
    table.add_column("name")
    table.add_column("shape")
    table.add_column("axes", max_width=50)
    table.add_column("variables")
    table.add_column("channels")

    with Live(table) as live:
        for i, path in enumerate(paths):
            table.add_row(*_parse_entry(i, path))
            live.update(table)
    
    # give option to interact
    shell = code.InteractiveConsole()
    msg = shell.raw_input("Do you wish to load an entry? (specify an index to load, or don't and exit) ")
    try:
        valid = 0 < int(msg) + 1 < len(paths)
    except ValueError:
        print("invalid index")
        return    
    if valid:
        _interact(shell, os.path.join(directory, paths[int(msg)]))


def _interact(shell, path):
    lines = [
        "import WrightTools as wt",
        "import matplotlib.pyplot as plt",
        f"d = wt.open(r'{path}')",
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
