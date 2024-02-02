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

    shell = code.InteractiveConsole()
    _interact(shell, path)


@cli.command(name="scan", help="scan a directory and survey the wt5 objects found.")
@click.option(
    "--directory", "-d", default=None, help="Directory to scan.  Defaults to current directory."
)
@click.option("--no-recursion", is_flag=True, help="Turns recursive scan off.")
# TODO: formatting options (e.g. json)?
def scan(directory=None, no_recursion=False):
    import os, code
    from rich.live import Live
    from rich.table import Table

    if directory is None:
        directory = os.getcwd()

    table = Table(title=directory)
    table.add_column("", justify="right")  # index
    table.add_column("path", max_width=60, no_wrap=True)
    table.add_column("size (MB)", justify="center")
    table.add_column("created", max_width=30)
    table.add_column("name")
    table.add_column("shape")
    table.add_column("axes", max_width=50)
    table.add_column("variables")
    table.add_column("channels")

    update_title = lambda n: directory + f" ({n} wt5 file{'s' if n != 1 else None} found)"
    paths = []

    with Live(table) as live:
        for i, path in enumerate(wt.kit.glob_wt5s(directory, not no_recursion)):
            paths.append(path)
            desc = wt.kit.describe_wt5(path)
            desc["filesize"] = f"{os.path.getsize(path) / 1e6:.1f}"
            desc["path"] = str(path.relative_to(directory))
            row = [str(i)] + [
                str(desc[k])
                for k in ["path", "filesize", "created", "name", "shape", "axes", "nvars", "nchan"]
            ]
            table.title = update_title(i + 1)
            table.add_row(*row)
            live.update(table)

    # give option to interact
    def raise_sys_exit():
        raise SystemExit

    shell = code.InteractiveConsole(locals={"exit": raise_sys_exit, "quit": raise_sys_exit})

    while True:
        msg = shell.raw_input(
            " ".join(
                [
                    "Specify an index to load that entry.",
                    "Use 't' to rerender table.",
                    "Use no argument to exit.",
                ]
            )
        )
        if msg == "t":
            with Live(table) as live:
                pass
            continue
        try:
            valid = 0 <= int(msg) < len(paths)
        except ValueError:
            break
        if valid:
            _interact(shell, str(paths[int(msg)]))
            continue


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
