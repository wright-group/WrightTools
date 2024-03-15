# --- import --------------------------------------------------------------------------------------


import click
import WrightTools as wt


# --- define --------------------------------------------------------------------------------------


@click.group()
@click.version_option(wt.__version__, prog_name="WrightTools")
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


@cli.command(name="glob", help="Find paths to all wt5 files within a directory.")
@click.option(
    "--directory", "-d", default=None, help="Directory to scan.  Defaults to current directory."
)
@click.option("--no-recursion", is_flag=True, help="Turns recursive scan off.")
def glob(directory=None, no_recursion=False):
    for path in wt.kit.glob_wt5s(directory, not no_recursion):
        print(str(path))


@cli.command(name="explore", help="Scan a directory and survey the wt5 objects found.")
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
    table.add_column("path", max_width=60, no_wrap=True, style="blue")
    table.add_column("size (MB)", justify="center", style="blue")
    table.add_column("created", max_width=30, style="blue")
    table.add_column("name", style="blue")
    table.add_column("shape", style="blue")
    table.add_column("axes", max_width=50, style="blue")
    table.add_column("variables", style="blue")
    table.add_column("channels", style="blue")

    update_title = lambda n: directory + f" ({n} wt5 file{'s' if n != 1 else None} found)"
    paths = []

    with Live(table) as live:
        for i, path in enumerate(wt.kit.glob_wt5s(directory, not no_recursion)):
            desc = wt.kit.describe_wt5(path)
            desc["filesize"] = f"{os.path.getsize(path) / 1e6:.1f}"
            path = path.relative_to(directory)
            paths.append(path)
            desc["path"] = (
                f"[link={path.parent}]{path.parent}[/link]" + r"\\"
                if str(path.parent) != "."
                else ""
            )
            desc["path"] += f"[bold]{path.name}[/bold]"
            # desc["path"] = f"[link={str(path)}]{path}[/link]"
            row = [f"{i}"] + [
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
                    "Use `t` to rerender table.",
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
            print("interacting...")
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


if __name__ == "__main__":
    cli()
