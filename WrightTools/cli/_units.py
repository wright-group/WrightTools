# --- import --------------------------------------------------------------------------------------


import click
from WrightTools import __version__ as __wt_version__
from WrightTools.units import is_valid_conversion, get_valid_conversions, convert
from WrightTools.exceptions import UnitsError


# --- define --------------------------------------------------------------------------------------


@click.command(name="convert", help="convert numbers to different units.")
@click.version_option(__wt_version__, prog_name="WrightTools")
@click.argument("number", type=float, nargs=1)
@click.argument("unit", nargs=1)
@click.argument("destination_unit", default=None, nargs=-1)
def cli(number, unit, destination_unit=None):
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
        if not is_valid_conversion(unit, destination_unit):
            raise UnitsError(get_valid_conversions(unit), destination_unit)
        new = convert(number, unit, destination_unit)
        print(f"{number} {unit} = {fmt(new)} {destination_unit}")
    else:
        valid_units = get_valid_conversions(unit)
        for d_unit in valid_units:
            new = convert(number, unit, d_unit)
            print(f"{fmt(new)} {d_unit}")


if __name__ == "__main__":
    cli()
