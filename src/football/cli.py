import typer

from football.backtest import commands as backtest_commands
from football.collect import commands as collect_commands
from football.features import commands as features_commands
from football.ingest import commands as ingest_commands
from football.opinions import commands as opinions_commands
from football.predict import commands as predict_commands
from football.report import commands as report_commands
from football.validate import commands as validate_commands

app = typer.Typer(no_args_is_help=True, add_completion=False)

for module in (
    ingest_commands,
    validate_commands,
    collect_commands,
    features_commands,
    backtest_commands,
    predict_commands,
    opinions_commands,
    report_commands,
):
    module.register(app)


def main() -> None:
    app()
