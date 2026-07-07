"""A module to load and process CPI-style Maccor battery cycler data."""

import csv
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger

from pyprobe.cyclers import column_maps
from pyprobe.cyclers.basecycler import BaseCycler


class MaccorCPITime(column_maps.ColumnMap):
    """Convert CPI test-time strings into numeric seconds."""

    def __init__(self, pyprobe_name: str, required_cycler_col: str) -> None:
        """Initialize the time mapping."""
        super().__init__(pyprobe_name, [required_cycler_col])
        self.required_cycler_col = required_cycler_col

    @property
    def expr(self) -> pl.Expr:
        """Return the expression to parse the test-time column."""
        return (
            self.get(self.required_cycler_col)
            .cast(pl.String)
            .map_elements(parse_test_time, return_dtype=pl.Float64)
            .alias(self.pyprobe_name)
        )


class MaccorCPIDateTime(column_maps.ColumnMap):
    """Convert CPI date strings into a datetime column."""

    def __init__(self, date_column: str) -> None:
        """Initialize the datetime mapping."""
        self.pyprobe_name = "Date"
        super().__init__(self.pyprobe_name, [date_column])
        self.date_column = date_column

    @property
    def expr(self) -> pl.Expr:
        """Return the expression to build the date column."""
        return self.get(self.date_column).map_elements(
            parse_datetime,
            return_dtype=pl.Datetime("us"),
        ).alias(self.pyprobe_name)


class MaccorCPICapacity(column_maps.ColumnMap):
    """Derive net capacity from the capacity column using the MD mode column."""

    def __init__(self, capacity_column: str, mode_column: str) -> None:
        """Initialize the CPI capacity importer."""
        super().__init__("Capacity [Ah]", [capacity_column, mode_column])
        self.capacity_column = capacity_column
        self.mode_column = mode_column

    @property
    def expr(self) -> pl.Expr:
        """Return the expression to compute net capacity from the mode column."""
        capacity = self.get(self.capacity_column).cast(pl.Float64)
        mode = (
            self.get(self.mode_column)
            .cast(pl.String)
            .str.to_uppercase()
            .str.strip_chars()
            .fill_null("")
        )
        mode_sign = (
            pl.when(mode.is_in(["C", "CHARGE"]))
            .then(1)
            .when(mode.is_in(["D", "DISCHARGE"]))
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
        )
        return (
            (capacity.diff().fill_null(0) * mode_sign)
            .cum_sum()
            + capacity.first().fill_null(0)
        ).alias(self.pyprobe_name)


class MaccorCPICurrent(column_maps.ColumnMap):
    """Derive signed current from the current column using the MD mode column."""

    def __init__(self, current_column: str, mode_column: str) -> None:
        """Initialize the CPI current importer."""
        super().__init__("Current [A]", [current_column, mode_column])
        self.current_column = current_column
        self.mode_column = mode_column

    @property
    def expr(self) -> pl.Expr:
        """Return the expression to compute signed current from the mode column."""
        current = self.get(self.current_column).cast(pl.Float64)
        mode = (
            self.get(self.mode_column)
            .cast(pl.String)
            .str.to_uppercase()
            .str.strip_chars()
            .fill_null("")
        )
        mode_sign = (
            pl.when(mode.is_in(["C", "CHARGE"]))
            .then(1)
            .when(mode.is_in(["D", "DISCHARGE"]))
            .then(-1)
            .otherwise(current.sign())
            .cast(pl.Int8)
        )
        return (current.abs() * mode_sign).alias(self.pyprobe_name)


class MaccorCPI(BaseCycler):
    """A class to load and process CPI-style Maccor battery cycler data."""

    column_importers: list[column_maps.ColumnMap] = [
        MaccorCPIDateTime("DPT Time"),
        MaccorCPITime("Time [s]", "Test Time (sec)"),
        column_maps.CastAndRenameMap("Step", "Step", pl.UInt64),
        MaccorCPICurrent("Current", "MD"),
        column_maps.CastAndRenameMap("Voltage [V]", "Voltage", pl.Float64),
        MaccorCPICapacity("Capacity", "MD"),
        column_maps.CastAndRenameMap("Temperature [C]", "Temp 1", pl.Float64),
    ]

    @staticmethod
    def read_file(
        filepath: str,
        header_row_index: int = 0,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Read a CPI-style Maccor export into a DataFrame.

        Args:
            filepath: The path to the file.
            header_row_index: The index of the header row.

        Returns:
            pl.DataFrame | pl.LazyFrame: The DataFrame.
        """
        _ = header_row_index
        path = Path(filepath)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            try:
                dataframe = pl.read_excel(
                    filepath,
                    sheet_name="Data",
                    read_options={"header_row": 2},
                ).lazy()
            except Exception:
                logger.warning(
                    "Failed to read sheet 'Data' from %s. Falling back to first sheet.",
                    filepath,
                )
                dataframe = pl.read_excel(
                    filepath,
                    read_options={"header_row": 2},
                ).lazy()
        elif path.suffix.lower() == ".csv":
            dataframe = pl.scan_csv(
                filepath,
                skip_rows=2,
                infer_schema=False,
                null_values=["", "N/A", "NaN"],
            )
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        normalized_columns = {col: normalize_header(col) for col in dataframe.collect_schema().names()}
        dataframe = dataframe.rename(normalized_columns)
        return _normalize_dataframe(dataframe)


def parse_test_time(value: object) -> float:
    """Parse CPI test time strings into seconds."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return 0.0

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return float(text)

    match = re.fullmatch(r"(?:(?P<days>\d+)d\s*)?(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)", text)
    if match:
        days = int(match.group("days") or 0)
        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes") or 0)
        seconds = int(match.group("seconds") or 0)
        return float(days * 24 * 3600 + hours * 3600 + minutes * 60 + seconds)

    match = re.fullmatch(r"(?:(?P<days>\d+)d\s*)?(?P<hours>\d+)h\s*(?P<minutes>\d+)m\s*(?P<seconds>\d+)s", text)
    if match:
        days = int(match.group("days") or 0)
        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes") or 0)
        seconds = int(match.group("seconds") or 0)
        return float(days * 24 * 3600 + hours * 3600 + minutes * 60 + seconds)

    return 0.0


def parse_datetime(value: object) -> datetime | None:
    """Parse CPI date strings into a datetime object."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    for date_format in (
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d-%b-%y %I:%M:%S %p",
        "%d-%m-%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue
    return None


def normalize_header(value: str) -> str:
    """Normalize header names to the standard Maccor names."""
    text = str(value).strip().replace("\ufeff", "")
    if text in {"Current (A)", "Current"}:
        return "Current"
    if text in {"Voltage (V)", "Voltage"}:
        return "Voltage"
    if text in {"EVTemp (C)", "EVTemp", "Temp 1", "Temp 1 (C)"}:
        return "Temp 1"
    if text in {"Capacity", "Capacity (Ah)", "Capacity(Ah)", "Capacity (AHr)", "Capacity(AHr)"}:
        return "Capacity"
    if text in {"Test Time (sec)", "Test Time", "Test Time (s)"}:
        return "Test Time (sec)"
    if text in {"DPT Time", "DateTime"}:
        return "DPT Time"
    return text


def _read_csv_rows(filepath: str) -> list[list[str]]:
    """Read a CSV file as a list of rows."""
    with open(filepath, encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle))


def _read_excel_rows(filepath: str) -> list[list[str]]:
    """Read an Excel file as a list of rows."""
    dataframe = pd.read_excel(filepath, header=None, dtype=str)
    dataframe = dataframe.fillna("")
    return dataframe.astype(str).values.tolist()


def _detect_header_row(rows: list[list[str]], preferred_row: int) -> int:
    """Find the row that contains the data-table headers."""
    if preferred_row >= 0 and preferred_row < len(rows):
        candidate = rows[preferred_row]
        if any(cell.strip() for cell in candidate) and any(
            keyword in " ".join(cell.strip().lower() for cell in candidate)
            for keyword in ["rec", "step", "test time", "current", "voltage", "temp"]
        ):
            return preferred_row

    for index, row in enumerate(rows):
        if not row:
            continue
        joined = " ".join(cell.strip().lower() for cell in row if cell)
        if any(keyword in joined for keyword in ["rec", "step", "test time", "current", "voltage", "temp"]):
            return index
    return 0


def _pad_row(row: list[str], width: int) -> list[str]:
    """Pad a row to the expected width."""
    if len(row) >= width:
        return row[:width]
    return row + [""] * (width - len(row))


def _normalize_dataframe(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Normalize the parsed CPI data to the standard Maccor schema."""
    for column_name in ["Current", "Voltage", "Capacity", "Temp 1"]:
        if column_name in dataframe.columns:
            dataframe = dataframe.with_columns(
                pl.col(column_name)
                .cast(pl.String)
                .str.replace_all(",", "")
                .str.strip_chars()
                .cast(pl.Float64, strict=False)
                .alias(column_name)
            )

    if "Test Time (sec)" in dataframe.columns:
        dataframe = dataframe.with_columns(
            pl.col("Test Time (sec)")
            .cast(pl.String)
            .map_elements(parse_test_time, return_dtype=pl.Float64)
            .alias("Test Time (sec)")
        )

    if "DPT Time" in dataframe.columns:
        dataframe = dataframe.with_columns(
            pl.col("DPT Time")
            .cast(pl.String)
            .map_elements(parse_datetime, return_dtype=pl.Datetime("us"))
            .alias("DPT Time")
        )

    if "Step" in dataframe.columns:
        dataframe = dataframe.with_columns(pl.col("Step").cast(pl.UInt64, strict=False))

    return dataframe
