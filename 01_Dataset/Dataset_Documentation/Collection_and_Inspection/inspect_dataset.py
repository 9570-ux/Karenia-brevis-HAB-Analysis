"""Inspect an original CSV without writing to it; uses only the standard library."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inspect(path, encoding="utf-8-sig", delimiter=","):
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    reader = csv.reader(io.StringIO(original.decode(encoding)), delimiter=delimiter)
    columns = next(reader, None)
    if not columns:
        raise ValueError("Missing CSV header")
    labels = [f"{name or '(unnamed)'} [column {i}]" for i, name in enumerate(columns, 1)]
    stats = [{"blank_values": 0, "nonblank_values": 0, "numeric_values": 0,
              "integer_values": 0, "nonfinite_values": 0, "minimum": None,
              "maximum": None, "unique": set()} for _ in columns]
    lookup = {name.upper(): i for i, name in enumerate(columns) if name}
    quality_no_data = {name: 0 for name in ("CELLCOUNT", "SALINITY", "WATER_TEMP", "WIND_DIR", "WIND_SPEED")
                       if name + "_QA" in lookup}
    date_min = date_max = None
    time_placeholders = {"12:00": 0, "0:00": 0, "00:00": 0}
    rows = 0
    for line, values in enumerate(reader, start=2):
        if len(values) != len(columns):
            raise ValueError(f"CSV line {line}: expected {len(columns)} fields, got {len(values)}")
        rows += 1
        for field in quality_no_data:
            if values[lookup[field + "_QA"]].strip() == "9":
                quality_no_data[field] += 1
        if "SAMPLE_DATE" in lookup:
            date = values[lookup["SAMPLE_DATE"]].strip()
            if date:
                date_min = date if date_min is None else min(date_min, date)
                date_max = date if date_max is None else max(date_max, date)
        if "SAMPLE_TIME" in lookup:
            sample_time = values[lookup["SAMPLE_TIME"]].strip()
            if sample_time in time_placeholders:
                time_placeholders[sample_time] += 1
        for value, stat in zip(values, stats):
            value = value.strip()  # Analysis only; original bytes are never rewritten.
            if not value:
                stat["blank_values"] += 1
                continue
            stat["nonblank_values"] += 1
            if stat["unique"] is not None:
                stat["unique"].add(value)
                if len(stat["unique"]) > 50:
                    stat["unique"] = None
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                stat["nonfinite_values"] += 1
                continue
            stat["numeric_values"] += 1
            stat["integer_values"] += int(number.is_integer())
            stat["minimum"] = number if stat["minimum"] is None else min(stat["minimum"], number)
            stat["maximum"] = number if stat["maximum"] is None else max(stat["maximum"], number)
    result = {}
    for column, stat in zip(labels, stats):
        count = stat["nonblank_values"]
        if not count:
            dtype = "empty"
        elif stat["numeric_values"] == count:
            dtype = "integer" if stat["integer_values"] == count else "float"
        elif stat["numeric_values"] or stat["nonfinite_values"]:
            dtype = "mixed text/numeric"
        else:
            dtype = "text (dates are not inferred)"
        unique = stat.pop("unique")
        stat["inferred_type"] = dtype
        if dtype not in ("integer", "float"):
            stat["minimum"] = stat["maximum"] = None
        stat["unique_values_if_at_most_50"] = sorted(unique) if unique is not None else "more than 50"
        result[column] = stat
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Raw file changed during inspection")
    return {"file": str(path.resolve()), "size_bytes": len(original), "sha256": digest,
            "row_count": rows, "column_names": columns, "columns": result,
            "sample_date_lexical_min": date_min, "sample_date_lexical_max": date_max,
            "qa_9_no_data_counts": quality_no_data,
            "sample_time_missing_placeholders": time_placeholders,
            "missing_value_policy": "Only empty/whitespace cells counted as blank. Scientific sentinels require verified metadata; none assumed.",
            "range_policy": "Ranges include all finite recorded values, without excluding QA flags; no cleaning or imputation.",
            "raw_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--delimiter", default=",")
    args = parser.parse_args()
    path = args.path
    if path is None:
        files = sorted((ROOT / "data" / "raw").glob("*.csv"))
        if len(files) != 1:
            parser.error("Expected one original CSV in data/raw; provide an explicit path otherwise. No data is generated.")
        path = files[0]
    print(json.dumps(inspect(path, args.encoding, args.delimiter), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
