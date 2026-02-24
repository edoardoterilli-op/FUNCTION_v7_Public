import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
from datetime import datetime
from pathlib import Path
import tempfile
import os
import io

# ==========================================
# CONFIGURAZIONE PAGINA
# ==========================================
st.set_page_config(page_title="Power Query Web UI", page_icon="⚡", layout="wide")

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", message="Columns .* have mixed types.*")
warnings.filterwarnings("ignore", message=".*low_memory.*")

# ==========================================
# INIZIALIZZAZIONE SESSION STATE
# ==========================================
if "STATE" not in st.session_state:
    st.session_state.STATE = {"files": {}, "order": []}
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if "multi_conds" not in st.session_state:
    st.session_state.multi_conds = []
if "run_conds" not in st.session_state:
    st.session_state.run_conds = []
if "calc_ops_list" not in st.session_state:
    st.session_state.calc_ops_list = []

# ==========================================
# COSTANTI UI: OPERATORI FILTRI (FIX NameError)
#   - value = codice usato dal motore (_filter_mask_single)
#   - label = testo mostrato in UI
# ==========================================
OPS_NUMBER = [
    ("gt", ">"),
    ("gte", "≥"),
    ("lt", "<"),
    ("lte", "≤"),
    ("eq", "="),
    ("neq", "≠"),
    ("between", "Tra (incluso)"),
]
OPS_DATE = [
    ("after", "Dopo"),
    ("before", "Prima"),
    ("eq", "Uguale"),
    ("neq", "Diversa"),
    ("between", "Tra (incluso)"),
]
OPS_TEXT = [
    ("eq", "Uguale"),
    ("neq", "Diverso"),
    ("contains", "Contiene"),
    ("not_contains", "Non contiene"),
    ("startswith", "Inizia con"),
    ("endswith", "Finisce con"),
    ("is_empty", "Vuoto"),
    ("not_empty", "Non vuoto"),
]
OPS_LABEL_BY_VALUE = {v: lab for (v, lab) in OPS_NUMBER + OPS_DATE + OPS_TEXT}

# ==========================================
# MOTORE LOGICO (Identico allo script originale)
# ==========================================
def _try_import_pyarrow():
    try:
        import pyarrow  # noqa: F401
        import pyarrow.csv  # noqa: F401
        return True
    except Exception:
        return False

HAS_PYARROW = _try_import_pyarrow()
STRING_DTYPE = "string[pyarrow]" if HAS_PYARROW else "string"

_BAD = re.compile(r'[\\/:*?"<>|]+')
def sanitize_name(name: str, fallback="output") -> str:
    name = (name or "").strip()
    name = _BAD.sub("_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("_")
    return name if name else fallback

def detect_csv_sep_from_sample(text: str) -> str:
    candidates = {";": text.count(";"), ",": text.count(","), "\t": text.count("\t"), "|": text.count("|")}
    sep = max(candidates, key=candidates.get)
    return sep if candidates[sep] > 0 else ";"

DATE_INPUT_MAP = {
    "Auto (prova a riconoscere)": {"kind": "auto"},
    "YYYY-MM-DD (es. 2024-01-31)": {"kind": "fmt", "fmt": "%Y-%m-%d"},
    "DD/MM/YYYY (es. 31/01/2024)": {"kind": "fmt", "fmt": "%d/%m/%Y"},
    "MM/DD/YYYY (es. 01/31/2024)": {"kind": "fmt", "fmt": "%m/%d/%Y"},
    "YYYY/MM/DD (es. 2024/01/31)": {"kind": "fmt", "fmt": "%Y/%m/%d"},
    "DD-MM-YYYY (es. 31-01-2024)": {"kind": "fmt", "fmt": "%d-%m-%Y"},
    "Datetime auto": {"kind": "auto_dt"},
    "YYYY-MM-DD HH:MM:SS": {"kind": "fmt", "fmt": "%Y-%m-%d %H:%M:%S"},
    "DD/MM/YYYY HH:MM:SS": {"kind": "fmt", "fmt": "%d/%m/%Y %H:%M:%S"},
}

def _robust_read_csv(path: Path, sep: str, enc: str, engine: str, chunksize: int, import_mode: str = "string", usecols=None):
    common = dict(sep=sep, encoding=enc, engine=engine, chunksize=chunksize, on_bad_lines="skip", usecols=usecols)
    if import_mode == "string":
        return pd.read_csv(path, dtype=STRING_DTYPE, keep_default_na=False, na_filter=False, **common)
    return pd.read_csv(path, low_memory=False, **common)

def read_csv_fast(path: Path, progress_cb=None, encoding_try=("utf-8", "utf-8-sig", "latin1"), import_mode: str = "string", usecols=None) -> pd.DataFrame:
    total = max(1, path.stat().st_size)
    with open(path, "rb") as f:
        sample = f.read(65536)
    try:
        sample_txt = sample.decode("utf-8", errors="ignore")
    except Exception:
        sample_txt = sample.decode("latin1", errors="ignore")
    sep = detect_csv_sep_from_sample(sample_txt)

    last_exc = None
    for enc in encoding_try:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reader = _robust_read_csv(path, sep=sep, enc=enc, engine="c", chunksize=300_000, import_mode=import_mode, usecols=usecols)
                chunks, approx = [], 0
                for ch in reader:
                    chunks.append(ch)
                    approx = min(total, approx + max(1, total // 180))
                    if progress_cb:
                        progress_cb(approx / total)
                if progress_cb:
                    progress_cb(1.0)
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            return df.fillna("") if import_mode == "string" else df
        except Exception as e:
            last_exc = e

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reader = _robust_read_csv(path, sep=sep, enc=enc, engine="python", chunksize=200_000, import_mode=import_mode, usecols=usecols)
                chunks, approx = [], 0
                for ch in reader:
                    chunks.append(ch)
                    approx = min(total, approx + max(1, total // 160))
                    if progress_cb:
                        progress_cb(approx / total)
                if progress_cb:
                    progress_cb(1.0)
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            return df.fillna("") if import_mode == "string" else df
        except Exception as e:
            last_exc = e

    raise last_exc

def read_file(path: Path, progress_cb=None, import_mode: str = "string", usecols=None) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return read_csv_fast(path, progress_cb=progress_cb, import_mode=import_mode, usecols=usecols)
    if suf in (".xlsx", ".xls", ".xlsb"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_excel(path, engine=None)
        return df.astype(STRING_DTYPE).fillna("") if import_mode == "string" else df
    raise ValueError(f"Formato non supportato: {suf}")

def add_loaded_df(name: str, df: pd.DataFrame, path: Path | None = None, import_mode: str = "string", usecols=None, meta_extra: dict | None = None):
    fname = name
    base = fname
    i = 2
    while fname in st.session_state.STATE["files"]:
        fname = f"{base} ({i})"
        i += 1

    meta = {"import_mode": import_mode, "usecols": list(usecols) if usecols else None, "loaded_at": datetime.now().isoformat()}
    if meta_extra:
        meta.update(meta_extra)

    st.session_state.STATE["files"][fname] = {
        "df_base": df,
        "path": str(path) if path else None,
        "steps": [],
        "smart_cache": {"step_idx": -1, "df": df},
        "meta": meta,
    }
    st.session_state.STATE["order"].append(fname)
    return fname

def clear_cache_from(fname: str, idx: int):
    rec = st.session_state.STATE["files"][fname]
    if "smart_cache" in rec and rec["smart_cache"]["step_idx"] >= idx:
        rec["smart_cache"] = {"step_idx": -1, "df": rec["df_base"]}

def declared_type_for_col(file_name: str, col: str) -> str | None:
    steps = st.session_state.STATE["files"][file_name]["steps"]
    declared = None
    for stp in steps:
        if stp.get("type") == "format":
            fmts = stp.get("params", {}).get("formats", {})
            if col in fmts and isinstance(fmts[col], dict):
                declared = fmts[col].get("type")
    return declared

def _parse_number_series(s: pd.Series, decimal_sep: str | None = None, thousands_sep: str | None = None) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s.dtype):
        return s.astype("Float64")
    x = s.astype(STRING_DTYPE).str.strip()
    if thousands_sep:
        x = x.str.replace(thousands_sep, "", regex=False)
    if decimal_sep and decimal_sep != ".":
        x = x.str.replace(decimal_sep, ".", regex=False)
    x = x.str.replace("€", "", regex=False).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(x.replace("", pd.NA), errors="coerce").astype("Float64")

def _apply_format(file_name: str, df: pd.DataFrame, formats: dict) -> pd.DataFrame:
    out = df.copy()
    for col, spec in formats.items():
        if col not in out.columns:
            continue
        t = spec.get("type", "auto")
        if t == "text":
            out[col] = out[col].astype(STRING_DTYPE).fillna("")
            continue
        if t in ("number", "currency", "percent"):
            dec = spec.get("decimal_sep", None)
            thou = spec.get("thousands_sep", None)
            if thou == "__NONE__":
                thou = None
            out[col] = _parse_number_series(out[col], decimal_sep=dec, thousands_sep=thou)
            continue
        if t in ("date", "datetime"):
            inp = spec.get("date_input", "Auto (prova a riconoscere)")
            mode = DATE_INPUT_MAP.get(inp, {"kind": "auto"}).get("kind", "auto")
            fmt = DATE_INPUT_MAP.get(inp, {}).get("fmt", None)
            if mode == "fmt" and fmt:
                out[col] = pd.to_datetime(out[col], format=fmt, errors="coerce")
            else:
                out[col] = pd.to_datetime(out[col], errors="coerce")
    return out

def _filter_mask_single(df: pd.DataFrame, col: str, kind: str, op: str, v1, v2):
    if col not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    s = df[col]

    if kind == "date":
        dt = pd.to_datetime(s, errors="coerce")
        d1 = pd.to_datetime(v1, errors="coerce")
        d2 = pd.to_datetime(v2, errors="coerce") if v2 else None
        if op == "after":
            return dt > d1
        if op == "before":
            return dt < d1
        if op == "eq":
            return dt == d1
        if op == "neq":
            return dt != d1
        if op == "between" and d2 is not None:
            lo, hi = (d1, d2) if d1 <= d2 else (d2, d1)
            return (dt >= lo) & (dt <= hi)
        return pd.Series([True] * len(df), index=df.index)

    if kind == "number":
        num = _parse_number_series(s)
        n1 = pd.to_numeric(pd.Series([v1]), errors="coerce").iloc[0]
        n2 = pd.to_numeric(pd.Series([v2]), errors="coerce").iloc[0] if v2 else None
        if op == "gt":
            return num > n1
        if op == "gte":
            return num >= n1
        if op == "lt":
            return num < n1
        if op == "lte":
            return num <= n1
        if op == "eq":
            return num == n1
        if op == "neq":
            return num != n1
        if op == "between" and n2 is not None:
            lo, hi = (n1, n2) if n1 <= n2 else (n2, n1)
            return (num >= lo) & (num <= hi)
        return pd.Series([True] * len(df), index=df.index)

    txt = s.astype(STRING_DTYPE).fillna("")
    v1s = "" if v1 is None else str(v1)

    if op == "eq":
        return txt == v1s
    if op == "neq":
        return txt != v1s
    if op == "contains":
        return txt.str.contains(re.escape(v1s), case=False, na=False)
    if op == "not_contains":
        return ~txt.str.contains(re.escape(v1s), case=False, na=False)
    if op == "startswith":
        return txt.str.startswith(v1s, na=False)
    if op in ("finisce con", "endswith"):
        return txt.str.endswith(v1s, na=False)
    if op == "is_empty":
        return txt.str.len().fillna(0) == 0
    if op == "not_empty":
        return txt.str.len().fillna(0) > 0

    return pd.Series([True] * len(df), index=df.index)

def _apply_filter(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    return df.loc[_filter_mask_single(df, params["col"], params["kind"], params["op"], params["v1"], params.get("v2"))].copy()

def _apply_filter_multi(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    logic = params.get("logic", "AND").upper()
    conds = params.get("conds", [])
    if not conds:
        return df
    masks = [_filter_mask_single(df, c["col"], c["kind"], c["op"], c["v1"], c.get("v2")) for c in conds]
    m = masks[0]
    for mm in masks[1:]:
        m = (m | mm) if logic == "OR" else (m & mm)
    return df.loc[m].copy()

def _apply_dedupe(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    if params["col"] not in df.columns:
        return df
    return df.drop_duplicates(subset=[params["col"]], keep=params.get("keep", "first")).copy()

def _safe_merge(left: pd.DataFrame, right: pd.DataFrame, how: str, left_on: str, right_on: str, collision: str = "suffix", suffix_left: str = "_A", suffix_right: str = "_B"):
    dup = (set(left.columns) & set(right.columns)) - {left_on, right_on}
    if not dup:
        return pd.merge(left, right, how=how, left_on=left_on, right_on=right_on, copy=False)

    if collision == "keep_left":
        return pd.merge(left, right.drop(columns=list(dup), errors="ignore"), how=how, left_on=left_on, right_on=right_on, copy=False)
    if collision == "keep_right":
        return pd.merge(left.drop(columns=list(dup), errors="ignore"), right, how=how, left_on=left_on, right_on=right_on, copy=False)

    return pd.merge(
        left,
        right.rename(columns={c: f"{c}{suffix_right}" for c in dup}),
        how=how,
        left_on=left_on,
        right_on=right_on,
        copy=False,
    )

def ifs_chunked_rowmatch(file_target: str, df_target: pd.DataFrame, lookup_file: str, lookup_df: pd.DataFrame, params: dict, progress_cb=None, chunk_rows: int = 300_000) -> pd.DataFrame:
    fun = params["metric"]
    match_a_col = params["match_a_col"]
    match_b_col = params["match_b_col"]
    val_col = params.get("value_col", "")

    if match_a_col not in df_target.columns:
        raise ValueError(f"Colonna A '{match_a_col}' non trovata.")
    if match_b_col not in lookup_df.columns:
        raise ValueError(f"Colonna B '{match_b_col}' non trovata.")

    b = lookup_df
    for c in params.get("conds", []):
        b = _apply_filter(b, c)

    bkey = b[match_b_col].astype(STRING_DTYPE).fillna("")

    if fun == "countifs":
        agg_map = bkey.value_counts(dropna=False)
    else:
        if declared_type_for_col(lookup_file, val_col) == "text":
            raise ValueError("Colonna valore formattata come Testo.")
        if val_col not in b.columns:
            raise ValueError("Colonna valore non trovata in B.")
        grp = _parse_number_series(b[val_col]).groupby(bkey, dropna=False)
        agg_map = grp.sum(min_count=1) if fun == "sumifs" else grp.mean()

    n = len(df_target)
    out = df_target.copy()
    outcol = params["outcol"]
    if outcol not in out.columns:
        out[outcol] = np.nan
    col_idx = out.columns.get_loc(outcol)

    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        tkey = out.iloc[start:end][match_a_col].astype(STRING_DTYPE).fillna("")
        res = tkey.map(agg_map)
        if fun == "countifs":
            res = res.fillna(0).astype("int64")
        elif fun == "sumifs":
            res = res.fillna(0)
        out.iloc[start:end, col_idx] = res.to_numpy()
        if progress_cb:
            progress_cb(end / max(1, n))
    return out

def _apply_calc_cols(file_name: str, df: pd.DataFrame, params: dict) -> pd.DataFrame:
    outcol = params["outcol"]
    base_col = params["col_left"] if "col_left" in params else params["base_col"]
    ops = [{"op": params["op"], "col": params["col_right"]}] if "col_left" in params else params.get("ops", [])

    if base_col not in df.columns:
        raise ValueError(f"Colonna base '{base_col}' non trovata.")
    if declared_type_for_col(file_name, base_col) == "text":
        raise ValueError("Operazioni non consentite su Testo.")

    res = _parse_number_series(df[base_col]).fillna(0.0).astype("Float64")
    for step in ops:
        op, c = step["op"], step["col"]
        if c not in df.columns:
            raise ValueError(f"Colonna '{c}' non trovata.")
        nxt = _parse_number_series(df[c]).fillna(0.0).astype("Float64")
        if op == "add":
            res = res + nxt
        elif op == "sub":
            res = res - nxt
        elif op == "mul":
            res = res * nxt
        elif op == "div":
            res = res / nxt

    res_obj = res.astype("object")
    err_mask = res_obj.isin([np.inf, -np.inf]) | pd.isna(res_obj)
    res_obj.loc[err_mask] = "ERRORE: Div/0"

    out = df.copy()
    out[outcol] = res_obj
    return out

_recompute_stack = set()
def apply_step(step: dict, df_in: pd.DataFrame, get_df_fn, current_file: str | None = None, progress_cb=None):
    stype, p = step["type"], step["params"]
    if stype == "format":
        return _apply_format(current_file, df_in, p.get("formats", {}))
    if stype == "filter":
        return _apply_filter(df_in, p)
    if stype == "filter_multi":
        return _apply_filter_multi(df_in, p)
    if stype == "dedupe":
        return _apply_dedupe(df_in, p)
    if stype == "calc":
        return _apply_calc_cols(current_file, df_in, p)
    if stype == "merge":
        right_file = p["right_file"]
        key = (current_file, right_file)
        if key in _recompute_stack:
            raise ValueError("Loop rilevato nel merge.")
        _recompute_stack.add(key)
        try:
            right_df = get_df_fn(right_file)
        finally:
            _recompute_stack.discard(key)
        return _safe_merge(df_in, right_df, how=p["how"], left_on=p["left_on"], right_on=p["right_on"], collision=p.get("collision", "suffix"))
    if stype == "ifs":
        lookup_file = p["lookup_file"]
        key = (current_file, lookup_file)
        if key in _recompute_stack:
            raise ValueError("Loop rilevato in IFS.")
        _recompute_stack.add(key)
        try:
            lookup_df = get_df_fn(lookup_file)
        finally:
            _recompute_stack.discard(key)
        return ifs_chunked_rowmatch(current_file, df_in, lookup_file, lookup_df, p, progress_cb=progress_cb, chunk_rows=300_000)
    raise ValueError(f"Step non supportato: {stype}")

def get_df(fname: str, progress_cb=None) -> pd.DataFrame:
    rec = st.session_state.STATE["files"][fname]
    base, steps = rec["df_base"], rec["steps"]
    if "smart_cache" not in rec:
        rec["smart_cache"] = {"step_idx": -1, "df": base}
    cache_idx, df = rec["smart_cache"]["step_idx"], rec["smart_cache"]["df"]

    if cache_idx >= len(steps):
        cache_idx, df = -1, base

    start_idx = cache_idx + 1
    if start_idx < len(steps):
        for i in range(start_idx, len(steps)):
            st_dict = steps[i]
            df = apply_step(st_dict, df, get_df, current_file=fname, progress_cb=progress_cb)
            st_dict["_row_count"] = len(df)
        rec["smart_cache"] = {"step_idx": len(steps) - 1, "df": df}
    return df

def compute_metric(file_name: str, df: pd.DataFrame, col: str, metric: str, q: float | None = None):
    if col not in df.columns:
        raise ValueError("Colonna non trovata.")
    if metric in ("sum", "average", "median", "quartile", "percentile") and declared_type_for_col(file_name, col) == "text":
        raise ValueError("Metrica numerica non consentita su Testo.")
    s = df[col]
    if metric == "counta":
        return int(s.notna().sum())
    if metric == "count":
        return int(pd.to_numeric(s, errors="coerce").notna().sum())
    vals = _parse_number_series(s)
    if metric == "sum":
        return float(vals.sum(skipna=True))
    if metric == "average":
        return float(vals.mean(skipna=True))
    if metric == "median":
        return float(vals.median(skipna=True))
    if metric == "quartile":
        return float(vals.quantile(q or 0.25))
    if metric == "percentile":
        return float(vals.quantile(q or 0.50))
    raise ValueError("Metrica non supportata.")

# ==========================================
# CREAZIONE NUOVO FILE DA MERGE (FIX: funzione mancante)
# ==========================================
def create_new_file_from_merge(new_name: str, file_A: str, file_B: str, how: str, left_on: str, right_on: str, collision: str):
    dfA = get_df(file_A)
    dfB = get_df(file_B)
    merged = _safe_merge(dfA, dfB, how=how, left_on=left_on, right_on=right_on, collision=collision)
    add_loaded_df(new_name, merged, path=None, import_mode="smart", meta_extra={"created_from": "merge", "A": file_A, "B": file_B})

# ==========================================
# UI HELPER: Etichette Step
# ==========================================
def step_label(st_dict: dict) -> str:
    t, p = st_dict["type"], st_dict["params"]
    if t == "format":
        return f"Formato: {', '.join([f'{c}:{v['type']}' for c, v in p.get('formats', {}).items()])}"
    if t == "filter":
        if p["op"] == "between":
            return f"Filtro: {p['col']} tra {p['v1']} e {p.get('v2')}"
        return f"Filtro: {p['col']} {p['op']} {p['v1']}"
    if t == "filter_multi":
        return f"Filtro multiplo ({p.get('logic','AND')}) con {len(p.get('conds',[]))} condiz."
    if t == "merge":
        return f"MERGE ({p['how'].upper()}) con '{p['right_file']}' su '{p['left_on']}'='{p['right_on']}'"
    if t == "ifs":
        return f"{p['metric'].upper()} → '{p['outcol']}' (B={p['lookup_file']})"
    if t == "calc":
        if "col_left" in p:
            return f"CALC: {p['outcol']} = [{p['col_left']}] {p['op']} [{p['col_right']}]"
        seq = [f"[{p['base_col']}]"] + [f"{s['op']} [{s['col']}]" for s in p.get("ops", [])]
        return f"CALC: {p['outcol']} = {' '.join(seq)}"
    if t == "dedupe":
        return f"Deduplica su: {p.get('col','')}"
    return t

def delete_step(fname, idx):
    del st.session_state.STATE["files"][fname]["steps"][idx:]
    clear_cache_from(fname, idx)

def _base_add_step(f, step):
    st.session_state.STATE["files"][f]["steps"].append(step)
    clear_cache_from(f, len(st.session_state.STATE["files"][f]["steps"]) - 1)

# ==========================================
# UI APP STREAMLIT
# ==========================================
st.title("⚡ Power Query Web UI")
st.markdown("Copia fedele della versione Jupyter, ottimizzata per RAM e Server Web.")

files_in_mem = st.session_state.STATE["order"]

t_files, t_prev, t_flt, t_merge, t_calc, t_metr, t_exp = st.tabs([
    "📁 File & Steps", "👁️ Formati & Preview", "🔎 Filtri", "🔗 Merge",
    "🧮 Calcoli & IFS", "📊 Metriche", "⬇️ Export"
])

# --- TAB 1: FILE E STEPS ---
with t_files:
    st.subheader("Importa File")
    import_mode = st.radio(
        "Modalità Importazione:",
        ["smart", "string"],
        format_func=lambda x: "SMART (Riconosce Date/Numeri - Veloce)" if x == "smart" else "TUTTO STRING (Testo - Massima Stabilità)"
    )

    uploaded_files = st.file_uploader("Trascina CSV o Excel qui", accept_multiple_files=True)
    if uploaded_files:
        if st.button("Carica File Selezionati", type="primary"):
            progress_bar = st.progress(0, "Lettura file...")
            for idx, uf in enumerate(uploaded_files):
                temp_path = Path(st.session_state.temp_dir) / uf.name
                with open(temp_path, "wb") as f:
                    f.write(uf.getbuffer())
                try:
                    df = read_file(
                        temp_path,
                        progress_cb=lambda frac, i=idx, nm=uf.name: progress_bar.progress(
                            (i + frac) / len(uploaded_files),
                            text=f"Caricamento {nm} ({frac*100:.0f}%)"
                        ),
                        import_mode=import_mode
                    )
                    add_loaded_df(uf.name, df, path=temp_path, import_mode=import_mode)
                    st.success(f"✅ {uf.name} caricato! ({len(df):,} righe)")
                except Exception as e:
                    st.error(f"❌ Errore caricando {uf.name}: {e}")
            progress_bar.progress(1.0, "Caricamento completato!")
            st.rerun()

    st.divider()
    st.subheader("Step Applicati")
    if not files_in_mem:
        st.info("Nessun file in memoria. Carica un file per iniziare.")
    else:
        for fname in files_in_mem:
            with st.expander(f"📄 {fname} ({len(st.session_state.STATE['files'][fname]['df_base']):,} righe base)", expanded=False):
                steps = st.session_state.STATE["files"][fname]["steps"]
                if not steps:
                    st.write("*Nessuno step applicato.*")
                else:
                    try:
                        get_df(fname)
                    except Exception:
                        pass

                    for i, step in enumerate(steps):
                        rc = step.get("_row_count", "—")
                        c1, c2 = st.columns([4, 1])
                        c1.write(f"**{i+1}.** {step_label(step)} *(Righe: {rc:,})*")
                        if c2.button("Elimina da qui", key=f"del_{fname}_{i}", help="Elimina questo step e i successivi"):
                            delete_step(fname, i)
                            st.rerun()

        if st.button("🗑️ Svuota Tutto la Memoria", type="secondary"):
            st.session_state.STATE = {"files": {}, "order": []}
            st.session_state.calc_ops_list = []
            st.session_state.multi_conds = []
            st.session_state.run_conds = []
            st.rerun()

def get_cols(fname):
    if not fname:
        return []
    try:
        return list(get_df(fname).columns)
    except Exception:
        return []

# --- TAB 2: FORMATI E PREVIEW ---
with t_prev:
    if files_in_mem:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Applica Formato")
            with st.form("fmt_form"):
                fmt_file = st.selectbox("File:", files_in_mem, key="fmt_f")
                fmt_col = st.selectbox("Colonna:", get_cols(fmt_file), key="fmt_c")
                fmt_type = st.selectbox("Tipo:", ["auto", "text", "number", "currency", "percent", "date", "datetime"])

                date_input = None
                dec_sep, thou_sep = None, None

                if fmt_type in ("date", "datetime"):
                    date_input = st.selectbox("Input Data:", list(DATE_INPUT_MAP.keys()))
                if fmt_type in ("number", "currency", "percent"):
                    c1, c2 = st.columns(2)
                    dec_sep = c1.selectbox("Decimali:", ["", ".", ","])
                    thou_sep = c2.selectbox("Migliaia:", ["", "__NONE__", ".", ",", " ", "'"])

                submitted = st.form_submit_button("✅ Applica Formato")
                if submitted:
                    spec = {"type": fmt_type}
                    if date_input:
                        spec["date_input"] = date_input
                    if dec_sep:
                        spec["decimal_sep"] = dec_sep
                    if thou_sep:
                        spec["thousands_sep"] = thou_sep

                    step = {"type": "format", "created_at": datetime.now().isoformat(), "params": {"formats": {fmt_col: spec}}}
                    _base_add_step(fmt_file, step)
                    st.success("Formato applicato!")
                    st.rerun()

        with col2:
            st.subheader("Anteprima Dati")
            prev_file = st.selectbox("Scegli file da visualizzare:", files_in_mem, key="prev_f")
            if prev_file:
                df_prev = get_df(prev_file).head(100)
                st.write(f"Mostrando le prime 100 righe su **{len(get_df(prev_file)):,}** totali.")
                st.dataframe(df_prev, use_container_width=True)

# --- TAB 3: FILTRI ---
with t_flt:
    if files_in_mem:
        st.subheader("Filtro Singolo")
        with st.form("single_filter_form"):
            c1, c2, c3 = st.columns(3)
            flt_file = c1.selectbox("File:", files_in_mem, key="sflt_f")
            flt_col = c2.selectbox("Colonna:", get_cols(flt_file), key="sflt_c")
            flt_kind = c3.selectbox("Tipo dato:", ["number", "text", "date"], key="sflt_kind")

            ops = OPS_NUMBER if flt_kind == "number" else (OPS_DATE if flt_kind == "date" else OPS_TEXT)
            op_values = [v for (v, _) in ops]
            op_labels = {v: lab for (v, lab) in ops}

            c4, c5 = st.columns([1, 2])
            flt_op = c4.selectbox(
                "Operatore:",
                op_values,
                format_func=lambda v: op_labels.get(v, v),
                key="sflt_op"
            )

            if flt_kind == "date":
                v1 = c5.date_input("Valore 1", key="sflt_v1_date")
                v2 = c5.date_input("Valore 2", key="sflt_v2_date") if flt_op == "between" else None
            else:
                v1 = c5.text_input("Valore 1", key="sflt_v1_txt")
                v2 = c5.text_input("Valore 2", key="sflt_v2_txt") if flt_op == "between" else None

            submitted = st.form_submit_button("✅ Applica Filtro Singolo")
            if submitted:
                val1 = v1.isoformat() if flt_kind == "date" else v1
                val2 = v2.isoformat() if (flt_kind == "date" and v2) else v2
                step = {"type": "filter", "created_at": datetime.now().isoformat(),
                        "params": {"col": flt_col, "kind": flt_kind, "op": flt_op, "v1": val1, "v2": val2}}
                _base_add_step(flt_file, step)
                st.success("Filtro applicato!")
                st.rerun()

        st.divider()
        st.subheader("Rimuovi Duplicati")
        with st.form("dedupe_form"):
            c1, c2 = st.columns(2)
            dd_file = c1.selectbox("File:", files_in_mem, key="dd_f")
            dd_col = c2.selectbox("Colonna chiave:", get_cols(dd_file), key="dd_c")
            submitted = st.form_submit_button("🧹 Rimuovi Duplicati")
            if submitted:
                step = {"type": "dedupe", "created_at": datetime.now().isoformat(), "params": {"col": dd_col, "keep": "first"}}
                _base_add_step(dd_file, step)
                st.success("Duplicati rimossi!")
                st.rerun()

# --- TAB 4: MERGE ---
with t_merge:
    if files_in_mem:
        st.subheader("Unisci due file (Join)")
        merge_mode = st.radio(
            "Destinazione:",
            ["existing", "new"],
            format_func=lambda x: "Mergia su file esistente (A)" if x == "existing" else "Crea NUOVO file in memoria",
            key="merge_mode"
        )
        new_name = st.text_input("Nome nuovo file:", "merged_1", key="merge_new_name") if merge_mode == "new" else None

        c1, c2 = st.columns(2)
        m_A = c1.selectbox("File A (Base):", files_in_mem, key="m_A")
        m_B = c2.selectbox("File B (Da unire):", files_in_mem, key="m_B")

        c3, c4 = st.columns(2)
        m_on_A = c3.selectbox("Chiave in A:", get_cols(m_A), key="mon_A")
        m_on_B = c4.selectbox("Chiave in B:", get_cols(m_B), key="mon_B")

        c5, c6 = st.columns(2)
        m_how = c5.selectbox("Tipo Join:", ["left", "right", "inner", "outer"], key="m_how")
        m_coll = c6.selectbox("Collisioni nomi:", ["suffix", "keep_left", "keep_right"], key="m_coll",
                              format_func=lambda x: "Aggiungi _B automatico" if x == "suffix" else x)

        if st.button("🔗 Esegui Merge", type="primary"):
            if merge_mode == "existing":
                step = {"type": "merge", "created_at": datetime.now().isoformat(),
                        "params": {"right_file": m_B, "how": m_how, "left_on": m_on_A, "right_on": m_on_B, "collision": m_coll}}
                _base_add_step(m_A, step)
                st.success("Merge completato!")
            else:
                nn = sanitize_name(new_name, "merged_1")
                create_new_file_from_merge(nn, m_A, m_B, m_how, m_on_A, m_on_B, m_coll)
                st.success(f"Nuovo file {nn} creato!")
            st.rerun()

# --- TAB 5: CALCOLI & IFS ---
with t_calc:
    if files_in_mem:
        st.subheader("Calcoli Multi-Colonna (Riga per Riga)")
        st.info("I vuoti valgono 0. Le operazioni vengono eseguite rigorosamente da sinistra a destra.")

        c1, c2 = st.columns(2)
        calc_f = c1.selectbox("File:", files_in_mem, key="calc_f")
        calc_base = c2.selectbox("Colonna Base:", get_cols(calc_f), key="calc_base")

        with st.container(border=True):
            st.write("Aggiungi operazioni in sequenza:")
            cc1, cc2, cc3 = st.columns([2, 3, 1])
            add_op = cc1.selectbox("Op:", ["add", "sub", "mul", "div"], key="calc_add_op",
                                   format_func=lambda x: {"add":"+ (Somma)","sub":"- (Sottrai)","mul":"* (Moltiplica)","div":"/ (Dividi)"}[x])
            add_col = cc2.selectbox("Con colonna:", get_cols(calc_f), key="calc_next")
            if cc3.button("➕ Aggiungi Op"):
                st.session_state.calc_ops_list.append({"op": add_op, "col": add_col})
                st.rerun()

            if st.session_state.calc_ops_list:
                formula = f"[{calc_base}]"
                for op in st.session_state.calc_ops_list:
                    formula += f" {op['op']} [{op['col']}]"
                st.code(formula, language="text")
                if st.button("Reset Operazioni"):
                    st.session_state.calc_ops_list = []
                    st.rerun()

        calc_out = st.text_input("Nome nuova colonna:", "calc_row_1", key="calc_out")
        if st.button("✅ Esegui Calcolo", type="primary"):
            step = {"type": "calc", "created_at": datetime.now().isoformat(),
                    "params": {"base_col": calc_base, "ops": st.session_state.calc_ops_list, "outcol": calc_out}}
            _base_add_step(calc_f, step)
            st.session_state.calc_ops_list = []
            st.success("Calcolo applicato!")
            st.rerun()

        st.divider()
        st.subheader("Aggregazione IFS (Es. Somma se, Conta se)")
        with st.form("ifs_form"):
            c1, c2 = st.columns(2)
            ifs_A = c1.selectbox("File Destinazione (A):", files_in_mem, key="ifs_a")
            ifs_B = c2.selectbox("File Ricerca (B):", files_in_mem, key="ifs_b")

            c3, c4 = st.columns(2)
            ifs_match_a = c3.selectbox("Col. Match in A:", get_cols(ifs_A), key="im_a")
            ifs_match_b = c4.selectbox("Col. Match in B:", get_cols(ifs_B), key="im_b")

            c5, c6 = st.columns(2)
            metric = c5.selectbox("Funzione:", ["countifs", "sumifs", "averageifs"], key="ifs_metric")
            val_col = c6.selectbox("Colonna Valore (in B):", [""] + get_cols(ifs_B), key="ifs_val_col")

            ifs_out = st.text_input("Nome nuova colonna in A:", "ifs_result_1", key="ifs_out")

            submitted = st.form_submit_button("✅ Esegui IFS")
            if submitted:
                if metric in ["sumifs", "averageifs"] and not val_col:
                    st.error("Seleziona una colonna valore per SUM/AVERAGE.")
                else:
                    step = {"type": "ifs", "created_at": datetime.now().isoformat(),
                            "params": {"lookup_file": ifs_B, "match_a_col": ifs_match_a, "match_b_col": ifs_match_b,
                                       "metric": metric, "value_col": val_col, "outcol": ifs_out, "conds": st.session_state.run_conds}}
                    _base_add_step(ifs_A, step)
                    st.success("IFS Completato!")
                    st.rerun()

# --- TAB 6: METRICHE ---
with t_metr:
    if files_in_mem:
        st.subheader("Calcola Metrica Veloce (Senza creare Step)")
        c1, c2 = st.columns(2)
        met_file = c1.selectbox("File:", files_in_mem, key="met_f")
        met_col = c2.selectbox("Colonna:", get_cols(met_file), key="met_c")

        c3, c4 = st.columns(2)
        met_kind = c3.selectbox("Metrica:", ["sum", "count", "counta", "average", "median", "quartile", "percentile"], key="met_kind")
        q_val = c4.text_input("Q/Percentile (es. 0.25):", "0.25", key="met_q") if met_kind in ["quartile", "percentile"] else None

        if st.button("📌 Calcola", type="primary"):
            df = get_df(met_file)
            try:
                val = compute_metric(met_file, df, met_col, met_kind, q=float(q_val) if q_val else None)
                st.success(f"Risultato **{met_kind.upper()}** su {met_col}: **{val}**")
            except Exception as e:
                st.error(f"Errore: {e}")

# --- TAB 7: EXPORT ---
with t_exp:
    if files_in_mem:
        st.subheader("Esporta Risultati (CSV)")
        st.info("L'esportazione usa la memoria in modo ottimizzato. I file pronti appariranno qui sotto come download.")

        export_f = st.selectbox("File da esportare:", files_in_mem, key="exp_f")
        exp_name = st.text_input("Nome file output:", f"export_{export_f}", key="exp_name")

        if st.button("Genera File CSV"):
            df_out = get_df(export_f)
            if HAS_PYARROW:
                import pyarrow as pa
                import pyarrow.csv as pacsv
                tbl = pa.Table.from_pandas(df_out, preserve_index=False)
                buffer = io.BytesIO()
                pacsv.write_csv(tbl, buffer)
                csv_data = buffer.getvalue()
            else:
                csv_data = df_out.to_csv(index=False, sep=",", encoding="utf-8-sig").encode("utf-8")

            st.download_button(
                label=f"⬇️ Scarica {sanitize_name(exp_name)}.csv",
                data=csv_data,
                file_name=f"{sanitize_name(exp_name)}.csv",
                mime="text/csv",
                type="primary"
            )
