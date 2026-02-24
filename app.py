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
import duckdb

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Power Query Web UI (HF / DuckDB)", page_icon="⚡", layout="wide")
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

APP_DIR = Path(__file__).parent.resolve()
WORK_DIR = Path(os.getenv("PQ_WORKDIR", APP_DIR / "pq_workdir"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = WORK_DIR / "pq.duckdb"

# =========================
# SESSION STATE
# =========================
if "STATE" not in st.session_state:
    st.session_state.STATE = {"files": {}, "order": []}
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp(dir=str(WORK_DIR))
if "calc_ops_list" not in st.session_state:
    st.session_state.calc_ops_list = []
if "run_conds" not in st.session_state:
    st.session_state.run_conds = []

# =========================
# OPS UI
# =========================
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

# =========================
# UTIL
# =========================
_BAD = re.compile(r'[\\/:*?"<>|]+')
def sanitize_name(name: str, fallback="output") -> str:
    name = (name or "").strip()
    name = _BAD.sub("_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("_")
    return name if name else fallback

def db():
    con = duckdb.connect(str(DB_PATH))
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA memory_limit='10GB';")
    con.execute("PRAGMA temp_directory='{}';".format(str(WORK_DIR / "duck_tmp")))
    (WORK_DIR / "duck_tmp").mkdir(parents=True, exist_ok=True)
    return con

def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def sql_lit(s: str) -> str:
    return "'" + (s or "").replace("'", "''") + "'"

def reset_keys(keys: list[str]):
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

def ensure_dep_updates(prefix: str, file_key: str, dep_keys: list[str]):
    """
    Se cambia il file selezionato, resetta chiavi dipendenti (colonna/operatore/valori)
    per evitare che Streamlit tenga valori non validi e non aggiorni le opzioni.
    """
    prev_key = f"{prefix}__prev_file"
    cur = st.session_state.get(file_key, None)
    prev = st.session_state.get(prev_key, None)
    if cur != prev:
        reset_keys(dep_keys)
        st.session_state[prev_key] = cur

def declared_type_for_col(file_name: str, col: str) -> str | None:
    steps = st.session_state.STATE["files"][file_name]["steps"]
    declared = None
    for stp in steps:
        if stp.get("type") == "format":
            fmts = stp.get("params", {}).get("formats", {})
            if col in fmts and isinstance(fmts[col], dict):
                declared = fmts[col].get("type")
    return declared

# =========================
# INGEST
# =========================
def detect_csv_sep_from_sample(text: str) -> str:
    candidates = {";": text.count(";"), ",": text.count(","), "\t": text.count("\t"), "|": text.count("|")}
    sep = max(candidates, key=candidates.get)
    return sep if candidates[sep] > 0 else ";"

def ingest_file_to_duckdb(file_path: Path, import_mode: str) -> tuple[str, list[str]]:
    suf = file_path.suffix.lower()
    tname = f"base_{sanitize_name(file_path.stem)}_{abs(hash(str(file_path)))%10_000_000}"
    tname = sanitize_name(tname)

    con = db()
    try:
        if suf == ".csv":
            with open(file_path, "rb") as f:
                sample = f.read(65536)
            try:
                sample_txt = sample.decode("utf-8", errors="ignore")
            except Exception:
                sample_txt = sample.decode("latin1", errors="ignore")
            sep = detect_csv_sep_from_sample(sample_txt)

            if import_mode == "string":
                con.execute(f"""
                    CREATE OR REPLACE TABLE {qident(tname)} AS
                    SELECT * FROM read_csv(
                        {sql_lit(str(file_path))},
                        delim={sql_lit(sep)},
                        header=true,
                        all_varchar=true,
                        ignore_errors=true
                    );
                """)
            else:
                con.execute(f"""
                    CREATE OR REPLACE TABLE {qident(tname)} AS
                    SELECT * FROM read_csv_auto(
                        {sql_lit(str(file_path))},
                        delim={sql_lit(sep)},
                        header=true,
                        ignore_errors=true,
                        sample_size=500000
                    );
                """)

        elif suf in (".xlsx", ".xls"):
            df = pd.read_excel(file_path, engine=None)
            if import_mode == "string":
                df = df.astype("string").fillna("")
            con.register("tmp_df", df)
            con.execute(f"CREATE OR REPLACE TABLE {qident(tname)} AS SELECT * FROM tmp_df;")
            con.unregister("tmp_df")
        else:
            raise ValueError(f"Formato non supportato: {suf}")

        cols = [r[0] for r in con.execute(f"DESCRIBE {qident(tname)}").fetchall()]
        return tname, cols
    finally:
        con.close()

def add_loaded_file(name: str, table_name: str, cols: list[str], path: Path, import_mode: str):
    fname = name
    base = fname
    i = 2
    while fname in st.session_state.STATE["files"]:
        fname = f"{base} ({i})"
        i += 1

    meta = {"import_mode": import_mode, "loaded_at": datetime.now().isoformat(), "path": str(path)}
    st.session_state.STATE["files"][fname] = {
        "table_base": table_name,
        "cols_base": cols,
        "steps": [],
        "smart_cache": {"step_idx": -1, "sql": f"SELECT * FROM {qident(table_name)}"},
        "meta": meta,
    }
    st.session_state.STATE["order"].append(fname)
    return fname

def clear_cache_from(fname: str, idx: int):
    rec = st.session_state.STATE["files"][fname]
    if rec["smart_cache"]["step_idx"] >= idx:
        rec["smart_cache"] = {"step_idx": -1, "sql": f"SELECT * FROM {qident(rec['table_base'])}"}

# =========================
# STEP -> SQL
# =========================
def sql_parse_number(expr: str, decimal_sep: str | None = None, thousands_sep: str | None = None) -> str:
    x = f"trim({expr})"
    if thousands_sep and thousands_sep != "__NONE__":
        x = f"replace({x}, {sql_lit(thousands_sep)}, '')"
    x = f"replace(replace({x}, '€',''), '%','')"
    if decimal_sep and decimal_sep != "." and decimal_sep != "":
        x = f"replace({x}, {sql_lit(decimal_sep)}, '.')"
    return f"try_cast(nullif({x}, '') as DOUBLE)"

def sql_parse_date(expr: str, date_input_label: str) -> str:
    spec = DATE_INPUT_MAP.get(date_input_label, {"kind": "auto"})
    kind = spec.get("kind", "auto")
    if kind == "fmt" and spec.get("fmt"):
        return f"try_cast(strptime({expr}, {sql_lit(spec['fmt'])}) as TIMESTAMP)"
    return f"try_cast({expr} as TIMESTAMP)"

def step_to_sql(fname: str, step: dict, sql_in: str) -> str:
    t = step["type"]
    p = step["params"]

    if t == "format":
        return sql_in

    if t == "filter":
        col = p["col"]; kind = p["kind"]; op = p["op"]; v1 = p["v1"]; v2 = p.get("v2")
        c = qident(col)

        if kind == "number":
            num = sql_parse_number(f"{c}")
            n1 = sql_parse_number(sql_lit(str(v1)))
            n2 = sql_parse_number(sql_lit(str(v2))) if v2 else None
            if op == "gt": cond = f"{num} > {n1}"
            elif op == "gte": cond = f"{num} >= {n1}"
            elif op == "lt": cond = f"{num} < {n1}"
            elif op == "lte": cond = f"{num} <= {n1}"
            elif op == "eq": cond = f"{num} = {n1}"
            elif op == "neq": cond = f"{num} <> {n1}"
            elif op == "between" and n2 is not None:
                cond = f"({num} between least({n1},{n2}) and greatest({n1},{n2}))"
            else:
                cond = "TRUE"

        elif kind == "date":
            dt = sql_parse_date(f"{c}", "Auto (prova a riconoscere)")
            d1 = sql_parse_date(sql_lit(str(v1)), "Auto (prova a riconoscere)")
            d2 = sql_parse_date(sql_lit(str(v2)), "Auto (prova a riconoscere)") if v2 else None
            if op == "after": cond = f"{dt} > {d1}"
            elif op == "before": cond = f"{dt} < {d1}"
            elif op == "eq": cond = f"{dt} = {d1}"
            elif op == "neq": cond = f"{dt} <> {d1}"
            elif op == "between" and d2 is not None:
                cond = f"({dt} between least({d1},{d2}) and greatest({d1},{d2}))"
            else:
                cond = "TRUE"

        else:
            txt = f"coalesce(cast({c} as VARCHAR),'')"
            v1s = str(v1 or "")
            if op == "eq": cond = f"{txt} = {sql_lit(v1s)}"
            elif op == "neq": cond = f"{txt} <> {sql_lit(v1s)}"
            elif op == "contains": cond = f"lower({txt}) like '%' || lower({sql_lit(v1s)}) || '%'"
            elif op == "not_contains": cond = f"not (lower({txt}) like '%' || lower({sql_lit(v1s)}) || '%')"
            elif op == "startswith": cond = f"lower({txt}) like lower({sql_lit(v1s)}) || '%'"
            elif op == "endswith": cond = f"lower({txt}) like '%' || lower({sql_lit(v1s)})"
            elif op == "is_empty": cond = f"length({txt}) = 0"
            elif op == "not_empty": cond = f"length({txt}) > 0"
            else: cond = "TRUE"

        return f"SELECT * FROM ({sql_in}) WHERE {cond}"

    if t == "filter_multi":
        logic = p.get("logic", "AND").upper()
        conds = p.get("conds", [])
        if not conds:
            return sql_in
        parts = []
        for cnd in conds:
            col = cnd["col"]; kind = cnd["kind"]; op = cnd["op"]; v1 = cnd["v1"]; v2 = cnd.get("v2")
            cc = qident(col)
            if kind == "number":
                num = sql_parse_number(f"{cc}")
                n1 = sql_parse_number(sql_lit(str(v1)))
                n2 = sql_parse_number(sql_lit(str(v2))) if v2 else None
                if op == "gt": cond = f"{num} > {n1}"
                elif op == "gte": cond = f"{num} >= {n1}"
                elif op == "lt": cond = f"{num} < {n1}"
                elif op == "lte": cond = f"{num} <= {n1}"
                elif op == "eq": cond = f"{num} = {n1}"
                elif op == "neq": cond = f"{num} <> {n1}"
                elif op == "between" and n2 is not None: cond = f"({num} between least({n1},{n2}) and greatest({n1},{n2}))"
                else: cond = "TRUE"
            elif kind == "date":
                dt = sql_parse_date(f"{cc}", "Auto (prova a riconoscere)")
                d1 = sql_parse_date(sql_lit(str(v1)), "Auto (prova a riconoscere)")
                d2 = sql_parse_date(sql_lit(str(v2)), "Auto (prova a riconoscere)") if v2 else None
                if op == "after": cond = f"{dt} > {d1}"
                elif op == "before": cond = f"{dt} < {d1}"
                elif op == "eq": cond = f"{dt} = {d1}"
                elif op == "neq": cond = f"{dt} <> {d1}"
                elif op == "between" and d2 is not None: cond = f"({dt} between least({d1},{d2}) and greatest({d1},{d2}))"
                else: cond = "TRUE"
            else:
                txt = f"coalesce(cast({cc} as VARCHAR),'')"
                v1s = str(v1 or "")
                if op == "eq": cond = f"{txt} = {sql_lit(v1s)}"
                elif op == "neq": cond = f"{txt} <> {sql_lit(v1s)}"
                elif op == "contains": cond = f"lower({txt}) like '%' || lower({sql_lit(v1s)}) || '%'"
                elif op == "not_contains": cond = f"not (lower({txt}) like '%' || lower({sql_lit(v1s)}) || '%')"
                elif op == "startswith": cond = f"lower({txt}) like lower({sql_lit(v1s)}) || '%'"
                elif op == "endswith": cond = f"lower({txt}) like '%' || lower({sql_lit(v1s)})"
                elif op == "is_empty": cond = f"length({txt}) = 0"
                elif op == "not_empty": cond = f"length({txt}) > 0"
                else: cond = "TRUE"
            parts.append(f"({cond})")
        glue = " OR " if logic == "OR" else " AND "
        where = glue.join(parts) if parts else "TRUE"
        return f"SELECT * FROM ({sql_in}) WHERE {where}"

    if t == "dedupe":
        col = p["col"]
        c = qident(col)
        return f"""
            SELECT * EXCLUDE(rn)
            FROM (
                SELECT *, row_number() OVER (PARTITION BY {c}) AS rn
                FROM ({sql_in})
            )
            WHERE rn = 1
        """

    if t == "merge":
        right_file = p["right_file"]
        how = p["how"]
        left_on = p["left_on"]
        right_on = p["right_on"]
        collision = p.get("collision", "suffix")

        right_sql = get_sql(right_file)

        con = db()
        try:
            left_cols = [r[0] for r in con.execute(f"DESCRIBE ({sql_in})").fetchall()]
            right_cols = [r[0] for r in con.execute(f"DESCRIBE ({right_sql})").fetchall()]
        finally:
            con.close()

        dup = (set(left_cols) & set(right_cols)) - {left_on, right_on}
        r_select = []
        for c in right_cols:
            if c in dup:
                if collision == "keep_left":
                    continue
                if collision == "suffix":
                    r_select.append(f"{qident(c)} AS {qident(c + '_B')}")
                else:
                    r_select.append(f"{qident(c)}")
            else:
                r_select.append(f"{qident(c)}")

        left = f"({sql_in})"
        right = f"(SELECT {', '.join(r_select)} FROM ({right_sql}))"

        join_type = how.upper()
        if join_type == "OUTER":
            join_type = "FULL OUTER"

        return f"""
            SELECT *
            FROM {left} AS A
            {join_type} JOIN {right} AS B
            ON A.{qident(left_on)} = B.{qident(right_on)}
        """

    if t == "calc":
        outcol = p["outcol"]
        base_col = p.get("col_left", p.get("base_col"))
        ops = [{"op": p["op"], "col": p["col_right"]}] if "col_left" in p else p.get("ops", [])

        if declared_type_for_col(fname, base_col) == "text":
            raise ValueError("Operazioni non consentite su Testo.")

        expr = f"coalesce({sql_parse_number(qident(base_col))}, 0.0)"
        for opstep in ops:
            op = opstep["op"]
            c = opstep["col"]
            if declared_type_for_col(fname, c) == "text":
                raise ValueError("Operazioni non consentite su Testo.")
            nxt = f"coalesce({sql_parse_number(qident(c))}, 0.0)"
            if op == "add": expr = f"({expr} + {nxt})"
            elif op == "sub": expr = f"({expr} - {nxt})"
            elif op == "mul": expr = f"({expr} * {nxt})"
            elif op == "div": expr = f"({expr} / nullif({nxt}, 0.0))"

        calc_expr = f"""
            CASE
              WHEN {expr} IS NULL OR isinf({expr}) THEN 'ERRORE: Div/0'
              ELSE cast({expr} as VARCHAR)
            END
        """
        return f"SELECT *, {calc_expr} AS {qident(outcol)} FROM ({sql_in})"

    if t == "ifs":
        lookup_file = p["lookup_file"]
        match_a = p["match_a_col"]
        match_b = p["match_b_col"]
        metric = p["metric"]
        value_col = p.get("value_col", "")
        outcol = p["outcol"]
        conds = p.get("conds", [])

        lookup_sql = get_sql(lookup_file)
        for cnd in conds:
            lookup_sql = step_to_sql(lookup_file, {"type": "filter", "params": cnd}, lookup_sql)

        bkey = f"coalesce(cast({qident(match_b)} as VARCHAR),'')"
        akey = f"coalesce(cast(A.{qident(match_a)} as VARCHAR),'')"

        if metric == "countifs":
            agg = f"SELECT {bkey} AS k, count(*)::BIGINT AS v FROM ({lookup_sql}) GROUP BY 1"
            vexpr = "coalesce(B.v, 0)::BIGINT"
        else:
            if declared_type_for_col(lookup_file, value_col) == "text":
                raise ValueError("Colonna valore formattata come Testo.")
            val = f"{sql_parse_number(qident(value_col))}"
            if metric == "sumifs":
                agg = f"SELECT {bkey} AS k, coalesce(sum({val}),0.0) AS v FROM ({lookup_sql}) GROUP BY 1"
                vexpr = "coalesce(B.v, 0.0)"
            else:
                agg = f"SELECT {bkey} AS k, avg({val}) AS v FROM ({lookup_sql}) GROUP BY 1"
                vexpr = "B.v"

        return f"""
            SELECT A.*, {vexpr} AS {qident(outcol)}
            FROM ({sql_in}) A
            LEFT JOIN ({agg}) B
            ON {akey} = B.k
        """

    raise ValueError(f"Step non supportato: {t}")

# =========================
# SQL builder + cache
# =========================
def get_sql(fname: str) -> str:
    rec = st.session_state.STATE["files"][fname]
    base_sql = f"SELECT * FROM {qident(rec['table_base'])}"
    steps = rec["steps"]

    cache_idx = rec["smart_cache"]["step_idx"]
    sql = rec["smart_cache"]["sql"]

    if cache_idx >= len(steps):
        cache_idx = -1
        sql = base_sql

    start_idx = cache_idx + 1
    if start_idx < len(steps):
        for i in range(start_idx, len(steps)):
            st_dict = steps[i]
            sql = step_to_sql(fname, st_dict, sql)
        rec["smart_cache"] = {"step_idx": len(steps) - 1, "sql": sql}

    return sql

def compute_metric_sql(file_name: str, sql: str, col: str, metric: str, q: float | None = None):
    if metric in ("sum", "average", "median", "quartile", "percentile") and declared_type_for_col(file_name, col) == "text":
        raise ValueError("Metrica numerica non consentita su Testo.")
    c = qident(col)

    if metric == "counta":
        qsql = f"SELECT count({c}) FROM ({sql})"
    elif metric == "count":
        qsql = f"SELECT count(try_cast({c} as DOUBLE)) FROM ({sql})"
    elif metric == "sum":
        qsql = f"SELECT sum({sql_parse_number(c)}) FROM ({sql})"
    elif metric == "average":
        qsql = f"SELECT avg({sql_parse_number(c)}) FROM ({sql})"
    elif metric == "median":
        qsql = f"SELECT median({sql_parse_number(c)}) FROM ({sql})"
    elif metric == "quartile":
        qq = q if q is not None else 0.25
        qsql = f"SELECT quantile_cont({sql_parse_number(c)}, {qq}) FROM ({sql})"
    elif metric == "percentile":
        qq = q if q is not None else 0.50
        qsql = f"SELECT quantile_cont({sql_parse_number(c)}, {qq}) FROM ({sql})"
    else:
        raise ValueError("Metrica non supportata.")

    con = db()
    try:
        return con.execute(qsql).fetchone()[0]
    finally:
        con.close()

# =========================
# UI HELPERS
# =========================
def step_label(st_dict: dict) -> str:
    t, p = st_dict["type"], st_dict["params"]
    if t == "format":
        fmts = p.get("formats", {})
        if not fmts: return "Formato"
        parts = []
        for c, v in fmts.items():
            parts.append(f"{c}:{v.get('type','auto')}")
        return "Formato: " + ", ".join(parts)
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
        seq = [f"[{p.get('base_col','')}]"] + [f"{s['op']} [{s['col']}]" for s in p.get("ops", [])]
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

def get_cols(fname):
    if not fname: return []
    sql = get_sql(fname)
    con = db()
    try:
        cols = [r[0] for r in con.execute(f"DESCRIBE ({sql})").fetchall()]
        return cols
    finally:
        con.close()

def preview_df(fname, limit=100):
    sql = get_sql(fname)
    con = db()
    try:
        return con.execute(f"SELECT * FROM ({sql}) LIMIT {int(limit)}").df()
    finally:
        con.close()

# =========================
# UI
# =========================
st.title("⚡ Power Query Web UI (Hugging Face / DuckDB)")
st.markdown("Stesse funzioni/step della versione pandas, ma esecuzione **on-disk** per file enormi (5GB+).")

files_in_mem = st.session_state.STATE["order"]

t_files, t_prev, t_flt, t_merge, t_calc, t_metr, t_exp = st.tabs([
    "📁 File & Steps", "👁️ Formati & Preview", "🔎 Filtri", "🔗 Merge", "🧮 Calcoli & IFS", "📊 Metriche", "⬇️ Export"
])

# --- TAB 1: FILE & STEPS ---
with t_files:
    st.subheader("Importa File")
    import_mode = st.radio(
        "Modalità Importazione:",
        ["smart", "string"],
        index=1,
        format_func=lambda x: "SMART (tipi auto - più veloce)" if x == "smart" else "TUTTO STRING (massima stabilità)"
    )

    uploaded_files = st.file_uploader("Trascina CSV o Excel qui", accept_multiple_files=True)
    if uploaded_files:
        if st.button("Carica File Selezionati", type="primary"):
            prog = st.progress(0.0, "Upload + ingest in DuckDB...")
            for idx, uf in enumerate(uploaded_files):
                temp_path = Path(st.session_state.temp_dir) / uf.name
                with open(temp_path, "wb") as f:
                    f.write(uf.getbuffer())

                try:
                    prog.progress(idx / max(1, len(uploaded_files)), text=f"Ingest: {uf.name}")
                    tname, cols = ingest_file_to_duckdb(temp_path, import_mode=import_mode)
                    add_loaded_file(uf.name, tname, cols, temp_path, import_mode)
                    st.success(f"✅ {uf.name} importato in DuckDB ({len(cols)} colonne)")
                except Exception as e:
                    st.error(f"❌ Errore su {uf.name}: {e}")

            prog.progress(1.0, text="Completato")
            st.rerun()

    st.divider()
    st.subheader("Step Applicati")
    if not files_in_mem:
        st.info("Nessun file in memoria. Carica un file per iniziare.")
    else:
        for fname in files_in_mem:
            rec = st.session_state.STATE["files"][fname]
            with st.expander(f"📄 {fname} (base={rec['table_base']})", expanded=False):
                steps = rec["steps"]
                if not steps:
                    st.write("*Nessuno step applicato.*")
                else:
                    for i, step in enumerate(steps):
                        c1, c2 = st.columns([4, 1])
                        c1.write(f"**{i+1}.** {step_label(step)}")
                        if c2.button("Elimina da qui", key=f"del_{fname}_{i}"):
                            delete_step(fname, i)
                            st.rerun()

        if st.button("🗑️ Svuota Memoria (state)"):
            st.session_state.STATE = {"files": {}, "order": []}
            st.session_state.calc_ops_list = []
            st.session_state.run_conds = []
            st.rerun()

# --- TAB 2: FORMATI & PREVIEW ---
with t_prev:
    if files_in_mem:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Applica Formato")
            # file selector + reset colonna quando cambia file
            fmt_file_key = "fmt_file"
            fmt_col_key = "fmt_col"
            st.selectbox("File:", files_in_mem, key=fmt_file_key)
            ensure_dep_updates("fmt", fmt_file_key, [fmt_col_key])

            with st.form("fmt_form"):
                fmt_file = st.session_state.get(fmt_file_key)
                cols = get_cols(fmt_file) if fmt_file else []
                st.selectbox("Colonna:", cols, key=fmt_col_key)
                fmt_col = st.session_state.get(fmt_col_key)

                fmt_type = st.selectbox("Tipo:", ["auto", "text", "number", "currency", "percent", "date", "datetime"], key="fmt_type")

                date_input = None
                dec_sep, thou_sep = None, None
                if fmt_type in ("date", "datetime"):
                    date_input = st.selectbox("Input Data:", list(DATE_INPUT_MAP.keys()), key="fmt_date_input")
                if fmt_type in ("number", "currency", "percent"):
                    c1, c2 = st.columns(2)
                    dec_sep = c1.selectbox("Decimali:", ["", ".", ","], key="fmt_dec")
                    thou_sep = c2.selectbox("Migliaia:", ["", "__NONE__", ".", ",", " ", "'"], key="fmt_thou")

                submitted = st.form_submit_button("✅ Applica Formato")
                if submitted:
                    if not fmt_file or not fmt_col:
                        st.error("Seleziona file e colonna.")
                    else:
                        spec = {"type": fmt_type}
                        if date_input: spec["date_input"] = date_input
                        if dec_sep: spec["decimal_sep"] = dec_sep
                        if thou_sep: spec["thousands_sep"] = thou_sep
                        step = {"type": "format", "created_at": datetime.now().isoformat(), "params": {"formats": {fmt_col: spec}}}
                        _base_add_step(fmt_file, step)
                        st.success("Formato applicato!")
                        st.rerun()

        with col2:
            st.subheader("Anteprima Dati")
            prev_file_key = "prev_file"
            st.selectbox("Scegli file:", files_in_mem, key=prev_file_key)
            prev_file = st.session_state.get(prev_file_key)
            if prev_file:
                df_prev = preview_df(prev_file, limit=100)
                st.write("Mostrando le prime 100 righe.")
                st.dataframe(df_prev, use_container_width=True)

# --- TAB 3: FILTRI ---
with t_flt:
    if files_in_mem:
        st.subheader("Filtro Singolo")

        # Selettore file fuori dal form, così al cambio file resettiamo le colonne/valori
        sflt_file_key = "sflt_file"
        sflt_col_key = "sflt_col"
        sflt_kind_key = "sflt_kind"
        sflt_op_key = "sflt_op"
        sflt_v1_key = "sflt_v1"
        sflt_v2_key = "sflt_v2"

        st.selectbox("File:", files_in_mem, key=sflt_file_key)
        ensure_dep_updates("sflt", sflt_file_key, [sflt_col_key, sflt_kind_key, sflt_op_key, sflt_v1_key, sflt_v2_key])

        with st.form("single_filter_form"):
            flt_file = st.session_state.get(sflt_file_key)
            cols = get_cols(flt_file) if flt_file else []
            c1, c2, c3 = st.columns(3)
            c1.write("")  # spazio
            c2.selectbox("Colonna:", cols, key=sflt_col_key)
            c3.selectbox("Tipo dato:", ["number", "text", "date"], key=sflt_kind_key)

            flt_kind = st.session_state.get(sflt_kind_key, "text")
            ops = OPS_NUMBER if flt_kind == "number" else (OPS_DATE if flt_kind == "date" else OPS_TEXT)
            op_values = [v for (v, _) in ops]
            op_labels = {v: lab for (v, lab) in ops}

            c4, c5 = st.columns([1, 2])
            c4.selectbox("Operatore:", op_values, format_func=lambda v: op_labels.get(v, v), key=sflt_op_key)
            flt_op = st.session_state.get(sflt_op_key)

            # Input valori coerenti con tipo
            if flt_kind == "date":
                v1 = c5.date_input("Valore 1", key=sflt_v1_key)
                v2 = c5.date_input("Valore 2", key=sflt_v2_key) if flt_op == "between" else None
            else:
                v1 = c5.text_input("Valore 1", key=sflt_v1_key)
                v2 = c5.text_input("Valore 2", key=sflt_v2_key) if flt_op == "between" else None

            submitted = st.form_submit_button("✅ Applica Filtro Singolo")
            if submitted:
                flt_col = st.session_state.get(sflt_col_key)
                flt_kind = st.session_state.get(sflt_kind_key)
                flt_op = st.session_state.get(sflt_op_key)

                if not flt_file or not flt_col:
                    st.error("Seleziona file e colonna.")
                else:
                    val1 = v1.isoformat() if flt_kind == "date" else v1
                    val2 = v2.isoformat() if (flt_kind == "date" and v2) else v2
                    step = {"type":"filter", "created_at": datetime.now().isoformat(),
                            "params":{"col": flt_col, "kind": flt_kind, "op": flt_op, "v1": val1, "v2": val2}}
                    _base_add_step(flt_file, step)
                    st.success("Filtro applicato!")
                    st.rerun()

        st.divider()
        st.subheader("Rimuovi Duplicati")

        dd_file_key = "dd_file"
        dd_col_key = "dd_col"
        st.selectbox("File:", files_in_mem, key=dd_file_key)
        ensure_dep_updates("dd", dd_file_key, [dd_col_key])

        with st.form("dedupe_form"):
            dd_file = st.session_state.get(dd_file_key)
            dd_cols = get_cols(dd_file) if dd_file else []
            st.selectbox("Colonna chiave:", dd_cols, key=dd_col_key)
            submitted = st.form_submit_button("🧹 Rimuovi Duplicati")
            if submitted:
                dd_col = st.session_state.get(dd_col_key)
                if not dd_file or not dd_col:
                    st.error("Seleziona file e colonna chiave.")
                else:
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
            format_func=lambda x: "Mergia su file esistente (A)" if x=="existing" else "Crea NUOVO file in memoria",
            key="merge_mode"
        )

        new_name = st.text_input("Nome nuovo file:", "merged_1", key="merge_new_name") if merge_mode == "new" else None

        mA_key = "mA_file"
        mB_key = "mB_file"
        monA_key = "monA"
        monB_key = "monB"
        mhow_key = "mhow"
        mcoll_key = "mcoll"

        c1, c2 = st.columns(2)
        c1.selectbox("File A (Base):", files_in_mem, key=mA_key)
        c2.selectbox("File B (Da unire):", files_in_mem, key=mB_key)

        # Reset chiavi dipendenti se cambia A o B
        ensure_dep_updates("mergeA", mA_key, [monA_key])
        ensure_dep_updates("mergeB", mB_key, [monB_key])

        m_A = st.session_state.get(mA_key)
        m_B = st.session_state.get(mB_key)

        c3, c4 = st.columns(2)
        c3.selectbox("Chiave in A:", get_cols(m_A) if m_A else [], key=monA_key)
        c4.selectbox("Chiave in B:", get_cols(m_B) if m_B else [], key=monB_key)

        c5, c6 = st.columns(2)
        c5.selectbox("Tipo Join:", ["left", "right", "inner", "outer"], key=mhow_key)
        c6.selectbox(
            "Collisioni nomi:",
            ["suffix", "keep_left", "keep_right"],
            format_func=lambda x: "Aggiungi _B automatico" if x=="suffix" else x,
            key=mcoll_key
        )

        if st.button("🔗 Esegui Merge", type="primary"):
            m_on_A = st.session_state.get(monA_key)
            m_on_B = st.session_state.get(monB_key)
            m_how = st.session_state.get(mhow_key)
            m_coll = st.session_state.get(mcoll_key)

            if not m_A or not m_B or not m_on_A or not m_on_B:
                st.error("Seleziona File A, File B e le rispettive chiavi.")
            else:
                if merge_mode == "existing":
                    step = {"type": "merge", "created_at": datetime.now().isoformat(),
                            "params": {"right_file": m_B, "how": m_how, "left_on": m_on_A, "right_on": m_on_B, "collision": m_coll}}
                    _base_add_step(m_A, step)
                    st.success("Merge aggiunto come step!")
                    st.rerun()
                else:
                    nn = sanitize_name(new_name, "merged_1")
                    sqlA = get_sql(m_A)
                    step = {"type": "merge", "created_at": datetime.now().isoformat(),
                            "params": {"right_file": m_B, "how": m_how, "left_on": m_on_A, "right_on": m_on_B, "collision": m_coll}}
                    merged_sql = step_to_sql(m_A, step, sqlA)

                    con = db()
                    try:
                        tname = f"base_{nn}_{abs(hash((m_A,m_B,datetime.now().isoformat())))%10_000_000}"
                        tname = sanitize_name(tname)
                        con.execute(f"CREATE OR REPLACE TABLE {qident(tname)} AS {merged_sql};")
                        cols = [r[0] for r in con.execute(f"DESCRIBE {qident(tname)}").fetchall()]
                    finally:
                        con.close()

                    add_loaded_file(nn, tname, cols, path=Path(""), import_mode="smart")
                    st.success(f"Nuovo file {nn} creato!")
                    st.rerun()

# --- TAB 5: CALCOLI & IFS ---
with t_calc:
    if files_in_mem:
        st.subheader("Calcoli Multi-Colonna (Riga per Riga)")
        st.info("I vuoti valgono 0. Le operazioni vengono eseguite rigorosamente da sinistra a destra.")

        calc_file_key = "calc_file"
        calc_base_key = "calc_base"
        calc_next_key = "calc_next"
        calc_add_op_key = "calc_add_op"
        calc_out_key = "calc_out"

        c1, c2 = st.columns(2)
        c1.selectbox("File:", files_in_mem, key=calc_file_key)
        ensure_dep_updates("calc", calc_file_key, [calc_base_key, calc_next_key])

        calc_f = st.session_state.get(calc_file_key)
        cols_calc = get_cols(calc_f) if calc_f else []

        c2.selectbox("Colonna Base:", cols_calc, key=calc_base_key)
        calc_base = st.session_state.get(calc_base_key)

        with st.container(border=True):
            st.write("Aggiungi operazioni in sequenza:")
            cc1, cc2, cc3 = st.columns([2, 3, 1])
            cc1.selectbox(
                "Op:",
                ["add", "sub", "mul", "div"],
                key=calc_add_op_key,
                format_func=lambda x: {"add":"+ (Somma)","sub":"- (Sottrai)","mul":"* (Moltiplica)","div":"/ (Dividi)"}[x]
            )
            cc2.selectbox("Con colonna:", cols_calc, key=calc_next_key)

            if cc3.button("➕ Aggiungi Op"):
                op = st.session_state.get(calc_add_op_key)
                col = st.session_state.get(calc_next_key)
                if calc_f and calc_base and col:
                    st.session_state.calc_ops_list.append({"op": op, "col": col})
                    st.rerun()

            if st.session_state.calc_ops_list:
                formula = f"[{calc_base}]"
                for op in st.session_state.calc_ops_list:
                    formula += f" {op['op']} [{op['col']}]"
                st.code(formula, language="text")
                if st.button("Reset Operazioni"):
                    st.session_state.calc_ops_list = []
                    st.rerun()

        st.text_input("Nome nuova colonna:", "calc_row_1", key=calc_out_key)
        if st.button("✅ Esegui Calcolo", type="primary"):
            calc_out = st.session_state.get(calc_out_key)
            if not calc_f or not calc_base or not calc_out:
                st.error("Seleziona file, colonna base e nome colonna output.")
            else:
                step = {"type": "calc", "created_at": datetime.now().isoformat(),
                        "params": {"base_col": calc_base, "ops": st.session_state.calc_ops_list, "outcol": calc_out}}
                _base_add_step(calc_f, step)
                st.session_state.calc_ops_list = []
                st.success("Calcolo applicato!")
                st.rerun()

        st.divider()
        st.subheader("Aggregazione IFS (Es. Somma se, Conta se)")

        ifsA_key = "ifsA"
        ifsB_key = "ifsB"
        imA_key = "imA"
        imB_key = "imB"
        ifs_metric_key = "ifs_metric"
        ifs_valcol_key = "ifs_valcol"
        ifs_out_key = "ifs_out"

        c1, c2 = st.columns(2)
        c1.selectbox("File Destinazione (A):", files_in_mem, key=ifsA_key)
        c2.selectbox("File Ricerca (B):", files_in_mem, key=ifsB_key)

        ensure_dep_updates("ifsA", ifsA_key, [imA_key])
        ensure_dep_updates("ifsB", ifsB_key, [imB_key, ifs_valcol_key])

        ifs_A = st.session_state.get(ifsA_key)
        ifs_B = st.session_state.get(ifsB_key)

        with st.form("ifs_form"):
            c3, c4 = st.columns(2)
            c3.selectbox("Col. Match in A:", get_cols(ifs_A) if ifs_A else [], key=imA_key)
            c4.selectbox("Col. Match in B:", get_cols(ifs_B) if ifs_B else [], key=imB_key)

            c5, c6 = st.columns(2)
            c5.selectbox("Funzione:", ["countifs", "sumifs", "averageifs"], key=ifs_metric_key)
            metric = st.session_state.get(ifs_metric_key, "countifs")

            colsB = get_cols(ifs_B) if ifs_B else []
            if metric in ["sumifs", "averageifs"]:
                c6.selectbox("Colonna Valore (in B):", [""] + colsB, key=ifs_valcol_key)
            else:
                # anche se non serve, teniamo key coerente
                if ifs_valcol_key in st.session_state:
                    pass
                c6.selectbox("Colonna Valore (in B):", [""] + colsB, key=ifs_valcol_key)

            st.text_input("Nome nuova colonna in A:", "ifs_result_1", key=ifs_out_key)

            submitted = st.form_submit_button("✅ Esegui IFS")
            if submitted:
                ifs_match_a = st.session_state.get(imA_key)
                ifs_match_b = st.session_state.get(imB_key)
                metric = st.session_state.get(ifs_metric_key)
                val_col = st.session_state.get(ifs_valcol_key, "")
                ifs_out = st.session_state.get(ifs_out_key)

                if not ifs_A or not ifs_B or not ifs_match_a or not ifs_match_b or not ifs_out:
                    st.error("Seleziona file A/B, colonne match e nome colonna output.")
                elif metric in ["sumifs", "averageifs"] and not val_col:
                    st.error("Seleziona una colonna valore per SUM/AVERAGE.")
                else:
                    step = {"type":"ifs", "created_at": datetime.now().isoformat(),
                            "params":{"lookup_file": ifs_B, "match_a_col": ifs_match_a, "match_b_col": ifs_match_b,
                                      "metric": metric, "value_col": val_col, "outcol": ifs_out, "conds": st.session_state.run_conds}}
                    _base_add_step(ifs_A, step)
                    st.success("IFS applicato!")
                    st.rerun()

# --- TAB 6: METRICHE ---
with t_metr:
    if files_in_mem:
        st.subheader("Calcola Metrica Veloce (Senza creare Step)")

        met_file_key = "met_file"
        met_col_key = "met_col"
        st.selectbox("File:", files_in_mem, key=met_file_key)
        ensure_dep_updates("met", met_file_key, [met_col_key])

        met_file = st.session_state.get(met_file_key)
        cols_met = get_cols(met_file) if met_file else []

        c1, c2 = st.columns(2)
        c1.selectbox("Colonna:", cols_met, key=met_col_key)
        c2.selectbox("Metrica:", ["sum", "count", "counta", "average", "median", "quartile", "percentile"], key="met_kind")

        met_kind = st.session_state.get("met_kind")
        q_val = None
        if met_kind in ["quartile", "percentile"]:
            q_val = st.text_input("Q/Percentile (es. 0.25):", "0.25", key="met_q")

        if st.button("📌 Calcola", type="primary"):
            try:
                met_col = st.session_state.get(met_col_key)
                if not met_file or not met_col:
                    st.error("Seleziona file e colonna.")
                else:
                    sql = get_sql(met_file)
                    val = compute_metric_sql(met_file, sql, met_col, met_kind, q=float(q_val) if q_val else None)
                    st.success(f"Risultato **{met_kind.upper()}** su {met_col}: **{val}**")
            except Exception as e:
                st.error(f"Errore: {e}")

# --- TAB 7: EXPORT ---
with t_exp:
    if files_in_mem:
        st.subheader("Esporta Risultati (CSV)")
        st.info("Export scalabile: DuckDB scrive su file via COPY senza caricare tutto in RAM.")

        exp_file_key = "exp_file"
        st.selectbox("File da esportare:", files_in_mem, key=exp_file_key)
        exp_file = st.session_state.get(exp_file_key)

        exp_name_key = "exp_name"
        st.text_input("Nome file output:", f"export_{exp_file}" if exp_file else "export", key=exp_name_key)

        if st.button("Genera File CSV"):
            if not exp_file:
                st.error("Seleziona un file.")
            else:
                sql = get_sql(exp_file)
                exp_name = st.session_state.get(exp_name_key) or f"export_{exp_file}"
                out_path = Path(st.session_state.temp_dir) / f"{sanitize_name(exp_name)}.csv"

                con = db()
                try:
                    con.execute(f"""
                        COPY (SELECT * FROM ({sql}))
                        TO {sql_lit(str(out_path))}
                        (HEADER, DELIMITER ',');
                    """)
                finally:
                    con.close()

                with open(out_path, "rb") as f:
                    data = f.read()

                st.download_button(
                    label=f"⬇️ Scarica {sanitize_name(exp_name)}.csv",
                    data=data,
                    file_name=f"{sanitize_name(exp_name)}.csv",
                    mime="text/csv",
                    type="primary"
                )
