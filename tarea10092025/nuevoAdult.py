# -*- coding: utf-8 -*-
"""
Imputación de 'native-country' (nacionalidad) en el dataset Adult (UCI)
usando máxima verosimilitud condicional (mínima sorpresa). SOLO imputación.

Salidas:
- adult_imputado.csv (dataset completo con 'native-country' ya imputado)
- (opcional) nacionalidades_sustituidas.csv con [index_original, nacionalidad_asignada, metodo]

Cómo usar (dos opciones):
1) Con rutas fijas: define DATA_PATH y OUTDIR abajo y ejecuta:  python solo_imputacion_adult.py
2) Con CLI:  python solo_imputacion_adult.py --data "ruta/a/adult.data" --outdir "carpeta_salida"
"""

from pathlib import Path
import argparse
from collections import Counter
import pandas as pd

# ======= Config por defecto (puedes editar estas dos líneas o usar la CLI) =======
DATA_PATH_DEFAULT = Path("adult.data")         # Cambia a tu ruta si quieres ejecución directa
OUTDIR_DEFAULT    = Path("./salida_imputacion_min")
# ================================================================================

ADULT_COLUMNS = [
    "age","workclass","fnlwgt","education","education-num",
    "marital-status","occupation","relationship","race","sex",
    "capital-gain","capital-loss","hours-per-week","native-country","income"
]
TARGET = "native-country"

# -------------------- Lectura --------------------
def load_adult(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None, names=ADULT_COLUMNS,
        na_values=["?"], skipinitialspace=True, dtype=str, encoding="utf-8"
    )
    # limpiar espacios
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

# -------------------- Núcleo de imputación --------------------
def _conditional_candidates(df_known: pd.DataFrame, row: pd.Series, attrs: list) -> Counter:
    mask = pd.Series(True, index=df_known.index)
    for a in attrs:
        mask &= (df_known[a] == row[a])
    subset = df_known.loc[mask, TARGET].dropna()
    return Counter(subset.values)

def _choose_value(candidates: Counter, global_counter: Counter):
    if sum(candidates.values()) == 0:
        if len(global_counter) == 0:
            return None, "fallback_empty"
        return max(global_counter.items(), key=lambda kv: kv[1])[0], "fallback_global_mode"
    return max(candidates.items(), key=lambda kv: kv[1])[0], "conditional_mode"

def impute_once(df: pd.DataFrame, conditioning_order: list):
    df_new = df.copy()
    df_known = df_new[df_new[TARGET].notna()]
    global_counter = Counter(df_known[TARGET].values)
    changes = []

    to_impute = df_new[df_new[TARGET].isna()]
    if len(to_impute) == 0:
        return df_new, changes

    for idx, row in to_impute.iterrows():
        chosen, source = None, None
        for attrs in conditioning_order:
            cand_counts = _conditional_candidates(df_known, row, attrs)
            cand, src = _choose_value(cand_counts, global_counter)
            if cand is not None and (sum(cand_counts.values()) > 0 or src == "fallback_global_mode"):
                chosen = cand
                source = f"cond({'+'.join(attrs)})" if sum(cand_counts.values())>0 else src
                break
        if chosen is None:  # última red de seguridad
            chosen = max(global_counter.items(), key=lambda kv: kv[1])[0] if len(global_counter)>0 else "United-States"
            source = "ultimate_global_mode"
        df_new.at[idx, TARGET] = chosen
        changes.append({"index_original": int(idx), "nacionalidad_asignada": chosen, "metodo": source})
    return df_new, changes

def impute_native_country(df: pd.DataFrame, max_iters: int = 3):
    """Devuelve (df_imputado, tabla_cambios, stop_reason, converged)"""
    conditioning_order = [
        ["race","sex","education","occupation","marital-status","relationship","income"],
        ["race","sex","education","occupation","marital-status"],
        ["race","sex","education"],
        ["race","sex"],
        ["race"],
        []  # global (moda)
    ]
    all_changes = []
    df_iter = df.copy()
    converged = False
    stop_reason = ""
    for it in range(1, max_iters+1):
        before = df_iter[TARGET].copy()
        df_iter, changes = impute_once(df_iter, conditioning_order)
        if it == 1:
            all_changes.extend(changes)   # almacenamos qué filas se imputaron
        if before.equals(df_iter[TARGET]):
            converged = True
            stop_reason = f"Sin cambios en iteración {it}. Proceso detenido."
            break
    if not converged:
        stop_reason = f"Iteraciones máximas ({max_iters}) alcanzadas."
    return df_iter, pd.DataFrame(all_changes), stop_reason, converged

# -------------------- Script/CLI --------------------
def main():
    p = argparse.ArgumentParser(description="Solo imputación de 'native-country' en Adult (UCI).")
    p.add_argument("--data", type=str, default=str(DATA_PATH_DEFAULT), help="Ruta al adult.data")
    p.add_argument("--outdir", type=str, default=str(OUTDIR_DEFAULT), help="Carpeta de salida")
    p.add_argument("--save-table", action="store_true", help="Guardar CSV con filas imputadas (opcional)")
    args = p.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_adult(data_path)
    total_missing = int(df[TARGET].isna().sum())

    df_imp, tabla, stop_reason, converged = impute_native_country(df, max_iters=3)

    # Guardar dataset imputado
    out_dataset = outdir / "adult_imputado.csv"
    df_imp.to_csv(out_dataset, index=False, encoding="utf-8")

    # (Opcional) guardar tabla de imputaciones de la 1ª pasada
    if args.save_table and not tabla.empty:
        out_table = outdir / "nacionalidades_sustituidas.csv"
        tabla.sort_values("index_original").to_csv(out_table, index=False, encoding="utf-8")
        print(f"Tabla de imputaciones: {out_table}")

    print("=== Resumen ===")
    print(f"Ausentes originales en '{TARGET}': {total_missing}")
    print(f"Convergencia: {converged} | {stop_reason}")
    print(f"Dataset imputado: {out_dataset}")

if __name__ == "__main__":
    main()
