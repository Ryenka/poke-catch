"""Funciones compartidas por los notebooks de Poké-Catch.

- Carga de los CSV de PokeAPI.
- Construcción de las características de cada especie.
- Fórmula de captura oficial (Gen V en adelante) y reglas de las balls.
- Simulación de los intentos de captura (una fila = un intento).
- Matriz de variables para los modelos.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "datos" / "raw"
DIR_PROC = RAIZ / "datos" / "procesados"
URL_BASE = "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/"
TABLAS = ["pokemon_species", "pokemon", "pokemon_stats", "stats", "pokemon_types", "types",
          "pokemon_egg_groups", "egg_groups", "growth_rates"]

SEMILLA = 42
CLASES = ["no captura", "captura"]          # 0 y 1

# Colores y estilo de los gráficos
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#b4b2a9"
ESTILO = {
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.titleweight": "bold", "axes.titlesize": 11,
    "font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white",
}


# --------------------------------------------------------------------------- carga
def cargar_tabla(nombre):
    """Lee un CSV de PokeAPI desde datos/raw; si no está, lo descarga."""
    ruta = DIR_RAW / f"{nombre}.csv"
    if not ruta.exists():
        DIR_RAW.mkdir(parents=True, exist_ok=True)
        pd.read_csv(URL_BASE + f"{nombre}.csv").to_csv(ruta, index=False, encoding="utf-8")
    return pd.read_csv(ruta, encoding="utf-8")


def cargar_crudos():
    return {t: cargar_tabla(t) for t in TABLAS}


# ----------------------------------------------------------------- especies
EXP_NIVEL_100 = {"slow": 1_250_000, "medium": 1_000_000, "fast": 800_000, "medium-slow": 1_059_860,
                 "slow-then-very-fast": 600_000, "fast-then-very-slow": 1_640_000}
ULTRAENTES = {793, 794, 795, 796, 797, 798, 799, 803, 804, 805, 806}


def construir_especies(t=None):
    """Una fila por especie (forma por defecto) con todas las características candidatas."""
    t = t or cargar_crudos()
    df = t["pokemon_species"][[
        "id", "identifier", "generation_id", "evolves_from_species_id", "evolution_chain_id",
        "gender_rate", "capture_rate", "base_happiness", "is_baby", "hatch_counter",
        "growth_rate_id", "is_legendary", "is_mythical"]].rename(columns={"id": "species_id"})

    # Forma por defecto: peso, altura y experiencia base
    pk = t["pokemon"][t["pokemon"].is_default == 1][["id", "species_id", "height", "weight", "base_experience"]]
    df = df.merge(pk.rename(columns={"id": "pokemon_id"}), on="species_id", validate="one_to_one")
    df["altura_m"] = df.height / 10          # PokeAPI entrega decímetros
    df["peso_kg"] = df.weight / 10           # y hectogramos
    df["imc"] = df.peso_kg / df.altura_m ** 2
    df["log_peso"] = np.log10(df.peso_kg)
    df["log_altura"] = np.log10(df.altura_m)

    # Estadísticas base (tabla larga -> ancha)
    st = t["pokemon_stats"].merge(t["stats"][["id", "identifier"]], left_on="stat_id", right_on="id")
    st = st.pivot(index="pokemon_id", columns="identifier", values="base_stat")
    st.columns = ["st_" + c.replace("-", "_") for c in st.columns]
    st["bst"] = st.sum(axis=1)
    df = df.merge(st, left_on="pokemon_id", right_index=True)

    # Tipos (one-hot)
    ti = t["pokemon_types"].merge(t["types"][["id", "identifier"]], left_on="type_id", right_on="id")
    ti = ti[ti.pokemon_id.isin(df.pokemon_id)]
    df = df.merge(pd.crosstab(ti.pokemon_id, ti.identifier).add_prefix("tipo_"),
                  left_on="pokemon_id", right_index=True)
    df["n_tipos"] = df.filter(like="tipo_").sum(axis=1)

    # Grupos huevo (multi-hot)
    hu = t["pokemon_egg_groups"].merge(t["egg_groups"], left_on="egg_group_id", right_on="id")
    oh = pd.crosstab(hu.species_id, hu.identifier).add_prefix("huevo_")
    oh.columns = [c.replace("-", "_") for c in oh.columns]
    df = df.merge(oh, left_on="species_id", right_index=True, how="left")
    df[oh.columns] = df[oh.columns].fillna(0).astype(int)

    # Evolución
    padre = df.set_index("species_id").evolves_from_species_id

    def etapa(sid):
        n = 1
        while pd.notna(padre.get(sid)):
            sid, n = int(padre[sid]), n + 1
        return n

    df["etapa_evolutiva"] = df.species_id.map(etapa)
    con_hijos = set(df.evolves_from_species_id.dropna().astype(int))
    df["es_final"] = (~df.species_id.isin(con_hijos)).astype(int)
    df["etapas_cadena"] = df.groupby("evolution_chain_id").etapa_evolutiva.transform("max")
    df["etapas_restantes"] = df.etapas_cadena - df.etapa_evolutiva
    df["especies_en_cadena"] = df.groupby("evolution_chain_id").species_id.transform("count")
    df["sin_evolucion"] = (df.especies_en_cadena == 1).astype(int)

    # Crianza
    ritmo = df.growth_rate_id.map(t["growth_rates"].set_index("id").identifier)
    df["exp_nivel_100"] = ritmo.map(EXP_NIVEL_100)
    df["sin_genero"] = (df.gender_rate == -1).astype(int)
    df["pct_hembra"] = np.where(df.gender_rate == -1, 0, df.gender_rate / 8)

    # Rareza
    df["es_ultraente"] = df.species_id.isin(ULTRAENTES).astype(int)
    return df


def variables_especie(df):
    """Variables de la especie que entran al modelo (ver decisiones en el notebook 01)."""
    return (
        ["st_hp", "st_attack", "st_defense", "st_special_attack", "st_special_defense", "st_speed", "bst",
         "base_experience",
         "is_legendary", "is_mythical", "is_baby", "es_ultraente",
         "etapa_evolutiva", "es_final", "etapas_restantes", "especies_en_cadena", "sin_evolucion",
         "exp_nivel_100", "hatch_counter", "sin_genero", "pct_hembra", "base_happiness",
         "log_altura", "log_peso", "n_tipos"]
        + sorted(c for c in df.columns if c.startswith("tipo_"))
        + sorted(c for c in df.columns if c.startswith("huevo_"))
    )


# ------------------------------------------------------- fórmula de captura
BONUS_ESTADO = {"ninguno": 1.0, "dormido": 2.5, "congelado": 2.5,
                "paralizado": 1.5, "envenenado": 1.5, "quemado": 1.5}
ESTADOS = list(BONUS_ESTADO)
P_ESTADO = [0.60, 0.12, 0.03, 0.15, 0.05, 0.05]          # frecuencia con que se sortea cada estado
BALLS = ["Poké Ball", "Super Ball", "Ultra Ball", "Malla Ball", "Rápida Ball",
         "Ocaso Ball", "Veloz Ball", "Turno Ball"]


def prob_captura(ratio, hp_pct, bonus_ball, estado):
    """Probabilidad de captura, Gen V en adelante, con el HP en % del máximo."""
    a = (300 - 2 * hp_pct) * ratio * bonus_ball / 300 * BONUS_ESTADO[estado]
    if a >= 255:
        return 1.0
    b = 65536 / (255 / a) ** (3 / 16)
    return (b / 65536) ** 4


def multiplicador_ball(ball, agua_o_bicho, velocidad, turno, noche_o_cueva, ultraente=False):
    """Multiplicador público de cada ball (Gen VII en adelante)."""
    if ultraente:
        return 0.1
    if ball == "Poké Ball":
        return 1.0
    if ball == "Super Ball":
        return 1.5
    if ball == "Ultra Ball":
        return 2.0
    if ball == "Malla Ball":
        return 3.5 if agua_o_bicho else 1.0
    if ball == "Rápida Ball":
        return 4.0 if velocidad >= 100 else 1.0
    if ball == "Ocaso Ball":
        return 3.0 if noche_o_cueva else 1.0
    if ball == "Veloz Ball":
        return 5.0 if turno == 1 else 1.0
    if ball == "Turno Ball":
        return min(4.0, 1 + 0.3 * (turno - 1))
    raise ValueError(ball)


def _mult_fila(esp, ball, turno, noche):
    return multiplicador_ball(ball, bool(esp["tipo_water"] or esp["tipo_bug"]), esp["st_speed"],
                              turno, noche, bool(esp["es_ultraente"]))


# ------------------------------------------------------------- simulación
def simular_intentos(especies, n_por_especie=40, semilla=SEMILLA):
    """Genera los registros: n intentos por especie con contexto al azar.

    La clase se sortea con la probabilidad que da la fórmula oficial usando el
    capture_rate real de la especie. Devuelve solo las columnas del lanzamiento,
    la probabilidad real y la clase; las de la especie se unen por species_id.
    """
    rng = np.random.default_rng(semilla)
    filas = []
    for _, esp in especies.iterrows():
        for _ in range(n_por_especie):
            hp = int(rng.integers(1, 101))
            estado = str(rng.choice(ESTADOS, p=P_ESTADO))
            ball = str(rng.choice(BALLS))
            turno = int(rng.integers(1, 11))
            noche = bool(rng.random() < 0.35)
            p = prob_captura(esp.capture_rate, hp, _mult_fila(esp, ball, turno, noche), estado)
            filas.append((esp.species_id, esp.identifier, hp, estado, ball, turno, int(noche),
                          round(p, 6), int(rng.random() < p)))
    return pd.DataFrame(filas, columns=["species_id", "identifier", "hp_pct", "estado", "ball", "turno",
                                        "noche_o_cueva", "p_real", "captura"])


def cargar_dataset():
    """Devuelve (especies, intentos). Usa datos/procesados si existe; si no, lo genera."""
    f_esp, f_int = DIR_PROC / "especies.csv", DIR_PROC / "intentos.csv"
    if f_esp.exists() and f_int.exists():
        return pd.read_csv(f_esp, encoding="utf-8"), pd.read_csv(f_int, encoding="utf-8")
    especies = construir_especies()
    intentos = simular_intentos(especies)
    DIR_PROC.mkdir(parents=True, exist_ok=True)
    especies.to_csv(f_esp, index=False, encoding="utf-8")
    intentos.to_csv(f_int, index=False, encoding="utf-8")
    return especies, intentos


# ------------------------------------------------------- matriz del modelo
def unir(especies, intentos):
    """Une cada intento con las características de su especie."""
    cols = ["species_id", "evolution_chain_id", "capture_rate", "generation_id"] + variables_especie(especies)
    return intentos.merge(especies[cols], on="species_id", validate="many_to_one")


def variables_lanzamiento(datos, con_multiplicador=False):
    """Variables del lanzamiento, con estado y ball en one-hot."""
    X = pd.concat([
        datos[["hp_pct", "turno", "noche_o_cueva"]],
        pd.get_dummies(datos.estado, prefix="estado").reindex(columns=[f"estado_{e}" for e in ESTADOS], fill_value=0).astype(int),
        pd.get_dummies(datos.ball, prefix="ball").reindex(columns=[f"ball_{b}" for b in BALLS], fill_value=0).astype(int),
    ], axis=1)
    if con_multiplicador:
        X["multiplicador_ball"] = [
            multiplicador_ball(b, bool(w or g), v, t, bool(n), bool(u))
            for b, w, g, v, t, n, u in zip(datos.ball, datos.tipo_water, datos.tipo_bug, datos.st_speed,
                                           datos.turno, datos.noche_o_cueva, datos.es_ultraente)]
    return X
