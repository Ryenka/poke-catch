# Poké-Catch

Proyecto semestral de minería de datos. **Técnica asignada: clasificación.**

**Integrantes:** Rayen Ancamilla - Esteban Cancino

## Descripción

Un jugador que lanza una Poké Ball no sabe si el lanzamiento va a funcionar. El resultado depende del Pokémon (cada especie tiene una dificultad de captura que los diseñadores fijaron sin criterios públicos) y de la situación del combate: HP restante, estado alterado, ball usada, turno y lugar.

**Problema de predicción:** dado un intento de captura, predecir su clase, **captura** o **no captura**.

**Qué representa cada registro:** una fila es un intento de captura. La fila describe al Pokémon (estadísticas, evolución, crianza, rareza, tipo, peso y altura) y al lanzamiento (HP restante, estado, ball, turno, noche o cueva). La clase es el resultado del intento.

**Usuarios destinatarios:** jugadores casuales, speedrunners y coleccionistas de la Pokédex que quieren saber si conviene lanzar y con qué ball.

## Datos

| | |
|---|---|
| Fuente | [PokeAPI](https://pokeapi.co), archivos CSV de su [repositorio oficial](https://github.com/PokeAPI/pokeapi/tree/master/data/v2/csv) |
| Cobertura | 1025 especies, generaciones 1 a 9 |
| Fecha de descarga | 14 de noviembre de 2026 |
| Registros | 41 000 intentos de captura (40 por especie), con 75 variables: 58 del Pokémon y 17 del lanzamiento |
| Clase | 48,4 % captura, 51,6 % no captura |

No existe un registro público de lanzamientos reales. Los intentos se **simulan** con la fórmula de captura oficial de los juegos (Gen V en adelante): se sortea el contexto del lanzamiento, la fórmula calcula la probabilidad real con el ratio de captura de la especie, y la clase se sortea con esa probabilidad. **El modelo nunca ve el ratio de captura**: tiene que deducir la dificultad del Pokémon desde sus características.

Los datos están incluidos en el repositorio:

- `datos/raw/`: los 9 CSV de PokeAPI, sin modificar.
- `datos/procesados/`: `especies.csv` (una fila por especie) e `intentos.csv` (una fila por intento). Los genera el notebook 01 con una semilla fija.

Para volver a descargar los datos de origen: `python src/descargar_datos.py`.

## Estructura del repositorio

```
├── README.md
├── requirements.txt
├── datos/
│   ├── raw/                  CSV de PokeAPI
│   └── procesados/           dataset de especies y de intentos
├── notebooks/
│   ├── 01_analisis_exploratorio.ipynb
│   └── 02_experimento_comparativo.ipynb
└── src/
    ├── pokecatch.py          carga, variables, fórmula de captura y simulación
    └── descargar_datos.py
```

## Instrucciones de ejecución

Requiere Python 3.10 o superior.

```bash
git clone <URL del repositorio>
cd poke-catch

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
jupyter notebook
```

Ejecutar los notebooks en orden, con «Run All»:

1. `notebooks/01_analisis_exploratorio.ipynb` (unos 20 segundos). Genera `datos/procesados/`.
2. `notebooks/02_experimento_comparativo.ipynb` (2 a 3 minutos).

No se necesita conexión a internet: los datos ya están en el repositorio. Los notebooks se entregan ejecutados, así que los resultados se pueden revisar sin correrlos.

## Principales resultados

### Análisis exploratorio

- **Calidad:** no hay duplicados en los datos de origen. `habitat` falta en el 62 % de las especies y se descartó. Se trabaja con la forma por defecto de cada especie.
- **Lo que más pesa en el resultado** es el poder del Pokémon: el quintil más débil se captura el 75 % de las veces y el más fuerte, el 19 %. Le siguen el HP restante, el estado alterado y la ball.
- **Las balls especiales dependen de una condición** (por ejemplo, la Malla Ball captura el 77 % con Pokémon de agua o bicho y el 36 % con el resto). Esto motivó probar la variable `multiplicador_ball`.
- **Peso y altura** son muy asimétricos y se pasaron a logaritmo. Su relación con la clase es más débil que la del poder.
- **La clase tiene azar:** el 38 % de los intentos tiene una probabilidad real entre 30 % y 70 %. Ni conociendo la probabilidad exacta se acierta más del **76,8 %**. Ese es el techo contra el que se comparan los modelos.

### Experimento comparativo

**Métrica principal:** F1 macro, porque ambos errores le cuestan al jugador (lanzar y fallar desperdicia una ball; no lanzar cuando sí resultaba desperdicia turnos). **Desempate:** Brier, porque la predicción se comunica como probabilidad.

**Procedimiento:** se reservó un 20 % de las familias evolutivas como conjunto de prueba. Las alternativas se compararon con validación cruzada de 5 particiones sobre el 80 % restante, también separando por familia evolutiva.

| Alternativa | F1 macro | Brier | AUC |
|---|---|---|---|
| Clase mayoritaria | 0,342 | 0,481 | 0,500 |
| Regresión logística · solo lanzamiento | 0,607 | 0,232 | 0,653 |
| Árbol de decisión · A | 0,681 | 0,201 | 0,753 |
| Regresión logística · A | 0,702 | 0,193 | 0,775 |
| Random Forest · A | 0,707 | 0,189 | 0,784 |
| Árbol de decisión · B | 0,709 | 0,189 | 0,783 |
| Regresión logística · B | 0,725 | 0,182 | 0,802 |
| Gradient Boosting · A · 31 hojas | 0,728 | 0,178 | 0,806 |
| Random Forest · B | 0,733 | 0,179 | 0,810 |
| Gradient Boosting · B · 4 hojas | 0,734 | 0,176 | 0,811 |
| **Gradient Boosting · B · 8 hojas** | **0,734** | **0,175** | **0,813** |
| Gradient Boosting · B · 31 hojas | 0,735 | 0,175 | 0,812 |
| _Techo (fórmula con el ratio real)_ | _0,765_ | _0,153_ | _0,853_ |

A = variables del Pokémon y del lanzamiento. B = A más el multiplicador de la ball.

**Modelo seleccionado: Gradient Boosting · B · 8 hojas.** Cuatro alternativas quedaron empatadas en F1 macro (a menos de una desviación entre particiones). Entre ellas se eligió la de mejor Brier, que además es la más simple.

**Evaluación final en el conjunto de prueba** (8 280 intentos de 108 familias evolutivas nunca vistas):

| | Precisión | Recobrado | F1 | Registros |
|---|---|---|---|---|
| no captura | 0,738 | 0,809 | 0,772 | 4 186 |
| captura | 0,784 | 0,706 | 0,743 | 4 094 |
| **F1 macro** | | | **0,758** | 8 280 |

Matriz de confusión (filas: clase real; columnas: clase predicha):

| | Predice no captura | Predice captura |
|---|---|---|
| **Real no captura** | 3 388 | 798 |
| **Real captura** | 1 202 | 2 892 |

Exactitud 75,8 %, AUC 0,848. El techo en el mismo conjunto es 0,779 de F1 macro.

**Análisis de errores:**

- El 73 % de los errores se debe al azar del propio juego: salió el resultado menos probable.
- El 27 % restante (6,4 % de los intentos) es atribuible al modelo. Se concentra en especies cuyo ratio de captura rompe el patrón general, como Mawile (débil pero difícil) o Watchog (evolución final pero muy fácil).
- El modelo es conservador: comete más falsos negativos que falsos positivos. Cuando dice *captura*, acierta el 78 %.
- Las probabilidades están bien calibradas. Cuando el modelo está seguro (≥ 80 % o ≤ 20 %) acierta el 93 %.

## Limitaciones

- Los registros son simulados. El modelo aprende el comportamiento del juego, no el de jugadores reales.
- Se usa la fórmula continua de la Gen V en adelante, sin captura crítica ni los bonus por nivel de la Gen 8.
- Solo se incluyen 8 balls, con los multiplicadores de la Gen VII en adelante.
- La predicción no es infalible y debe comunicarse como probabilidad.
