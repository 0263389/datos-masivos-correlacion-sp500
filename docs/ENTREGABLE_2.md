# Entregable 2 — Del laptop al data lake

**Materia:** Datos Masivos · **Profesor:** Oscar Enrique Landivar Carrasco

Este documento cuenta qué se construyó en el Entregable 2, por qué, y qué se encontró.
Para el detalle del Entregable 1, ver el [README](../README.md) y
[`notebooks/01_entregable1.ipynb`](../notebooks/01_entregable1.ipynb).

---

## El punto de partida: una pregunta sin responder

El Entregable 1 midió que las acciones del mismo sector GICS correlacionan **0.440** entre sí,
contra **0.316** entre sectores distintos. Pero dejó algo sin explicar:

> La correlación entre sectores **sin relación** —un banco y una aerolínea— salió **0.316**,
> no cero.

Si dos industrias no tienen nada en común, su correlación debería rondar el cero. Que no lo haga
significa que **hay algo moviendo a todas las acciones a la vez**. El Entregable 1 detectó el
síntoma; el Entregable 2 lo identifica, lo mide y lo separa.

---

## Los tres cambios

### 1. Escala: de 503 a 1,503 empresas

| | Entregable 1 | Entregable 2 |
|---|---|---|
| Universo | S&P 500 | **S&P Composite 1500** (500 large + 400 mid + 600 small) |
| Empresas | 503 | **1,503** (1,502 con datos) |
| Filas | 1.5M celdas | **4,104,670** |
| Formato | Ancho (una columna por empresa) | **Largo** (`fecha, ticker, precio`) |

El salto de volumen importa menos que **la columna nueva: `cap_tier`** (large / mid / small). Con
el S&P 500 solo, preguntar si el tamaño de la empresa cambia su relación con el mercado era
imposible — todas sus empresas son grandes.

Las tres listas vienen de Wikipedia y comparten las mismas columnas, así que el scraping del
Entregable 1 sirvió sin modificarse.

### 2. Infraestructura: de archivos a un data lake

| | Entregable 1 | Entregable 2 |
|---|---|---|
| Almacenamiento | Parquet suelto en `data/` | **Tablas Delta en Unity Catalog** |
| Motor | pandas | **PySpark** |
| Dónde corre | Laptop | **Databricks** (Free Edition, serverless) |
| Organización | Una carpeta | **Arquitectura medallion** |

**Las tres capas y su regla:**

| Capa | Contenido | Regla |
|---|---|---|
| **Bronze** | Dato crudo tal como llegó de la fuente | **No se limpia nada**, ni los errores |
| **Silver** | Panel balanceado, tickers corruptos fuera, rendimientos log | Aquí vive toda la limpieza |
| **Gold** | Betas, residuales, cohesión, riesgo de portafolio | Solo resultados |

Que Bronze conserve el dato sin tocar no es purismo: permite **recalcular Silver sin volver a
descargar**. Se aprovechó dos veces durante el desarrollo — al recalibrar el umbral de historia y
al corregir el criterio de datos corruptos.

### 3. Análisis: descomponer el movimiento

```
movimiento de la acción = beta × movimiento del mercado + residual
                          └──── lo que comparte ────┘    └─ lo suyo ─┘
```

| Medida | Qué mide |
|---|---|
| **Beta** | Cuánto amplifica. Beta 1.5 = se mueve 1.5% cuando el mercado sube 1% |
| **R²** | Qué *fracción* de su movimiento explica el mercado. Mide **dependencia** |
| **Residual** | Lo que queda tras quitar el mercado. Lo genuinamente propio |

---

## Los seis notebooks

| Notebook | Qué hace |
|---|---|
| [`01_bronze_ingesta`](../databricks/01_bronze_ingesta.py) | Las 3 fuentes → tablas Delta crudas, particionadas por año |
| [`02_silver_limpieza`](../databricks/02_silver_limpieza.py) | Auditoría y rendimientos log con *window functions* |
| [`03_gold_analisis`](../databricks/03_gold_analisis.py) | Factor de mercado, betas, R², residuales |
| [`04_benchmark_pandas_vs_spark`](../databricks/04_benchmark_pandas_vs_spark.py) | ¿Desde qué volumen conviene Spark? |
| [`05_motor_de_riesgo`](../databricks/05_motor_de_riesgo.py) | Descomposición del riesgo de un portafolio |
| [`06_regimenes_de_crisis`](../databricks/06_regimenes_de_crisis.py) | La cohesión del mercado en el tiempo |

La matemática de riesgo no vive en los notebooks: está en
[`src/risk.py`](../src/risk.py) con **25 pruebas automáticas** en
[`tests/test_risk.py`](../tests/test_risk.py). Los notebooks solo orquestan y narran.

---

## Hallazgos

### 1. Las empresas chicas no dependen menos del mercado — reaccionan más

| Tamaño | Empresas | Beta | R² | Volatilidad diaria |
|---|---|---|---|---|
| Large | 455 | 0.859 | 0.323 | 0.0200 |
| Mid | 317 | 1.060 | 0.359 | 0.0240 |
| Small | 439 | 1.102 | 0.319 | 0.0264 |

El **R² es prácticamente idéntico** entre grandes y chicas, pero el **beta crece de forma
monótona** conforme baja el tamaño. Cuando el mercado se mueve 1%, una small cap se mueve 1.10% y
una large cap 0.86% — pero la *fracción* de su movimiento que el mercado explica es la misma.

Ninguna de las dos hipótesis iniciales era correcta: las small caps no son más ni menos
*dependientes*, son más *reactivas*.

El caso extremo lo ilustra: **CHRD con beta 1.99 y R² 0.047**. Amplifica el doble al mercado los
días que lo sigue, pero el 95% de lo que hace es por razones propias.

### 2. El efecto sector del Entregable 1 era real

| Tamaño | Correlación con el mercado | Correlación con su sector (residual) |
|---|---|---|
| Large | 0.537 | **0.379** |
| Mid | 0.535 | 0.356 |
| Small | 0.518 | 0.303 |

Tras quitar el factor de mercado, el vínculo sectorial **sobrevive** entre 0.30 y 0.38. La
correlación de 0.316 del Entregable 1 no era solo un reflejo del movimiento común: había un efecto
sectorial genuino debajo.

Y es **más fuerte en las grandes** que en las chicas.

### 3. Un motor de riesgo: la diversificación no se cuenta en posiciones

| Portafolio | Volatilidad anual | Ratio de diversificación |
|---|---|---|
| 10 bancos, pesos iguales | **27.36%** | 1.12 |
| 10 sectores distintos, pesos iguales | **15.59%** | 1.53 |

Mismo número de posiciones, mismo peso en cada una, y **43% menos volatilidad**. La diferencia
completa está en las correlaciones.

El ratio de diversificación de los bancos es 1.12 de un máximo teórico de 3.16: diez bancos se
comportan casi como **una sola posición**.

Con 1,211 activos hay 733,866 parámetros de covarianza que estimar y 2,903 observaciones. El
shrinkage de Ledoit-Wolf mejoró el número de condición de **232,640 a 14,910** — 16 veces.

### 4. Restringir el problema mejoró más que mejorar el modelo

Prueba fuera de muestra con 150 activos (2,032 días de entrenamiento, 871 de prueba):

| Estrategia | Vol. real | Prometida | En corto |
|---|---|---|---|
| Mín. varianza muestral (libre) | 13.44% | 11.45% | −155% |
| Mín. varianza encogida (libre) | 13.29% | 11.84% | −144% |
| **Mín. varianza muestral (solo largo)** | **12.11%** | 15.40% | 0% |
| **Mín. varianza encogida (solo largo)** | **12.13%** | 15.51% | 0% |
| Pesos iguales | 16.95% | 22.50% | 0% |

- Prohibir el corto: **+1.34 pp** de mejora
- Mejorar el estimador dentro de solo-largo: **−0.02 pp**

La restricción pesó **67 veces más** que el estimador. Sin ella, la optimización explota los
errores de estimación: si por ruido cree que dos activos se cancelan, apalanca esa creencia sin
límite, y produce portafolios óptimos en el papel e imposibles de implementar.

Nótese además la columna *prometida*: los portafolios libres **entregan más riesgo del que
prometen** (+2 pp) a pesar de que el periodo de prueba fue más tranquilo que el de entrenamiento.
Se engañan a sí mismos.

### 5. La diversificación se rompe con la turbulencia, no con las caídas

Cohesión del mercado (R² promedio de las 1,211 acciones contra el factor), 138 meses:

| | |
|---|---|
| Correlación con la **volatilidad** del mes | **+0.755** |
| Correlación con el **rendimiento** del mes | −0.259 |

Los meses de mayor cohesión son todos eventos reconocibles: **abril 2025** (0.660), **marzo 2020**
(0.645), junio 2022, diciembre 2018, junio 2016 (Brexit), agosto 2019.

Pero abril de 2020 también aparece arriba **con rendimiento de +12.5%**: fue el rebote posterior al
desplome. **Pánico y euforia rompen la diversificación por igual.** Lo que la destruye no es que el
mercado baje, es que se agite.

### 6. El mismo portafolio, distinto régimen

| Régimen | Vol. portafolio | Ratio de diversificación |
|---|---|---|
| Todo el periodo | 15.65% | 1.52 |
| **Meses malos** | **20.22%** | **1.36** |
| Meses normales | 11.96% | 1.71 |
| Meses buenos | 13.11% | 1.74 |

Un motor calibrado con la historia completa reporta 15.65%, pero en los meses agitados el riesgo
real es 20.22%: **lo subestima en 29%**, justo en los momentos en que alguien consulta un motor de
riesgo.

Y el deterioro **no es parejo por tamaño**:

| Tamaño | Cuánto sube su cohesión de meses buenos a malos |
|---|---|
| Large | **1.33x** |
| Mid | 1.14x |
| Small | 1.08x |

Las **grandes** son las que más se sincronizan en las crisis. Eso completa el retrato del hallazgo
1: las small caps reaccionan más en magnitud, pero conservan más independencia cuando el mercado se
agita. Las grandes se vuelven casi intercambiables entre sí.

### 7. ¿Cuándo conviene Spark?

| | |
|---|---|
| *Lazy evaluation* | Transformation 0.06 ms vs action 583 ms — **9,239x** |
| Costo fijo de Spark | 1.14 s, contra 0.07 s de pandas |
| Con 350x más datos | pandas tardó **42x** más; Spark **1.4x** más |
| Punto de cruce | **entre 8 y 10 millones de filas** (2–3x el conjunto actual) |
| *Partition pruning* | Lee el 8.7% de los datos y tarda el 96% del tiempo |
| Techo de pandas | El *self-join* evitado pediría **127 GB** de RAM |

**pandas ganó en todos los tamaños medidos.** Decirlo es más honesto que disimularlo: 3.5 millones
de filas siguen siendo pocas para Spark.

La ventaja real de Spark aquí no es la velocidad, es **el techo**: la matriz de correlación
completa son 732,655 pares, y un `self-join` produciría 4,257 millones de filas. Spark puede
expresarlo y procesarlo por bloques; pandas moriría con `MemoryError`.

El resultado del *partition pruning* fue aparentemente malo y resultó revelador: el plan de
ejecución confirma que Spark descarta once de los doce años sin abrirlos, y aun así el tiempo apenas
baja. A esta escala el cuello de botella no es leer, es coordinarse.

**La optimización que más ahorró fue algorítmica, no de infraestructura**: cambiar el enfoque de
O(n²) a O(n) con el factor de mercado evitó por completo el cálculo de 732,655 pares.

---

## Problemas de calidad encontrados

### El criterio del Entregable 1 fallaba en small caps

El Entregable 1 marcaba como corrupto cualquier ticker con más de dos movimientos de ±50%. Sobre
el S&P 500 funcionaba; sobre el 1500 empezó a descartar **eventos reales**:

- **GME** con 9 movimientos — el *short squeeze* de enero de 2021
- **CLSK**, **MARA** — mineras de bitcoin
- **TGTX**, **SRPT**, **PTGX** — biotecnológicas que saltan con cada resultado de ensayo clínico

El sesgo no era menor: excluir a las small caps más volátiles contamina justo la pregunta de este
entregable.

**El criterio nuevo distingue por reversión, no por frecuencia.** Un split mal aplicado se deshace
—el precio regresa a menos del 15% de su nivel previo en tres días—; un desplome real se queda
abajo. Resultado: las exclusiones por datos corruptos pasaron de **16 a 2**.

### Precios estancados

Se detectaron con el patrón *gaps and islands* sobre una suma acumulada. El caso extremo es **IVT
(InvenTrust Properties) con 302 días hábiles sin que el precio cambie ni un centavo** — imposible
para una acción de índice. También AMCR (66 días), SW (34) y FERG (19).

---

## Limitaciones

| Limitación | Impacto |
|---|---|
| **Sesgo de supervivencia, agravado** | Al exigir ~2,900 días de historia nos quedamos con las small caps que *sobrevivieron* once años. Las chicas rotan mucho más que las grandes, así que el sesgo es más fuerte que en el Entregable 1 |
| **Cobertura** | De 1,502 tickers con datos quedan 1,211 (81%). Se perdieron 291 por historia insuficiente y 16 por calidad |
| **Beta medido contra un índice de igual ponderación** | 756 de las 1,211 empresas son mid y small, así que las grandes se miden contra un índice que no dominan. El beta < 1 de las large caps es en parte mecánico |
| **El factor de mercado no es PCA** | Es el promedio de igual ponderación. En la práctica es casi idéntico al primer componente principal, pero no se verificó en estos datos |
| **Punto de cruce inestable** | Dos corridas del benchmark dieron 7.8 y 10.4 millones. Solo el orden de magnitud es defendible |
| **Una sola matriz de covarianza** | El notebook 06 demuestra que el riesgo depende del régimen, pero el motor del 05 sigue usando una sola matriz. Resolverlo requiere escenarios de estrés |

---

## Lo que queda planteado para el Entregable 3

El notebook 06 deja un problema medido y sin resolver: **un motor de riesgo calibrado con la
historia completa subestima el riesgo en las crisis en 29%**.

Es exactamente la razón por la que las plataformas reales no usan una sola matriz, sino escenarios
de estrés con correlaciones de crisis. Construir eso —y la red de correlación que lo acompaña, con
la matriz de distancia que ya existe en `src/analysis.py`— es el siguiente capítulo natural.
