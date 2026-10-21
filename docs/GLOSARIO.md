# Glosario

Explicación en español llano de los términos técnicos que aparecen en el README y en el notebook.
Pensado para alguien que abre el proyecto por primera vez y no tiene fondo en finanzas ni en datos.

## Correlación

Número entre -1 y 1 que mide qué tanto se mueven juntas dos cosas. 0 significa que no tienen
relación; 1 que se mueven idéntico; -1 que se mueven exactamente al revés. En este proyecto mide
qué tanto sube o baja una acción cuando otra sube o baja el mismo día.

## Rendimiento logarítmico

La forma de medir "cuánto cambió el precio de un día a otro" que usa este proyecto, en vez del
precio mismo. Se calcula como el logaritmo natural del precio de hoy dividido entre el precio de
ayer. Se usa en lugar del precio porque los precios crecen con el tiempo (no son "estacionarios") y
correlacionarlos directamente daría un número artificialmente alto entre cualquier par de empresas
que simplemente hayan subido en el periodo.

## Precio ajustado

El precio de cierre de una acción, corregido para que los splits (ver abajo) y el pago de
dividendos no se vean como caídas o subidas falsas. Sin este ajuste, un split 4:1 aparece como una
caída de -75% en un solo día, aunque el valor real de lo que tiene el inversionista no cambió.

## Split

Cuando una empresa divide cada acción existente en varias (por ejemplo, un split 2:1 convierte 1
acción de $100 en 2 acciones de $50). El valor total no cambia, solo el número de acciones y su
precio unitario. Si los datos no reflejan bien un split, el precio se ve con una caída o un salto
que nunca ocurrió realmente — eso fue justo lo que se encontró con el ticker `MNST`.

## Panel balanceado

Una tabla de datos rectangular, sin huecos: todas las columnas (aquí, todos los tickers) tienen
exactamente las mismas fechas. Construir un panel balanceado obliga a elegir cuántos días de
historia exigir, porque la empresa que salió a bolsa más recientemente limita cuánta historia
pueden tener todas las demás dentro del panel.

## Sector GICS

GICS (*Global Industry Classification Standard*) es el sistema estándar con el que Wall Street
agrupa empresas por industria (por ejemplo: Tecnología, Salud, Energía, Bienes Raíces). Este
proyecto usa esa clasificación para comparar si las acciones del mismo sector se mueven más
parecido entre sí que las de sectores distintos.

## Parquet

Un formato de archivo para guardar tablas de datos, pensado para ser rápido de leer y ocupar poco
espacio en disco (a diferencia de un CSV, que es texto plano). Este proyecto lo usa para guardar el
respaldo de precios, constituyentes y datos macro en `data/processed/`.

## Sesgo de supervivencia

Un error que se comete al analizar solo las empresas que existen *hoy*, ignorando las que
quebraron, se fusionaron o salieron del índice en el periodo estudiado. El resultado tiende a verse
mejor de lo que fue en realidad, porque las historias de fracaso quedan fuera de los datos.

## Entorno virtual

Una copia aislada de Python con sus propias librerías instaladas, separada del Python que ya tiene
el sistema operativo. Sirve para que las versiones exactas de las librerías que pide
`requirements.txt` no choquen con otros proyectos ni con lo que ya esté instalado en la
computadora. En este repositorio se crea con `python3 -m venv .venv`.

## API

*Application Programming Interface*: una forma en la que un programa le pide datos a otro programa
(normalmente a través de internet) usando un formato definido, en vez de que una persona entre a un
sitio web y copie la información a mano. Este proyecto usa la API de Yahoo Finance (vía la librería
`yfinance`) y la API de FRED para obtener datos automáticamente.

## Scraping

Extraer información de una página web diseñada para que la lea una persona (HTML), en vez de una
API pensada para programas. Este proyecto usa scraping para obtener de Wikipedia la lista de
empresas del S&P 500 y su sector GICS, ya que Wikipedia no ofrece una API oficial para esos datos.

## Ticker

El código corto con el que se identifica una acción en la bolsa (por ejemplo, `AAPL` para Apple o
`GOOGL` para Alphabet/Google). Es la llave que se usa en todo el proyecto para conectar precios,
sector y nombre de cada empresa.

---

# Términos del Entregable 2

Los siguientes términos aparecen en `docs/ENTREGABLE_2.md`, en los notebooks de `databricks/` y en
`src/risk.py`.

## Data lake

Un almacén central donde se guardan los datos de un proyecto, en su forma original o casi, para que
cualquiera del equipo los consulte desde el mismo lugar. A diferencia de una carpeta con archivos
sueltos en la laptop de alguien, un data lake vive en la nube y lo leen varias personas a la vez.
En este proyecto el data lake está en Databricks, y por eso los datos del Entregable 2 ya no se
guardan en el repositorio de GitHub: se reconstruyen corriendo los notebooks.

## Tabla Delta

El formato con el que Databricks guarda las tablas del data lake. Por dentro son archivos Parquet,
pero con un registro de cambios encima que permite deshacer, auditar y actualizar la tabla sin
corromperla si dos procesos la tocan a la vez. Cuando el README dice "tablas Delta", piensa en
"tablas de una base de datos que viven en Databricks".

## Arquitectura medallion (bronze / silver / gold)

Una forma de ordenar un data lake en tres capas, de lo más crudo a lo más pulido:

- **Bronze:** el dato tal como llegó de la fuente, con todo y errores. No se limpia nada.
- **Silver:** el dato ya limpio (sin tickers corruptos, con rendimientos calculados).
- **Gold:** los resultados finales del análisis (betas, riesgo de portafolio).

La ventaja es que, si descubres que una regla de limpieza estaba mal, recalculas Silver desde Bronze
sin volver a descargar nada.

## PySpark

La forma de usar Spark escribiendo Python. Spark es un motor que reparte el trabajo sobre una tabla
grande entre varios procesadores (o varias computadoras) en vez de hacerlo en uno solo, como hace
pandas. En el proyecto sirve para que el análisis de 4.1 millones de filas siga funcionando aunque
el universo crezca mucho más.

## Window function (función de ventana)

Un cálculo que, para cada fila, mira otras filas "cercanas" dentro de un grupo ordenado. El ejemplo
del proyecto: el rendimiento de hoy necesita el precio de *ayer*, y "ayer" solo tiene sentido dentro
de la misma empresa y ordenado por fecha. Eso se expresa como una ventana particionada por ticker y
ordenada por fecha, en vez de tener que cruzar la tabla consigo misma.

## Lazy evaluation (evaluación perezosa)

Spark no ejecuta una instrucción en cuanto la escribes: primero las anota (*transformations*) y
solo trabaja de verdad cuando pides un resultado (*action*, como contar filas o mostrar la tabla).
Así puede revisar todo el plan y ejecutarlo de la forma más barata. El benchmark del proyecto lo
mide: anotar una transformación tarda 0.06 ms; pedir el resultado, 583 ms.

## Partición

Dividir una tabla grande en pedazos según el valor de una columna (aquí, por año) y guardar cada
pedazo por separado. Si una consulta solo necesita 2024, Spark puede saltarse los demás años sin
abrirlos. A ese salto se le llama *partition pruning*.

## Beta

Cuánto amplifica una acción el movimiento del mercado. Beta 1 significa que se mueve igual que el
mercado; beta 1.5, que cuando el mercado sube 1% ella sube 1.5% (y baja 1.5% cuando él baja 1%);
beta menor a 1, que se mueve menos. En el proyecto, las empresas chicas tienen beta de 1.10 y las
grandes de 0.86.

## R² (R cuadrada)

Qué fracción del movimiento de una acción se explica por el mercado, de 0 a 1. R² de 0.32 significa
que el 32% de lo que hace la acción es "el mercado" y el 68% restante son razones propias de la
empresa. No es lo mismo que el beta: una acción puede amplificar mucho al mercado los días que lo
sigue (beta alto) y aun así moverse casi siempre por su cuenta (R² bajo).

## Covarianza

Parecida a la correlación, pero sin normalizar: mide cómo varían juntas dos cosas, y su tamaño
depende de qué tan volátiles son. Una matriz de covarianza junta todos los pares de acciones en una
tabla, y es la pieza que necesita el motor de riesgo para calcular la volatilidad de un portafolio.
Con 1,211 acciones son más de 700,000 números que estimar a partir de unos 2,900 días de datos.

## Shrinkage (encogimiento)

Una técnica para que la matriz de covarianza sea más confiable cuando hay muchas acciones y pocos
días de datos. Consiste en mezclar la matriz estimada con una versión muy simple y estable, para
"encoger" los valores extremos que probablemente son ruido. El proyecto usa el método de Ledoit y
Wolf, que bajó el número de condición de la matriz de 232,640 a 14,910.

## Ratio de diversificación

Cuántas posiciones *verdaderamente independientes* tiene un portafolio. Se calcula dividiendo el
promedio de la volatilidad de cada posición entre la volatilidad real del portafolio completo. Un
ratio de 1 significa que todas las posiciones se comportan como una sola; el máximo posible con 10
posiciones es √10 ≈ 3.16, y solo si no tuvieran correlación alguna. Diez bancos dan 1.12; diez
sectores distintos, 1.53.

---

## Términos de Databricks

Aparecen en las instrucciones de "Entregable 2: el data lake en Databricks" del README.

## Unity Catalog

El sistema de Databricks que organiza las tablas en niveles: catálogo → esquema → tabla. En este
proyecto las tablas se llaman, por ejemplo, `workspace.bronze.prices` (catálogo `workspace`,
esquema `bronze`, tabla `prices`).

## Serverless (compute)

El tipo de cómputo de Databricks donde no administras ningún servidor: eliges "Serverless" y
Databricks te asigna los recursos al ejecutar. Es el que hay que usar para correr los notebooks de
Python; un *SQL warehouse* solo ejecuta celdas SQL y las celdas de Python fallan.

## Git folder

Una carpeta dentro de Databricks que está conectada a un repositorio de GitHub. Permite traer el
código del repositorio al workspace de Databricks para ejecutarlo ahí.
