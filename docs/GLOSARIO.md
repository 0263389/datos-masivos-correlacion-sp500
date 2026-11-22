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

## Contribución al riesgo

Qué parte del riesgo total del portafolio aporta cada posición. **No es proporcional al dinero
invertido**: una posición del 5% puede aportar el 30% del riesgo si es volátil y se mueve junto con
el resto. Las contribuciones suman exactamente el 100%, y no por aproximación sino por el teorema
de Euler, así que el reparto es exacto.

## Mínima varianza

La combinación de pesos que produce el portafolio menos volátil posible con esos activos. En este
proyecto se calcula **sin permitir ventas en corto**: la versión libre llegó a posiciones de -155%
del capital, porque si por ruido cree que dos activos se cancelan, apalanca esa creencia sin
límite. Exigir pesos positivos actúa como regularización y el portafolio sobrevive mucho mejor
fuera de muestra.

## Error estándar

El margen de error de una estimación. Mientras más chico, más confiable es el número. Un
coeficiente se considera significativo cuando es grande comparado con su error estándar —
concretamente, más de 1.96 veces, que es el corte al 5%.

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

---

# Términos de los Entregables 3, 4 y 5

Aparecen en el terminal, en los módulos de predicción y riesgo, y en la guía de estudio.

## Volatilidad

Qué tanto se mueve un precio, hacia arriba o hacia abajo. **No mide si gana o pierde**: mide qué
tan brusco es. Es la desviación estándar de los rendimientos. Se reporta anualizada, multiplicando
la diaria por la raíz de 252 (los días que abre la bolsa en un año). Va la raíz y no el número
completo porque lo que crece proporcional al tiempo es la varianza, que es la volatilidad al
cuadrado. Una acción que se mueve 1% al día tiene 15.87% de volatilidad anual.

## Volatilidad condicional

La volatilidad que el modelo estima para **un día concreto**, usando solo lo que ocurrió antes de
ese día. Se distingue de la volatilidad "a secas", que es un promedio de todo el periodo y por eso
no cambia aunque el mercado esté en pánico.

## GARCH

Modelo que predice la volatilidad aprovechando que **viene en rachas**: después de un día movido es
más probable que venga otro, igual que el clima. La varianza de mañana se arma con tres piezas:
`ω + α × (golpe de hoy)² + β × (varianza de hoy)`. El golpe va al cuadrado porque importa su tamaño
y no su dirección. Está implementado a mano en `src/volatility.py`.

## Persistencia

La suma `α + β` del modelo GARCH. Mide **cuánto duran las rachas** de volatilidad. Cerca de 0 no hay
rachas y el modelo no sirve para predecir; entre 0.95 y 0.99 —lo normal en mercados reales— después
de un susto el mercado tarda semanas en calmarse; en 1 o más el modelo explota y predice volatilidad
infinita.

## Variance targeting

Truco de ajuste del GARCH: en lugar de estimar tres parámetros, `ω` se despeja para que la varianza
de largo plazo del modelo coincida con la observada en los datos. Deja solo dos parámetros que
buscar, y garantiza que el modelo nunca reporte un nivel de volatilidad que los datos no respaldan.

## Validación hacia adelante (walk-forward)

Método de prueba que simula **ser un inversionista del pasado**: con solo la información disponible
hasta cierto día arma la cartera, la mantiene un mes, mide lo que realmente pasó, y avanza. Se repite
con distintas cantidades de historia para encontrar cuál funciona mejor. Lo crucial es que el modelo
nunca ve el futuro; si lo viera, los resultados saldrían preciosos y serían falsos.

## Look-ahead bias

El error de usar, para predecir una fecha, información que en esa fecha todavía no existía. Produce
resultados excelentes y completamente falsos. Todo el diseño de `src/validation.py` existe para
evitarlo.

## Rotación

Qué porcentaje de la cartera hay que mover en cada rebalanceo. Más rotación significa más costos de
operación, así que una ventana que da poca volatilidad pero exige rotar el 25% cada mes puede no
convenir en la práctica.

## Información mutua

Medida de dependencia que detecta **cualquier** relación entre dos series, no solo las lineales. Vale
cero si y solo si son independientes. Existe porque la correlación es, por definición, la pendiente
de la mejor recta: si la relación tiene forma de U, la correlación da casi cero aunque la relación
sea perfecta. Se estima discretizando las dos series en casillas por cuantiles.

## Transformación de Fisher

`z = arctanh(r)`. Convierte una correlación, que está acotada entre -1 y 1, en un número sin techo.
Hace falta al medir **dispersión** de correlaciones: cuando la media sube hacia 1, la dispersión
tiene que encogerse por geometría, sin que el mercado haya hecho nada. Sobre `z` ese artefacto
desaparece. En este proyecto el signo del hallazgo se invirtió al aplicarla.

## Dependencia de cola

La probabilidad de que dos activos estén simultáneamente en sus peores días. Si fueran
independientes sería igual al cuantil elegido (10% con el decil). Mide si **caen juntos**, que es
justo lo que la correlación promedio no puede distinguir: la correlación es un solo número para toda
la nube de puntos y es simétrica, así que no puede expresar que las caídas van más sincronizadas que
las subidas.

## Cofactor

Variable que **modifica el efecto de otra** en lugar de sumar el suyo propio. Entra en el modelo como
producto: `γ × (r_mercado × tensión)`. El coeficiente γ es el escalón — cuánto cambia la sensibilidad
al mercado al pasar de calma a crisis.

## Régimen

El estado del mercado. En este proyecto son dos, calma y tensión, y no se eligen a mano: son tensos
los días en que la volatilidad condicional que estimó el GARCH está en su quintil más alto. Definirlo
con el modelo y no a criterio evita el grado de libertad de escoger el corte que da el resultado
conveniente.

## Beta condicional

La beta medida por separado dentro de cada régimen, en vez de un solo número para toda la historia.
El hallazgo del proyecto es que **converge hacia 1 en las crisis**: las acciones defensivas suben su
beta y las agresivas la bajan, así que las defensivas dejan de proteger exactamente cuando se
necesita.

## Heterocedasticidad

Que la varianza del error no sea constante a lo largo del tiempo. Es exactamente lo que el GARCH de
este proyecto demostró que ocurre, y por eso los errores estándar de mínimos cuadrados de libro —que
suponen lo contrario— no se pueden usar aquí.

## Errores estándar robustos (White, HC1)

Margen de error calculado **sin** suponer varianza constante: en lugar de una varianza única para
todos los días, cada día aporta su propio residuo. Hacen falta porque los clásicos subestiman el
margen —medido en este proyecto, 1.54 veces— lo que infla los estadísticos t y hace parecer
significativo lo que no lo es.

## Significancia estadística

Que un resultado sea lo bastante grande frente a su margen de error como para no atribuirlo al azar.
El corte usado en todo el proyecto es el 5%.

## VaR (Valor en Riesgo)

Cuánto puedes perder en un mal día. Más preciso: en el peor 5% de los días, pierdes al menos esa
cantidad. En este proyecto se calcula **histórico**, leyendo la distribución observada, y no
suponiendo una campana normal: las colas reales son más gordas que la normal, y un VaR gaussiano
heredaría ese error justo en la cola que pretende medir.

## Expected Shortfall

El promedio de pérdida de **toda** la cola, no solo de su borde. Acompaña al VaR porque el VaR dice
cuánto pierdes en el peor 5% de los días *en el mejor de esos casos*, y no dice nada sobre qué tan
grave puede ser el resto. Por eso el marco regulatorio de Basilea lo adoptó en sustitución del VaR.

## Máxima caída (drawdown)

La pérdida acumulada desde el punto más alto hasta el más bajo posterior. Es la medida que un
inversionista reconoce de inmediato, porque es la que habría visto en su estado de cuenta: no la de
un día suelto, sino la del recorrido completo.

## Árbol de expansión mínima

El conjunto mínimo de conexiones que mantiene unido a todo un grafo: exactamente `n − 1` aristas, las
más cortas posibles, sin ciclos. Aplicado al mercado, reduce 104,653 correlaciones a 457 conexiones,
y las que sobreviven son las que de verdad sostienen la estructura. Se calcula con el algoritmo de
Prim, implementado a mano en `src/network.py`.

## Distancia de Mantegna

`d = √(2 × (1 − ρ))`. Convierte una correlación en una distancia: ρ=1 da distancia 0, ρ=0 da 1.414 y
ρ=−1 da 2. Hace falta porque la correlación no sirve como distancia tal cual, y además cumple la
desigualdad del triángulo, que es la propiedad que permite meterla a un árbol. Viene de Mantegna
(1999), *Hierarchical structure in financial markets*.

## Concentrador (hub)

En la red del mercado, un activo que toca muchas más conexiones que el promedio. En un árbol el grado
promedio es 2 y casi todo son hojas, así que los pocos que se salen son puntos de paso. Un
concentrador no es el activo más volátil ni el más correlacionado con el índice: es el que, si se
mueve, arrastra a más vecinos directos.
