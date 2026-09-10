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
