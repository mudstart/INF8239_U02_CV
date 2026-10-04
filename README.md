# INF-8239 · Unidad 02 · Proyecto de visión (LAB07)

Autor académico: Edwin Ramón José Nolasco
Estudiante Jose Miguel Maduro Valenzuela

CNN reproducible con Fashion-MNIST: un baseline denso y una CNN comparados con la misma
partición, con métricas por clase, matrices de confusión, errores visuales, costo
computacional (Green AI) y Model Card. Proyecto separado del de PLN para no mezclar
TensorFlow con ese entorno.

## Reproducción

### CPU (Windows, Linux o macOS)

```bash
uv python install 3.12
uv sync --extra cpu
uv run python scripts/check_runtime.py
uv run pytest -q
uv run python scripts/train_cv.py --epochs 8
```

`check_runtime.py` informa Python, sistema operativo, versión de TensorFlow y GPU detectada.
Que no detecte GPU no impide el laboratorio: la ejecución registrada se hizo en CPU, en
alrededor de un minuto. El dataset (unos 30 MB) se descarga en la primera ejecución a la caché de
Keras (`~/.keras/datasets/`) y no se versiona.

### GPU

Solo en WSL2/Linux con una GPU NVIDIA correctamente configurada. TensorFlow no admite GPU en
Windows nativo, y la ruta GPU no sirve para tarjetas AMD.

```bash
uv sync --extra gpu
```

### Google Colab (ruta alternativa)

Colab no usa `uv`: se instala con `pip` desde `requirements-colab.txt`, y además hay que
clonar el repositorio y registrar el paquete del proyecto.

```python
!git clone https://github.com/mudstart/INF8239_U02_CV.git
%cd INF8239_U02_CV
%pip install -r requirements-colab.txt
%pip install --no-deps -e .
!python scripts/train_cv.py --epochs 8
```

Si Colab reinstala TensorFlow, puede pedir reiniciar el entorno antes de entrenar. Esta
ruta no se ejecutó en Colab: las versiones de Colab no están fijadas como en `uv.lock`.
No suba nunca `.venv`.

## Archivos que genera `train_cv.py`

| Archivo | Contenido |
|---|---|
| `reports/data_summary.json` | Forma y conteo por clase de entrenamiento, validación y prueba (se imprime antes de entrenar) |
| `reports/training_history.json` | Pérdida y accuracy por época de cada modelo |
| `reports/learning_curves.png` | Curvas de pérdida de entrenamiento y validación |
| `reports/cv_metrics.json` | F1 macro, accuracy, parámetros, tamaño del modelo, tiempos de entrenamiento e inferencia, épocas |
| `reports/per_class_metrics.csv` | Precisión, recall y F1 por clase de los dos modelos |
| `reports/confusion_dense.png`, `reports/confusion_cnn.png` | Matrices de confusión con los nombres de las clases |
| `reports/cnn_errors.png`, `reports/cnn_error_examples.csv` | Errores de la CNN en los 4 pares más confundidos, con su confianza |
| `reports/runtime.json` | CPU, sistema operativo, Python, TensorFlow y dispositivo |
| `models/dense.keras`, `models/cnn.keras` | Modelos para inferencia (arquitectura y pesos, sin optimizador) |
| `models/best_cnn.keras` | Punto de control de la CNN con el estado del optimizador |

Las métricas de desempeño son idénticas en cada ejecución (semilla 42). Los tiempos varían
ligeramente según la carga del equipo.

## Notebook ejecutado

`notebooks/ejercicio04.ipynb` reúne la evidencia con salidas y figuras: entorno, auditoría y
partición, curvas, métricas globales y por clase, matrices de confusión, errores visuales,
costo y pruebas. **No reentrena:** carga `models/dense.keras` y `models/cnn.keras`, recalcula
las métricas y comprueba que coinciden con `reports/`. Para reejecutarlo:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/ejercicio04.ipynb
```

## Dependencias

`uv.lock` es la fuente de verdad. `requirements.txt` es su exportación con TensorFlow para
CPU, para entornos sin uv (`pip install -r requirements.txt`); se regenera con
`uv export --format requirements.txt --no-hashes --extra cpu --output-file requirements.txt`.
`requirements-colab.txt` es la lista mínima para Google Colab.

## Pruebas

`uv run pytest -q` ejecuta 15 pruebas rápidas (no descargan datos ni entrenan) sobre los
contratos del proyecto: forma y rango de las imágenes, número de clases y etiquetas válidas,
dimensión de la salida de ambos modelos, y probabilidades finitas entre 0 y 1 que suman 1.

## Resultados

Partición de prueba de Fashion-MNIST (10.000 imágenes), 8 épocas, CPU AMD Ryzen 5 5600X.

| | Baseline denso | CNN |
|---|---|---|
| F1 macro | **0,865** | 0,789 |
| Accuracy | **0,866** | 0,793 |
| Recall de Camisa (clase más difícil) | **0,597** | 0,374 |
| Parámetros | 50.890 | **19.466** |
| Archivo del modelo | 215,5 KB | **102,9 KB** |
| Tiempo de entrenamiento | **unos 6 s** | unos 50 s |
| Inferencia por imagen | **0,027–0,034 ms** | 0,054–0,072 ms |

Análisis completo por paso (curvas, comparación, errores): `reports/lab07_analisis.md`.
Ficha del modelo: `MODEL_CARD.md`.

## Cierre interpretativo

**Resultado principal:** con un presupuesto de 8 épocas, la CNN (F1 macro 0,789) rinde
peor que el baseline denso (0,865). Las curvas de aprendizaje muestran que la CNN está
subentrenada: su pérdida de validación bajó en todas las épocas, la mejor época fue la
última y todavía mejoraba 0,024 al final, mientras que la red densa estaba cerca de su
techo (0,006).

**Evidencia predictiva:** misma partición (54.000 / 6.000 / 10.000) para ambos modelos. La
red densa supera a la CNN en F1 macro (+0,075), en accuracy (86,6 % frente a 79,3 %) y en
recall en 9 de las 10 clases; la única excepción es Zapatilla. Los resultados se
reprodujeron idénticos en ejecuciones sucesivas.

**Clase más difícil:** Camisa, en los dos modelos: recall 0,374 en la CNN y 0,597 en la
red densa. Se confunde sobre todo con Camiseta/top (274 casos en la CNN), Abrigo y Jersey.
Las prendas superiores comparten silueta, y lo que las distingue (tejido, botones, cierres)
ocupa uno o dos píxeles en 28×28. Algunas prendas sin mangas etiquetadas como Camisa
sugieren además etiquetas discutibles.

**Costo comparado:** la CNN tiene un 62 % menos de parámetros y un archivo un 52 % más
pequeño, pero tarda unas 9 veces más en entrenar (unos 50 s frente a 6 s) y entre 1,8 y
2,3 veces más por predicción. Realiza unos 3,84 millones de multiplicaciones-acumulaciones
por imagen frente a unas 51.000 de la red densa: menos parámetros no significa menos
cómputo, porque cada filtro se aplica en todas las posiciones de la imagen.

**¿La mejora justifica el costo?:** no, porque no hay mejora: con 8 épocas la CNN es más
cara y además peor. Con esta evidencia se mantiene el baseline denso. Entrenar la CNN hasta
que converja podría invertir el resultado, pero aumentaría su costo; habría que repetir la
comparación antes de decidir.

**Limitación del benchmark:** Fashion-MNIST es didáctico: 28×28 píxeles en escala de
grises, una sola prenda centrada sobre fondo negro, sin oclusiones ni variaciones de
iluminación, y 10 clases fijas. Además se usó una sola partición y una sola semilla, con
validación no estratificada. Las métricas no dicen nada sobre fotografías reales de ropa
ni sobre otros dominios.

**Decisión antes de usar otro dominio:** no usar este modelo fuera de Fashion-MNIST. Para
otro dominio (fotos reales de ropa, imágenes médicas, OCR, vigilancia) haría falta un
dataset propio del dominio con licencia y etiquetado documentados, reentrenar y evaluar
con métricas por clase en ese contexto, entrenar hasta la convergencia, repetir la
comparación de costo en el hardware real de uso y mantener revisión humana de las
predicciones, sobre todo en las clases que se confunden.

## Uso de herramientas de IA

**Herramienta:** Claude Code (Anthropic), modelo Claude Opus 5.5, en la aplicación de
escritorio de Claude, con acceso al repositorio y a la terminal.

**Uso:** apoyo para contrastar el proyecto base con el manual del LAB07, proponer y escribir
código, ejecutar comandos, redactar borradores de documentación y detectar errores. El
estudiante decidió el alcance de cada cambio, ejecutó y comprobó los resultados, y es
responsable de los datos, el código, las referencias y las conclusiones.

**Prompts relevantes** (parafraseados y agrupados por propósito):

| Propósito | Solicitud |
|---|---|
| Diagnóstico inicial | Analizar el proyecto base de visión y resumir su estado. |
| Organización | Evaluar si convenía trabajar el LAB07 dentro del repositorio de PLN o en uno propio. |
| Verificación de requisitos | Validar, punto por punto, si el proyecto cumplía cada paso del manual. |
| Corrección | Aplicar las soluciones propuestas para los pasos que no se cumplían. |
| Control de calidad | Volver a evaluar cada paso después de los cambios. |

**Componentes desarrollados con apoyo de IA:** las ampliaciones de `scripts/train_cv.py`
(resumen de particiones, historial y curvas, métricas por clase, medición de inferencia con
calentamiento y repeticiones, guardado de modelos sin optimizador, información del entorno,
selección de errores y matrices con nombres), la validación de etiquetas en
`src/inf8239_u02_cv/data.py`, las 12 pruebas nuevas, `MODEL_CARD.md`,
`reports/lab07_analisis.md`, `notebooks/ejercicio04.ipynb` y los borradores del cierre
interpretativo.

**Verificaciones realizadas:**

- Las métricas de desempeño se reprodujeron idénticas en todas las ejecuciones, incluida
  una copia limpia del repositorio clonada desde GitHub siguiendo solo este README.
- El notebook recalcula las métricas desde los modelos guardados y comprueba que coinciden
  con `reports/cv_metrics.json`, y que la partición coincide con `data_summary.json`.
- Cada cifra del reporte, la Model Card y este README se comprobó contra los archivos de
  `reports/`.

**Correcciones realizadas durante el trabajo:**

- El `.gitignore` del proyecto base excluía `reports/` y `models/`, por lo que la evidencia
  no llegaba a GitHub; se corrigió.
- `include_optimizer=False` no tiene efecto en Keras 3: los archivos `.keras` seguían
  incluyendo el estado del optimizador, que triplicaba su tamaño. Se reconstruye el modelo
  solo con arquitectura y pesos.
- Esa reconstrucción consumía el generador aleatorio y cambiaba la inicialización de la CNN
  (F1 0,7893 → 0,7865); se movió al final del script y el resultado volvió a ser idéntico.
- La primera medición de inferencia incluía la llamada inicial de TensorFlow; se añadió una
  predicción de calentamiento y 5 repeticiones.
- Varias cifras de los borradores no coincidían con los datos (variación de la inferencia,
  rangos de confianza, tiempos de una ejecución anterior) y se corrigieron.
- La hipótesis inicial de que la CNN rendía peor por su arquitectura se reemplazó, con las
  curvas de aprendizaje como evidencia, por la de subentrenamiento.
