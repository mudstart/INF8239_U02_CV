# Model Card · Fashion-MNIST CNN

> **Resumen.** CNN pequeña entrenada 8 épocas sobre Fashion-MNIST: F1 macro 0,789 en prueba.
> Con esta configuración **rinde peor que el baseline denso** (F1 0,865), es más lenta y
> está subentrenada. **No se recomienda usarla** tal como está; con el mismo presupuesto de
> épocas conviene el baseline denso. Es un modelo didáctico, sin validez fuera de
> Fashion-MNIST.

## Modelo y versión

| Campo | Valor |
|---|---|
| Nombre | `small_cnn` (`src/inf8239_u02_cv/models.py`, función `build_cnn`) |
| Versión | 0.1.0 (paquete `inf8239_u02_cv`) |
| Arquitectura | Entrada 28×28×1 → Conv2D 32 filtros 3×3 (ReLU) → MaxPooling → Conv2D 64 filtros 3×3 (ReLU) → MaxPooling → GlobalAveragePooling → Dropout 0,25 → Dense 10 (softmax) |
| Parámetros | 19.466 |
| Entrenamiento | Adam, pérdida `sparse_categorical_crossentropy`, lotes de 128, máximo 8 épocas, `EarlyStopping` (paciencia 2, restaura los mejores pesos según la pérdida de validación) |
| Semilla | 42 (Python, NumPy y TensorFlow); las métricas se reprodujeron idénticas en ejecuciones sucesivas |
| Archivos | `models/cnn.keras` (102,9 KB, solo arquitectura y pesos, para inferencia) y `models/best_cnn.keras` (262 KB, punto de control con el estado del optimizador) |
| Modelo de referencia | `dense_baseline` (`build_dense`): Flatten → Dense 64 (ReLU) → Dense 10; `models/dense.keras` |
| Reproducción | `uv sync --extra cpu` y `uv run python scripts/train_cv.py --epochs 8` |

## Uso previsto

- **Didáctico:** comparar una red densa y una CNN con la misma partición, analizar curvas de
  aprendizaje, errores y costo computacional (Green AI) en el curso INF-8239.
- Clasificar imágenes **de Fashion-MNIST** o con su mismo formato exacto: una sola prenda
  centrada, 28×28 píxeles en escala de grises, fondo negro, en una de las 10 clases del
  dataset.
- La salida es una distribución de probabilidad sobre las 10 clases; se interpreta como una
  sugerencia, no como una decisión.

## Usos fuera de alcance

- **Fotografías reales de ropa** (comercio electrónico, inventarios, recomendación): tienen
  color, fondos, iluminación variable, personas o varias prendas por imagen.
- **Cualquier otro dominio:** imágenes médicas, OCR, vigilancia, biometría o identificación
  de personas. Un buen resultado en Fashion-MNIST no demuestra preparación para ninguno de
  ellos.
- **Decisiones automáticas** con consecuencias para personas (precios, devoluciones,
  control de calidad) sin revisión humana.
- Clases distintas de las 10 del dataset: el modelo siempre responde con una de ellas,
  aunque la imagen no sea una prenda.

## Dataset y particiones

**Fashion-MNIST** (Zalando Research; Xiao, Rasul y Vollgraf, 2017), descargado con
`tf.keras.datasets.fashion_mnist` a la caché de Keras (`~/.keras/datasets/fashion-mnist/`).
70.000 imágenes de 28×28 en escala de grises, 10 clases balanceadas: Camiseta/top,
Pantalón, Jersey, Vestido, Abrigo, Sandalia, Camisa, Zapatilla, Bolso y Botín.

| Partición | Origen | Imágenes | Por clase |
|---|---|---|---|
| Entrenamiento | Primeras 54.000 del conjunto oficial de entrenamiento | 54.000 | 5.367–5.445 |
| Validación | Últimas 6.000 del conjunto oficial de entrenamiento | 6.000 | 555–633 |
| Prueba | Conjunto oficial de prueba | 10.000 | 1.000 |

La validación se separa por posición, sin estratificar; queda ligeramente desbalanceada
(555 a 633 por clase). La CNN y el baseline usan exactamente la misma partición. El
detalle por clase está en `reports/data_summary.json`.

## Preprocesamiento

1. Conversión a `float32` y división por 255: valores en [0, 1].
2. Se añade el canal: forma `(n, 28, 28, 1)`.
3. Validación (`validate_images`) de las tres particiones antes de entrenar: forma 28×28×1,
   valores finitos en [0, 1], etiquetas enteras entre 0 y 9, mismo número de imágenes y
   etiquetas.

No se usa aumento de datos ni ninguna otra transformación.

## Métricas globales y por clase

Partición de prueba (10.000 imágenes). Fuente: `reports/cv_metrics.json` y
`reports/per_class_metrics.csv`.

| Métrica global | CNN | Baseline denso |
|---|---|---|
| F1 macro | 0,789 | **0,865** |
| Accuracy | 0,793 | **0,866** |
| Precisión macro | 0,791 | **0,868** |
| Recall macro | 0,793 | **0,866** |

| Clase | Precisión CNN | Recall CNN | F1 CNN | F1 baseline |
|---|---|---|---|---|
| Camiseta/top | 0,645 | 0,807 | 0,717 | 0,819 |
| Pantalón | 0,988 | 0,932 | 0,959 | 0,975 |
| Jersey | 0,706 | 0,642 | 0,672 | 0,753 |
| Vestido | 0,753 | 0,843 | 0,796 | 0,869 |
| Abrigo | 0,661 | 0,663 | 0,662 | 0,782 |
| Sandalia | 0,950 | 0,885 | 0,916 | 0,948 |
| **Camisa** | **0,500** | **0,374** | **0,428** | 0,653 |
| Zapatilla | 0,855 | 0,952 | 0,901 | 0,934 |
| Bolso | 0,917 | 0,937 | 0,927 | 0,963 |
| Botín | 0,935 | 0,896 | 0,915 | 0,952 |

Camisa es la clase más difícil para los dos modelos. Las principales confusiones de la CNN
son Camisa → Camiseta/top (274 casos), Jersey → Abrigo (155), Jersey → Camisa (143) y
Camisa → Abrigo (130): siempre entre prendas superiores. Matrices completas en
`reports/confusion_cnn.png` y `reports/confusion_dense.png`; ejemplos en
`reports/cnn_errors.png`.

## Comparación de costo

- **Hardware:** CPU AMD Ryzen 5 5600X (6 núcleos, 12 hilos), 32 GB de RAM, sin aceleración
  por GPU. El equipo tiene una AMD Radeon RX 6800 XT, que no se usó: TensorFlow no admite
  GPU en Windows nativo y la ruta GPU del proyecto requiere NVIDIA.
- **Entorno:** Windows 11 (10.0.26300), Python 3.12.14, TensorFlow 2.21.0, Keras 3.15.1,
  NumPy 2.5.3, scikit-learn 1.9.1. Versiones fijadas en `uv.lock`; detalle de la ejecución
  en `reports/runtime.json`.
- **Parámetros:** CNN 19.466; baseline 50.890 (la CNN tiene un 62 % menos).
- **Tiempo de entrenamiento (8 épocas):** CNN unos 50 s; baseline unos 6 s. En varias
  ejecuciones osciló entre 48 y 54 s (CNN) y entre 5 y 6 s (baseline): la CNN tarda
  entre 8 y 10 veces más (54,1 s frente a 5,8 s en la ejecución registrada).
- **Tiempo de inferencia:** CNN 0,072 ms por imagen; baseline 0,031 ms por imagen en la
  ejecución registrada (mediana de 5 repeticiones sobre las 10.000 imágenes de prueba, tras
  una predicción de calentamiento). Entre ejecuciones, la CNN osciló entre 0,054 y
  0,072 ms según la carga del equipo: es entre 1,8 y 2,3 veces más lenta.
- **Tamaño del modelo para inferencia:** CNN 102,9 KB; baseline 215,5 KB.
- **Cómputo:** la CNN realiza unos 3,84 millones de multiplicaciones-acumulaciones por
  imagen frente a unas 51.000 del baseline. Tiene menos parámetros, pero cada filtro se
  aplica en todas las posiciones de la imagen.

**Lectura Green AI:** con 8 épocas, la CNN cuesta más en entrenamiento e inferencia y no
aporta mejora predictiva (−0,075 de F1 macro frente al baseline). Su único ahorro es el
tamaño en disco.

## Limitaciones y riesgos

- **Subentrenamiento:** la pérdida de validación bajó en todas las épocas y la mejor época
  fue la última (8 de 8): el límite de épocas cortó el aprendizaje. Las métricas reflejan
  este presupuesto, no la capacidad de la arquitectura (`reports/learning_curves.png`).
- **Confusión entre prendas superiores:** Camisa, Jersey, Abrigo y Camiseta/top comparten
  silueta; lo que las distingue (tejido, botones, cierres) ocupa uno o dos píxeles a esta
  resolución. El recall de Camisa es 0,374: se pierden casi dos de cada tres camisas.
- **Errores con alta confianza:** hay errores con 70–82 % de confianza en la clase
  equivocada, por lo que la confianza no basta como señal de alerta.
- **Etiquetas discutibles:** algunas prendas sin mangas o de tirantes están etiquetadas
  como Camisa; parte del error medido puede venir del dataset.
- **Dominio cerrado:** el modelo siempre elige una de las 10 clases; ante una imagen que no
  es una prenda de Fashion-MNIST responde igual, sin advertir que está fuera de dominio.
- **Validación no estratificada** y una sola partición y semilla: no se estimó la
  variabilidad de las métricas entre particiones.
- **Integridad del dataset:** la descarga en caché no se verifica con un hash.

## Supervisión y monitoreo

- **No desplegar** esta versión: antes, reentrenar hasta que el `EarlyStopping` se active y
  repetir la comparación de desempeño y costo con el baseline.
- **Revisión humana** de las predicciones de prendas superiores, en particular cuando el
  modelo predice Camiseta/top, Abrigo o Camisa.
- Si se usara con imágenes nuevas, **controlar que tengan el formato de Fashion-MNIST**
  (28×28, escala de grises, prenda centrada, fondo oscuro) y rechazar las demás.
- **Monitorear** la distribución de clases predichas y la confianza media; un cambio
  brusco indica datos distintos a los de entrenamiento.
- **Reevaluar** el recall por clase (en especial Camisa) con cada reentrenamiento, junto con
  el tiempo de entrenamiento e inferencia en el hardware real de uso.
- Ejecutar `uv run pytest -q` antes de cada entrenamiento: valida forma, rango, etiquetas y
  el contrato de salida de ambos modelos.
