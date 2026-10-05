# SmartEye — Estructura del proyecto

## Concepto actual del proyecto

Clasificación de Imágenes Médicas para Detección de Cancer de Mama mediante Aprendizaje Profundo con ResNet50, Mejora de Contraste CLAHE e Integracion Multimodal de Características Auxiliares *

## Estructura
Contiene los siguientes componentes y experimentos:

- Clasificación de imágenes en 3 clases: `benign`, `malignant` y `normal`.
- Imágenes en escala de grises redimensionadas a `224x224` píxeles.
- Uso de **ResNet50** preentrenada con **ImageNet**.
- Uso de **ViT** preentrenada como segundo modelo de comparación.
- Conversión de imágenes de escala de grises a **RGB** antes de ingresarlas a ResNet50.
- Experimentos con modelos basados exclusivamente en imágenes y modelos **multimodales/híbridos**.
- Incorporación de variables de metadatos:
  - `tipo A`: radiografías.
  - `tipo B`: imágenes de ultrasonido.
- Uso opcional de **CLAHE** como técnica de preprocesamiento.
- Experimentos de **fine-tuning** de la red preentrenada.
- Evaluación del modelo mediante métricas, gráficos y **matrices de confusión**.

## Flujo previsto

El flujo general del proyecto es:

**Datos → Preprocesamiento → Generadores → Modelo → Entrenamiento → Evaluación → Resultados**

## Reporte tecnico

Se explica el desarrollo del experimento en el reporte tecnico comparando resultados y analizando las diferentes versiones obtenidas.
