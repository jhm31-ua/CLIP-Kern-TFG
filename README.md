# CLIP Kern - TFG Universidad de Alicante
Implementación del modelo CLIP (Contrastive Language–Image Pretraining) para tareas de alineación Kern-imagen mediante PyTorch Lightning.

> Este proyecto se basa en la siguiente implementación: 
> 
> Nguyen, M. (2024). *Building CLIP from Scratch*. Disponible en [el siguiente enlace](https://medium.com/correll-lab/building-clip-from-scratch-68f6e42d35f4). Accedido el 26 de agosto de 2024

### Estructura del proyecto

- `data/` — Código asociado al tratamiento de los datos, como la creación de particiones para validación cruzada y la gestión del conjunto de datos personalizado.  
- `model/` — Implementación del modelo CLIP adaptado, con clases que definen su arquitectura y lógica de entrenamiento.  
- `util/` — Funciones auxiliares, definición de parámetros, y scripts de apoyo para el preprocesamiento de datos y otras tareas.

Adicionalmente encontramos:

- `main.py` — Clase principal para entrenar y validar el modelo.  
- `main_tokenizer.py` — Clase para entrenar el tokenizador BPE personalizado.  
- `Dockerfile` — Fichero de definición de contenedores para la tecnología Docker. 
- Otros *scripts* para lanzamiento de experimentos en el servidor.
