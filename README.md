# OmniVLA CPU + Docker (Edge Deployment)

OmniVLA es un modelo de política de aprendizaje por imitación que combina visión, lenguaje y acción para generar trayectorias a partir de imágenes de estado actual y objetivo.

Este proyecto corresponde a la adaptación, corrección y despliegue del modelo OmniVLA-edge en un entorno local utilizando CPU + Docker, como parte de un reto académico de Programación Distribuida y Paralela.

---

##  Objetivo del proyecto

- Verificar el funcionamiento del modelo OmniVLA-edge en entorno local.
- Adaptar el modelo para ejecución en CPU (sin GPU).
- Corregir errores del repositorio original.
- Empaquetar el sistema en un contenedor Docker reproducible.
- Evaluar inferencias usando imágenes egocéntricas (estado actual vs objetivo).

---

##  Descripción del modelo

OmniVLA es un modelo que integra:

-  Visión (imágenes del entorno)
-  Lenguaje (condiciones o instrucciones)
-  Acción (salida de control o trayectoria)

La versión Edge está optimizada para entornos con recursos limitados, permitiendo su ejecución sin GPUs dedicadas.

Repositorio original:
https://github.com/NHirose/OmniVLA

---

##  Metodología de implementación

### 1. Corrección del repositorio
- Ajuste de errores en `pyproject.toml`
- Corrección de estructura de dependencias
- Resolución de conflictos de instalación

### 2. Actualización de dependencias
- Sustitución de librerías obsoletas por versiones compatibles
- Mejora de estabilidad del entorno de ejecución
- Eliminación de conflictos entre versiones

### 3. Adaptación a CPU
- Modificación del dispositivo de inferencia (`cuda → cpu`)
- Ajustes en la carga del modelo y pipeline de ejecución
- Optimización para entornos sin GPU

### 4. Contenerización con Docker
- Creación de `Dockerfile`
- Configuración de `docker-compose.yml`
- Encapsulamiento completo del entorno
- Garantía de reproducibilidad del proyecto

---

##  Requisitos del sistema

- Windows con WSL (Ubuntu)
- Docker Desktop instalado
- Mínimo 50 GB de espacio libre en disco
- Conexión a internet para descarga de dependencias y modelo

---

##  Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/Bransteek/Omnivla_CPU_OV.git
cd Omnivla_CPU_OV
```

---

### 2. Descargar y preparar el modelo

Este paso es obligatorio, ya que descarga el modelo OmniVLA-edge:

```bash
sh start.sh
```

---

### 3. Construir la imagen Docker

```bash
sudo docker compose build
```

---

### 4. Ejecutar el sistema

```bash
sudo docker compose up omnivla-cpu
```

---

##  Optimización con OpenVINO (opcional)

```bash
sudo docker compose run --rm omnivla-cpu python inference/convert_to_openvino.py
```

Luego de la optimización, ejecutar nuevamente:

```bash
sudo docker compose up omnivla-cpu
```

---

##  Entrada de datos

Ubicación:

```
inference/
```

Archivos:

- current_img.png → estado actual del entorno
- goal_img.png → estado objetivo

 Para usar imágenes personalizadas, reemplazar estos archivos manteniendo el mismo nombre.

---

##  Salida del modelo

El resultado se guarda en:

```
inference/0_ex_omnivla_edge.jpg
```

---

##  Notas importantes

- El modelo está optimizado para CPU.
- Docker garantiza la reproducibilidad del entorno.
- El tiempo de inferencia depende del hardware.

---


Proyecto académico desarrollado para la asignatura de Programación Distribuida y Paralela.
