# Comandos por dataset

Referencia completa de comandos para correr el proyecto con cada dataset,
tanto en modo CLI (terminal) como en modo servidor (dashboard en browser).

---

## Estructura de carpetas esperada

```
proyecto/
├── run.py
├── data/
│   ├── optdigits.tra          ← ya lo tienes
│   ├── optdigits.tes          ← ya lo tienes
│   ├── MNIST/                 ← se crea automático (torchvision)
│   ├── KMNIST/                ← se crea automático (torchvision)
│   ├── FashionMNIST/          ← se crea automático (torchvision)
│   ├── cifar-10-batches-py/   ← se crea automático (torchvision)
│   ├── UCI HAR Dataset/       ← descargar y descomprimir manualmente
│   │   ├── train/
│   │   │   ├── X_train.txt
│   │   │   └── y_train.txt
│   │   └── test/
│   │       ├── X_test.txt
│   │       └── y_test.txt
│   └── wisdm/                 ← descargar manualmente
│       └── WISDM_ar_v1.1_raw.txt
└── src/
```

---

## Descargas necesarias

| Dataset | Descarga automática | Enlace |
|---------|-------------------|--------|
| OptDigits | No (ya lo tienes) | https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits |
| MNIST | Sí (torchvision) | — |
| KMNIST | Sí (torchvision) | — |
| Fashion-MNIST | Sí (torchvision) | — |
| CIFAR-10 | Sí (torchvision) | — |
| UCI HAR | No | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |
| WISDM | No | https://www.cis.fordham.edu/wisdm/dataset.php |

### Requisitos

```bash
pip install numpy
pip install torch torchvision   # para MNIST, KMNIST, Fashion-MNIST, CIFAR-10
pip install pandas              # para WISDM
```

---

## Modo CLI

Resultados en terminal. Útil para pruebas rápidas.

### OptDigits

```bash
# Con parámetros de prueba (rápido)
python3 run.py cli

# Parámetros completos del paper (lento, ~horas)
python3 run.py cli \
  --dataset optdigits \
  --train data/optdigits.tra \
  --test data/optdigits.tes \
  --generations 1800 \
  --pop-size 100 \
  --max-epochs 5000
```

### MNIST

```bash
# Prueba rápida
python3 run.py cli \
  --dataset mnist \
  --data-dir ./data \
  --generations 10 \
  --pop-size 20 \
  --max-epochs 50

# Experimento completo
python3 run.py cli \
  --dataset mnist \
  --data-dir ./data \
  --generations 1800 \
  --pop-size 100 \
  --max-epochs 5000
```

### KMNIST

```bash
# Prueba rápida
python3 run.py cli \
  --dataset kmnist \
  --data-dir ./data \
  --generations 10 \
  --pop-size 20 \
  --max-epochs 50

# Experimento completo
python3 run.py cli \
  --dataset kmnist \
  --data-dir ./data \
  --generations 1800 \
  --pop-size 100 \
  --max-epochs 5000
```

### Fashion-MNIST

```bash
# Prueba rápida
python3 run.py cli \
  --dataset fashion \
  --data-dir ./data \
  --generations 10 \
  --pop-size 20 \
  --max-epochs 50

# Experimento completo
python3 run.py cli \
  --dataset fashion \
  --data-dir ./data \
  --generations 1800 \
  --pop-size 100 \
  --max-epochs 5000
```

### CIFAR-10

> Nota: CIFAR-10 se aplana a 3072 features (32×32×3). El rendimiento con MLP
> plano será inferior a una CNN — esto es esperado para la comparación experimental.

```bash
# Prueba rápida
python3 run.py cli \
  --dataset cifar10 \
  --data-dir ./data \
  --generations 10 \
  --pop-size 20 \
  --max-epochs 50

# Experimento completo
python3 run.py cli \
  --dataset cifar10 \
  --data-dir ./data \
  --generations 1800 \
  --pop-size 100 \
  --max-epochs 5000
```

### UCI HAR Dataset

> 561 features pre-extraídas, 6 actividades (walking, upstairs, downstairs,
> sitting, standing, laying). Descomprimir en `data/UCI HAR Dataset/`.

```bash
# Prueba rápida
python3 run.py cli \
  --dataset ucihar \
  --data-dir ./data \
  --generations 10 \
  --pop-size 20 \
  --max-epochs 50

# Experimento completo
python3 run.py cli \
  --dataset ucihar \
  --data-dir ./data \
  --generations 1800 \
  --pop-size 100 \
  --max-epochs 5000
```

### WISDM

> 6 features (media + std de aceleración x/y/z por ventana), 6 actividades.
> Colocar `WISDM_ar_v1.1_raw.txt` en `data/wisdm/`.

```bash
# Prueba rápida
python3 run.py cli \
  --dataset wisdm \
  --data-dir ./data/wisdm \
  --generations 10 \
  --pop-size 20 \
  --max-epochs 50

# Experimento completo
python3 run.py cli \
  --dataset wisdm \
  --data-dir ./data/wisdm \
  --generations 1800 \
  --pop-size 100 \
  --max-epochs 5000
```

---

## Modo servidor (dashboard)

Abre el dashboard en el browser en `http://localhost:8765/dashboard.html`.

### OptDigits

```bash
# Con parámetros de prueba (rápido)
python3 run.py server

# Parámetros completos del paper
python3 run.py server \
  --dataset optdigits \
  --train data/optdigits.tra \
  --test data/optdigits.tes \
  --generations 1800 \
  --pop-size 100 \
  --max-epochs 5000
```

### MNIST

```bash
python3 run.py server \
  --dataset mnist \
  --data-dir ./data \
  --generations 50 \
  --pop-size 100 \
  --max-epochs 500
```

### KMNIST

```bash
python3 run.py server \
  --dataset kmnist \
  --data-dir ./data \
  --generations 50 \
  --pop-size 100 \
  --max-epochs 500
```

### Fashion-MNIST

```bash
python3 run.py server \
  --dataset fashion \
  --data-dir ./data \
  --generations 50 \
  --pop-size 100 \
  --max-epochs 500
```

### CIFAR-10

```bash
python3 run.py server \
  --dataset cifar10 \
  --data-dir ./data \
  --generations 50 \
  --pop-size 100 \
  --max-epochs 500
```

### UCI HAR Dataset

```bash
python3 run.py server \
  --dataset ucihar \
  --data-dir ./data \
  --generations 50 \
  --pop-size 100 \
  --max-epochs 500
```

### WISDM

```bash
python3 run.py server \
  --dataset wisdm \
  --data-dir ./data/wisdm \
  --generations 50 \
  --pop-size 100 \
  --max-epochs 500
```

---

## Argumentos disponibles

| Argumento | Descripción | Default CLI | Default servidor |
|-----------|-------------|-------------|-----------------|
| `--dataset` | Dataset a usar | `optdigits` | `optdigits` |
| `--data-dir` | Directorio de datos | `./data` | `./data` |
| `--train` | Ruta explícita `.tra` (solo optdigits) | `None` | `None` |
| `--test` | Ruta explícita `.tes` (solo optdigits) | `None` | `None` |
| `--generations` | Generaciones de evolución | `10` | `50` |
| `--pop-size` | Tamaño de población | `20` | `100` |
| `--max-epochs` | Épocas máximas por sesión | `50` | `500` |
| `--no-dual` | Desactivar fast weights | `False` | `False` |
| `--seed` | Semilla aleatoria | `42` | `42` |
| `--verbose-indiv` | Individuos con log detallado | `3` | `3` |
| `--port` | Puerto del servidor | — | `8765` |
| `--log` | Archivo de log | — | `neuroevo.log` |

---

## Verificar un dataset antes de correr

```bash
# Diagnóstico rápido — verifica que el dataset carga correctamente
python3 src/infrastructure/data/dataset_loader.py optdigits ./data
python3 src/infrastructure/data/dataset_loader.py mnist ./data
python3 src/infrastructure/data/dataset_loader.py kmnist ./data
python3 src/infrastructure/data/dataset_loader.py fashion ./data
python3 src/infrastructure/data/dataset_loader.py cifar10 ./data
python3 src/infrastructure/data/dataset_loader.py ucihar ./data
python3 src/infrastructure/data/dataset_loader.py wisdm ./data/wisdm
```
