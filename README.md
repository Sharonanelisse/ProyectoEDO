# Calculadora de Ecuaciones Diferenciales Ordinarias (EDO)

**Calculadora web interactiva diseñada para resolver y explicar paso a paso Ecuaciones Diferenciales Ordinarias de Primer Orden.**

---

## Descripción del Proyecto

Calculadora web desarrollada con **Vue.js** (frontend) y **Python/FastAPI** (backend), enfocándose no solo en el resultado, sino en la metodología académica de resolución.

### Características Principales

- Entrada Natural: Escribe la ecuación tal como la ves en tu cuaderno (ej: y' + 3y = e^x o (x+y)dx + dy = 0). No requiere sintaxis compleja.
- Detección Automática: El sistema identifica automáticamente las variables dependientes e independientes y el tipo de notación (y' o dy/dx).
- Paso a Paso Académico: Los algoritmos no solo dan la respuesta, sino que desglosan el procedimiento.
- Vista Previa en Vivo: Renderizado LaTeX en tiempo real para verificar que la ecuación se interpreta correctamente antes de calcular.
-Barra de Herramientas Matemática: Botones rápidos para insertar símbolos complejos (√, sin, e (Euler)).

### Tipos de EDO Soportadas

1. **Ecuaciones Separables**: `dy/dx = g(x)h(y)`
2. **Ecuaciones Exactas**: `M(x,y)dx + N(x,y)dy = 0`
3. **Ecuaciones Lineales**: `dy/dx + P(x)y = Q(x)`

---

## Instalación y Ejecución

### Requisitos Previos

- Python 3.8 o superior
- Navegador web moderno
- pip (gestor de paquetes de Python)

### Paso 1: Configurar el Backend

```bash
# Crear entorno virtual (recomendado)
python -m venv venv

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el servidor
python main.py


# Si da error el main.py es porque no esta dentro del venv ejecutar
source venv/bin/activate
# Instalar dependencias
pip install -r requirements.txt

#Luego correr uvicorn para cargar python
uvicorn main:app --reload --port 8000
```

El backend estará disponible en: `http://localhost:8000`

### Paso 2: Abrir el Frontend

Simplemente abra el archivo `index.html` en su navegador web, o puede usar un servidor local:

```bash
# Con Python
python -m http.server 3000

# Luego abrir: http://localhost:3000
```

---

## Fundamento Matemático

### 1. Ecuaciones Separables

**Forma general**: `dy/dx = g(x)h(y)`

**Método**:

# lógica algebraica para identificar y separar los componentes de 'x' y de 'y'.
Ecuacion original: `Toma la expresión despejada (rhs) y crea una cadena de texto en formato LaTeX para mostrársela al usuario.`
Formato Simbolico: `Crea un objeto matemático formal de ecuación (Eq) dentro de SymPy.`
Separación Heurística: `Parte la operación en dos, una que solo contenga 'x' y la otra solo contenga lo de 'y'`

1. Separar variables: `dy/h(y) = g(x)dx`
2. Integrar ambos lados: `∫dy/h(y) = ∫g(x)dx + C`
3. Resultado Implícito: `Mostrar el resultado de las integrales.`
4. Solución General: `Despejar y si es posible.`

**Ejemplo**:

- Ecuación: `dy/dx = xy`
- Separación: `dy/y = x dx`
- Integración: `ln|y| = x²/2 + C`
- Solución: `y = C₁e^(x²/2)`

### 2. Ecuaciones Exactas

**Forma general**: `M(x,y)dx + N(x,y)dy = 0`

**Método**:

1. Verificar exactitud: Comprobar si `∂M/∂y = ∂N/∂x`
2. Integrar M: `F(x,y)` tal que:
   - `∂F/∂x = M(x,y)`
   - `∂F/∂y = N(x,y)`
3. Derivar y Comparar: Derivar `F` respecto a `y` e igualar a `N` para encontrar `h'(y)`.
4. Solución implícita: `F(x,y) = C`

**Ejemplo**:

- Ecuación: `(2xy) + (x²)y' = 0`
- `M = 2xy, N = x²`
- `∂M/∂y = 2x, ∂N/∂x = 2x ✓ (es exacta)`
- `F(x,y) = x²y + h(y)`
- `F/∂y = x²+h'(y) = x²`
   - `h'(y) = 0`
   - `h(y) = ∫(0)dy = 0`
- Solución: `x²y = C`

### 3. Ecuaciones Lineales

**Forma general**: `y' + P(x)y = Q(x)`

**Método del Factor Integrante**:

1. Identificar P y Q
2. Encontrar el factor integrador: `μ(x) = e^(∫P(x)dx)`
3. Resolver usando la formula: `d/dx[μ(x)y] = μ(x)Q(x)`
4. Integrar: `μ(x)y = ∫μ(x)Q(x)dx + C`
5. Despejar: `y = [∫μ(x)Q(x)dx + C]/μ(x)`

**Ejemplo**:

- Ecuación: `y' + (1/x)y = x²`
- `P(x) = 1/x, Q(x) = √√`
- `μ = e^(∫(1/x) dx) = x`
- `μy = ∫μQdx`
- `∫(x)(x²)dx = x^4/4 + C`
- Solución: `y = (C + (c^4/4))/x`


## Guía de Uso

### Ecuaciones Separables

**Formato de entrada**: `expresión en términos de dx/dy y y'`

Ejemplos:

- `x*y` → y' = xy
- `y/(1+x²)` → dy/dx = y/(1+x²)
- `x*y` → (1+x^2)y' =x*y

### Ecuaciones Exactas

**Formato de entrada**: `M(x,y)|N(x,y)`

Ejemplos:

- `(2xy)dx + (x²)dy` → (2xy)dx + (x²)dy = 0
- `(2xy + y²)dx + (x²+2xy)dy` → (2xy+y²)dx + (x²+2xy)dy = 0
- `(y-x)y' = (x+y)` → (y-x)' = (x+y)dy = 0

### Ecuaciones Lineales

**Formato de entrada**: `P(x)|Q(x)`

Ejemplos:

- `y' + (1/x)*y` → y' + (1/x)*y = x²
- `y' + y` → y' + y = e^x
- `dy/dx + 2*y` → dy/dx + 2*y = 4

### Operadores Matemáticos Disponibles

- Suma: `+`
- Resta: `-`
- Multiplicación: `*`
- División: `/`
- Potencia: `^` 
- Logaritmo natural: `log(x)` para ln(x)
- Seno: `sin(x)`
- Coseno: `cos(x)`
- Tangente: `tan(x)`

Hay un menu con varios operadores disponibles

---

## Estructura del Proyecto

```
proyecto-edo/
│
├── main.py                 # Backend FastAPI
├── index.html             # Frontend Vue.js
├── requirements.txt       # Dependencias Python
└── README.md             # Documentación
```
---

## Aprendizajes y Conclusiones

### Aspectos Técnicos

- Implementación de algoritmos simbólicos usando SymPy
- Integración de frontend-backend mediante API REST
- Manejo de expresiones matemáticas en formato texto

### Aspectos Matemáticos

- Comprensión profunda de métodos de solución de EDO
- Identificación automática de tipos de ecuaciones
- Validación de condiciones necesarias (exactitud, linealidad)

### Desafíos Enfrentados

1. Parsing correcto de expresiones matemáticas
2. Manejo de casos especiales y excepciones
3. Presentación clara de pasos de solución

### Mejoras Futuras

- Graficación de soluciones
- Más tipos de EDO (Ricatti, Bernouilli)
- Resolución de sistemas de EDO
- Métodos numéricos (Euler)

---

## Licencia y Créditos

**Autor**: Alex, Sharon, Josimar

**Curso**: Ecuaciones Diferenciales  

**Institución**: USPG

**Fecha**: Noviembre 2025

**Tecnologías Utilizadas**:

- Python 3.x
- FastAPI
- SymPy
- Vue.js 3
- Axios

---

## Soporte

Para preguntas o problemas:

- Revisar la documentación de SymPy: https://docs.sympy.org
- Documentación de FastAPI: https://fastapi.tiangolo.com

---
