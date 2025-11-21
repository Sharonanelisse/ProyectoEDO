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
4. **Ecuaciones de Bernoulli**: `dy/dx + P(x)y = Q(x)y^n`

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
- `P(x) = 1/x, Q(x) = x²`
- `μ = e^(∫(1/x) dx) = x`
- `μy = ∫μQdx`
- `∫(x)(x²)dx = x^4/4 + C`
- Solución: `y = (C + (c^4/4))/x`

### 4. Ecuaciones de Bernoulli

**Forma general**: `y' + P(x)y = Q(x)y^n`

**Método de Sustitución**:

1. Ecuacion diferencial general
2. Hacer u = y^(1-n)
3. Despejar y
4. Dividir ecuación original entre `y' y u`
5. Sustituir ecuacion original
6. Ordernar (Ecuacion Lineal en u)
7. Resolver Lineal (Identificar P y Q)

**Ejemplo**:

- Ecuación: `dy/dx + (1/x)y = xy²` (n=2)
- Sustitución: `u = y^(-1)`
- Despejar `y = u^(1/-1)`
- Derivar y' y u: `dy/dx = - (d/dx u(x))/(u^2(x)`
- Sustituir ecuacion original: `-(d/dx u(x))/(u^2(x)) + (1/x)(1/u(x)) = (x)(1/u(x))^2`
- Ordenar `u' + (-1/x)u = -x`
- Resolver Lineal `P(u) = 1/x, Q(u) = -x`
   - Solucion para u: `u = Cx -x²`
   - Solucion Final: `1/y = Cx - x²`

---

## Guía de Uso

### Ecuaciones Separables

**Formato de entrada**: `expresión en términos de x e y`

Ejemplos:

- `x*y` → dy/dx = xy
- `x**2/(1+y**2)` → dy/dx = x²/(1+y²)
- `exp(x)*y` → dy/dx = e^x · y

### Ecuaciones Exactas

**Formato de entrada**: `M(x,y)|N(x,y)`

Ejemplos:

- `2*x*y|x**2` → (2xy)dx + (x²)dy = 0
- `2*x + y|x` → (2x+y)dx + (x)dy = 0

### Ecuaciones Lineales

**Formato de entrada**: `P(x)|Q(x)`

Ejemplos:

- `2*x|x**2` → dy/dx + 2xy = x²
- `1/x|x` → dy/dx + (1/x)y = x
- `1|exp(x)` → dy/dx + y = e^x

### Ecuaciones de Bernoulli

**Formato de entrada**: `P(x)|Q(x)|n`

Ejemplos:

- `1/x|x|2` → dy/dx + (1/x)y = xy²
- `1|1|3` → dy/dx + y = y³

### Operadores Matemáticos Disponibles

- Suma: `+`
- Resta: `-`
- Multiplicación: `*`
- División: `/`
- Potencia: `**` (ejemplo: `x**2` para x²)
- Exponencial: `exp(x)` para e^x
- Logaritmo natural: `log(x)` para ln(x)
- Seno: `sin(x)`
- Coseno: `cos(x)`
- Tangente: `tan(x)`

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

## Ejemplos de Uso

### Ejemplo 1: Ecuación Separable

**Input**: `x*y`  
**Tipo**: Separable  
**Salida**: `Eq(log(y(x)), C1 + x**2/2)`  
**Interpretación**: `ln(y) = C₁ + x²/2` → `y = Ce^(x²/2)`

### Ejemplo 2: Ecuación Exacta

**Input**: `2*x*y|x**2`  
**Tipo**: Exacta  
**Salida**: `Eq(x**2*y, C1)`  
**Interpretación**: `x²y = C`

### Ejemplo 3: Ecuación Lineal

**Input**: `1|exp(x)`  
**Tipo**: Lineal  
**Salida**: Solución con factor integrante

### Ejemplo 4: Ecuación de Bernoulli

**Input**: `1/x|x|2`  
**Tipo**: Bernoulli  
**Salida**: Solución mediante sustitución v = y^(-1)

---

## Validación y Pruebas

### Casos de Prueba Implementados

1. **Separables**:

   - ✅ `xy` → Separable simple
   - ✅ `x**2/(1+y**2)` → Con funciones complejas

2. **Exactas**:

   - ✅ `2*x*y|x**2` → Exacta directa
   - ✅ Verificación de condición ∂M/∂y = ∂N/∂x

3. **Lineales**:

   - ✅ `2*x|x**2` → Con factor integrante
   - ✅ `1/x|x` → Con P(x) racional

4. **Bernoulli**:
   - ✅ `1/x|x|2` → n=2
   - ✅ Verificación de sustitución correcta

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
- Más tipos de EDO (Ricatti, Clairaut)
- Resolución de sistemas de EDO
- Métodos numéricos (Euler, Runge-Kutta)

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
