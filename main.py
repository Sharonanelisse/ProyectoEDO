"""
Calculadora de Ecuaciones Diferenciales Ordinarias (EDO) de Primer Orden
Backend API con FastAPI

Resuelve EDO de tipo:
- Separables
- Exactas
- Lineales
- Bernoulli

Mejorado con:
- Pasos de integración explícitos para todos los tipos.
- Salida de pasos y soluciones en formato LaTeX.
- Código refactorizado para eliminar duplicados (DRY).
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Tuple, List, Any
from sympy import (
    symbols, Function, Eq, dsolve, diff, integrate, simplify, exp, latex, Symbol,
    Integral, Wild
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
import traceback

# --- Configuración Global ---

# Parser transforms (permite multiplicación implícita)
transformations = (standard_transformations + (implicit_multiplication_application,))

app = FastAPI(title="EDO Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos Pydantic ---

class EDORequest(BaseModel):
    equation: str
    tipo: str
    dep: Optional[str] = None
    indep: Optional[str] = None
    var_dep: Optional[str] = None
    var_indep: Optional[str] = None

    initial_x: Optional[float] = None
    initial_y: Optional[float] = None
    initial_indep: Optional[float] = None
    initial_dep: Optional[float] = None

class EDOResponse(BaseModel):
    success: bool
    solution: str = ""
    solution_latex: str = ""
    particular_solution: str = ""
    steps: List[str] = []
    error: str = ""

# --- Solvers de EDO ---

def format_latex_step(step_text: str, formula: str) -> str:
    """Envuelve una fórmula de LaTeX para MathJax"""
    return f"{step_text} \\( {formula} \\)"

#Ecuaciones separables

def solve_separable(equation_str: str, dep_name: str, indep_name: str) -> Tuple[Any, str, List[str]]:
    """Resuelve EDO Separables: d(dep)/d(indep) = RHS"""
    steps = []
    try:
        # 1. Setup
        x = symbols(indep_name)
        y_func = Function(dep_name)(x)
        indep_sym = symbols(indep_name)
        dep_sym = symbols(dep_name)
        C1 = Symbol("C1")
        local = {
            indep_name: x, dep_name: y_func,
            indep_sym: indep_sym, dep_sym: dep_sym
        }

        # 2. Parseo
        rhs_input = equation_str.strip().split("=", 1)[-1].strip()
        rhs = parse_expr(rhs_input, local_dict=local, transformations=transformations)
        ode = Eq(diff(y_func, x), rhs)
        
        steps.append(f"1. Ecuación original: \\( \\frac{{d{dep_name}}}{{d{indep_name}}} = {latex(rhs)} \\)")
        steps.append(f"2. Ecuación en formato simbólico: \\( {latex(ode)} \\)")
        steps.append("3. Método: Separación de variables")

        # 3. Heurística de Separación
        rhs_for_factor = parse_expr(rhs_input, local_dict={indep_name: indep_sym, dep_name: dep_sym}, transformations=transformations)
        
        # Usamos .match() para una separación más robusta g(x) * h(y)
        g = Wild('g', exclude=[dep_sym])
        h = Wild('h', exclude=[indep_sym])
        g_part, h_part = 1, 1
        
        match = rhs_for_factor.match(g * h)
        if match:
            g_part = match.get(g, 1)
            h_part = match.get(h, 1)
        else: # Si no matchea, intentamos factorizar y separar
            factors = rhs_for_factor.as_ordered_factors()
            for fac in factors:
                if fac.has(indep_sym) and not fac.has(dep_sym):
                    g_part *= fac
                elif fac.has(dep_sym) and not fac.has(indep_sym):
                    h_part *= fac
                else:
                    # Si es mixto (e.g., x+y) o constante, se complica.
                    # Asumimos que el usuario da algo separable.
                    if not fac.free_symbols: g_part *= fac # Constante
                    else: h_part *= fac # Parte mixta o no factorizable
        
        steps.append(f"4. Separación (heurística): RHS ≈ g({indep_name}) · h({dep_name})")
        steps.append(f"   \\( g({indep_name}) \\approx {latex(simplify(g_part))} \\)")
        steps.append(f"   \\( h({dep_name}) \\approx {latex(simplify(h_part))} \\)")

        # 4. Pasos de Integración
        h_inv_part = simplify(1 / h_part)
        g_simp = simplify(g_part)

        steps.append(f"5. Separar variables: \\( {latex(h_inv_part)} \, d{dep_name} = {latex(g_simp)} \, d{indep_name} \\)")

        integral_y = Integral(h_inv_part, dep_sym)
        integral_x = Integral(g_simp, indep_sym)
        steps.append(format_latex_step("6. Plantear integrales:", f"{latex(integral_y)} = {latex(integral_x)} + {latex(C1)}"))

        result_y = integral_y.doit()
        result_x = integral_x.doit()
        steps.append(format_latex_step("7. Resolver integrales:", f"{latex(simplify(result_y))} = {latex(simplify(result_x))} + {latex(C1)}"))

        # 5. Solución Final
        sol = dsolve(ode, y_func)
        sol_latex = latex(sol)
        steps.append(format_latex_step("8. Solución general (despejada):", sol_latex))

        return sol, sol_latex, steps

    except Exception as e:
        print(f"Error en separable: {e}\n{traceback.format_exc()}")
        raise Exception(f"Error resolviendo separable: {str(e)}")

# Ecuaciones exactas
def solve_exact(M_str: str, N_str: str, dep_name: str, indep_name: str) -> Tuple[Any, str, List[str]]:
    """Resuelve EDO exactas: M dx + N dy = 0"""
    steps = []
    try:
        # 1. Setup
        x = symbols(indep_name)
        y = symbols(dep_name)
        C1 = Symbol("C1")
        local = {indep_name: x, dep_name: y}

        # 2. Parseo
        M = parse_expr(M_str, local_dict=local, transformations=transformations)
        N = parse_expr(N_str, local_dict=local, transformations=transformations)

        steps.append(f"1. Ecuación: \\( M({indep_name},{dep_name}) d{indep_name} + N({indep_name},{dep_name}) d{dep_name} = 0 \\)")
        steps.append(f"   \\( M = {latex(M)} \\)")
        steps.append(f"   \\( N = {latex(N)} \\)")

        # 3. Verificar exactitud
        dM_dy = diff(M, y)
        dN_dx = diff(N, x)

        steps.append(f"2. Verificar exactitud: \\( \\frac{{\\partial M}}{{\\partial {dep_name}}} = {latex(dM_dy)} \\)")
        steps.append(f"   \\( \\frac{{\\partial N}}{{\\partial {indep_name}}} = {latex(dN_dx)} \\)")

        if simplify(dM_dy - dN_dx) != 0:
            steps.append("   ❌ \\( \\frac{{\\partial M}}{{\\partial {dep_name}}} \\neq \\frac{{\\partial N}}{{\\partial {indep_name}}} \\). La ecuación no es exacta.")
            raise Exception("La ecuación no es exacta")

        steps.append("   ✓ La ecuación es exacta.")

        # 4. Integrar M
        int_M_dx = Integral(M, x)
        F_partial = int_M_dx.doit()
        steps.append(f"3. Integrar M respecto a {indep_name}: \\( F = {latex(int_M_dx)} = {latex(F_partial)} + g({dep_name}) \\)")

        # 5. Encontrar g'(y)
        dF_dy = diff(F_partial, y)
        g_prime = simplify(N - dF_dy)
        steps.append(f"4. Encontrar g'({dep_name}): \\( g'({dep_name}) = N - \\frac{{\\partial F}}{{\\partial {dep_name}}} = {latex(N)} - ({latex(dF_dy)}) = {latex(g_prime)} \\)")

        # 6. Integrar g'(y)
        int_g_prime = Integral(g_prime, y)
        g = int_g_prime.doit()
        steps.append(f"5. Integrar g': \\( g({dep_name}) = {latex(int_g_prime)} = {latex(g)} \\)")

        # 7. Solución
        F_complete = F_partial + g
        sol = Eq(F_complete, C1)
        sol_latex = latex(sol)
        steps.append(format_latex_step("6. Solución implícita F(x,y) = C:", sol_latex))

        return sol, sol_latex, steps

    except Exception as e:
        print(f"Error en exacta: {e}\n{traceback.format_exc()}")
        raise Exception(f"Error resolviendo exacta: {str(e)}")

# Ecuaciones lineales y Bernoulli (comparten pasos)
def _solve_linear_core(P: Any, Q: Any, x: Symbol, y_func: Function, C1: Symbol, indep_name: str, dep_name: str, steps_list: List[str]) -> Tuple[Any, str, List[str]]:
    """
    Función interna (DRY) que resuelve la EDO lineal y = (∫μQ + C) / μ.
    Es usada por solve_linear y solve_bernoulli.
    """
    
    # 1. Factor integrante
    int_P = Integral(P, x)
    P_integral = int_P.doit()
    steps_list.append(f"   a. Calcular integral de P: \\( \\int P d{indep_name} = {latex(int_P)} = {latex(P_integral)} \\)")

    mu = simplify(exp(P_integral))
    steps_list.append(f"   b. Factor integrante: \\( \\mu({indep_name}) = e^{{\\int P d{indep_name}}} = {latex(mu)} \\)")

    # 2. Integrar μQ
    right_side = simplify(mu * Q)
    int_mu_Q = Integral(right_side, x)
    integral_result = int_mu_Q.doit()
    steps_list.append(f"   c. Integrar μQ: \\( \\int \\mu Q d{indep_name} = {latex(int_mu_Q)} = {latex(integral_result)} \\)")

    # 3. Solución
    solution_expr = simplify((integral_result + C1) / mu)
    sol = Eq(y_func, solution_expr)
    sol_latex = latex(sol)
    steps_list.append(format_latex_step(f"   d. Solución {dep_name} = (∫μQ + C) / μ:", sol_latex))
    
    return sol, sol_latex, steps_list, solution_expr # Devolvemos expr para Bernoulli

def solve_linear(P_str: str, Q_str: str, dep_name: str, indep_name: str) -> Tuple[Any, str, List[str]]:
    """Resuelve EDO lineal: y' + P(x)y = Q(x)"""
    steps = []
    try:
        # 1. Setup
        x = symbols(indep_name)
        y = symbols(dep_name) # Símbolo simple para parsear P y Q
        y_func = Function(dep_name)(x)
        C1 = Symbol("C1")
        
        # P y Q solo deben depender de indep_name (x)
        local_P_Q = {indep_name: x}

        # 2. Parseo
        P = parse_expr(P_str, local_dict=local_P_Q, transformations=transformations)
        Q = parse_expr(Q_str, local_dict=local_P_Q, transformations=transformations)

        steps.append(f"1. Ecuación lineal: \\( \\frac{{d{dep_name}}}{{d{indep_name}}} + P({indep_name}){dep_name} = Q({indep_name}) \\)")
        steps.append(f"   \\( P({indep_name}) = {latex(P)} \\)")
        steps.append(f"   \\( Q({indep_name}) = {latex(Q)} \\)")
        steps.append("2. Resolver usando factor integrante \\( \\mu = e^{{\\int P d{indep_name}}} \\):")

        # 3. Llamar al solver (DRY)
        sol, sol_latex, steps, _ = _solve_linear_core(P, Q, x, y_func, C1, indep_name, dep_name, steps)
        
        # Renombrar último paso para que sea el final
        steps[-1] = format_latex_step("3. Solución general:", sol_latex)

        return sol, sol_latex, steps

    except Exception as e:
        print(f"Error en lineal: {e}\n{traceback.format_exc()}")
        raise Exception(f"Error resolviendo lineal: {str(e)}")

# Ecuaciones de Bernoulli
def solve_bernoulli(P_str: str, Q_str: str, n_val: str, dep_name: str, indep_name: str) -> Tuple[Any, str, List[str]]:
    """Resuelve Bernoulli: y' + P(x)y = Q(x)y^n"""
    steps = []
    try:
        # 1. Setup
        x = symbols(indep_name)
        y = symbols(dep_name)
        y_func = Function(dep_name)(x)
        C1 = Symbol("C1")
        local_P_Q = {indep_name: x}

        # 2. Parseo
        P = parse_expr(P_str, local_dict=local_P_Q, transformations=transformations)
        Q = parse_expr(Q_str, local_dict=local_P_Q, transformations=transformations)
        n = simplify(parse_expr(n_val))

        steps.append(f"1. Ecuación de Bernoulli: \\( \\frac{{d{dep_name}}}{{d{indep_name}}} + P({indep_name}){dep_name} = Q({indep_name}){dep_name}^{n} \\)")
        steps.append(f"   \\( P = {latex(P)} \\), \\( Q = {latex(Q)} \\), \\( n = {latex(n)} \\)")

        if n == 0 or n == 1:
            steps.append("   (n=0 o n=1, la ecuación ya es lineal. Resolviendo como lineal...)")
            sol, sol_latex, steps_linear = solve_linear(P_str, Q_str, dep_name, indep_name)
            steps.extend(steps_linear)
            return sol, sol_latex, steps

        # 3. Sustitución v = y^(1-n)
        n_float = float(n) # para cálculo
        v_exp = 1 - n
        steps.append(f"2. Sustitución: \\( v = {dep_name}^{{{v_exp}}} \\)")
        
        P_new = simplify((1 - n) * P)
        Q_new = simplify((1 - n) * Q)
        steps.append(f"3. Ecuación lineal en v: \\( \\frac{{dv}}{{d{indep_name}}} + ({latex(P_new)}) v = {latex(Q_new)} \\)")
        
        # 4. Resolver EDO lineal en v
        v = symbols('v')
        v_func = Function('v')(x)
        
        # Llamamos al core, pero con las variables de 'v'
        _, _, steps, v_sol_expr = _solve_linear_core(P_new, Q_new, x, v_func, C1, indep_name, 'v', steps)
        
        # 5. Volver a 'y'
        # v = y^(1-n)  =>  y = v^(1 / (1-n))
        y_exp = simplify(1 / (1 - n))
        dep_solution = simplify(v_sol_expr ** y_exp)
        sol = Eq(y_func, dep_solution)
        sol_latex = latex(sol)
        
        steps.append(format_latex_step(f"4. Volver a {dep_name} (usando \\( {dep_name} = v^{{{latex(y_exp)}}} \\) ):", sol_latex))

        return sol, sol_latex, steps

    except Exception as e:
        print(f"Error en bernoulli: {e}\n{traceback.format_exc()}")
        raise Exception(f"Error resolviendo Bernoulli: {str(e)}")


# --- Endpoint principal de API ---

@app.post("/solve", response_model=EDOResponse)
async def solve_edo(request: Request):
    """
    Endpoint principal que recibe el request, delega al solver
    y devuelve la solución formateada.
    """
    body = await request.json()
    try:
        # 1. Normalizar Nombres de Variables
        dep_name = body.get("dep") or body.get("var_dep") or "y"
        indep_name = body.get("indep") or body.get("var_indep") or "x"

        # 2. Normalizar Condiciones Iniciales (PVI)
        initial_x = body.get("initial_x") if body.get("initial_x") is not None else body.get("initial_indep")
        initial_y = body.get("initial_y") if body.get("initial_y") is not None else body.get("initial_dep")

        equation = body.get("equation")
        tipo = body.get("tipo")

        if not equation or not tipo:
            raise HTTPException(status_code=400, detail="Faltan 'equation' o 'tipo'")
        if dep_name == indep_name:
            raise HTTPException(status_code=400, detail="Variable dependiente e independiente no pueden ser iguales")

        # 3. Dispatch (Delegar al solver)
        
        # Inicializar variables de respuesta
        sol: Any = None
        sol_latex: str = ""
        steps: List[str] = []

        if tipo == "separable":
            sol, sol_latex, steps = solve_separable(equation, dep_name, indep_name)

        elif tipo == "exacta":
            parts = equation.split("|")
            if len(parts) != 2:
                raise HTTPException(status_code=400, detail="Formato para exacta: M(x,y) | N(x,y)")
            sol, sol_latex, steps = solve_exact(parts[0].strip(), parts[1].strip(), dep_name, indep_name)

        elif tipo == "lineal":
            parts = equation.split("|")
            if len(parts) != 2:
                raise HTTPException(status_code=400, detail="Formato para lineal: P(x) | Q(x)")
            sol, sol_latex, steps = solve_linear(parts[0].strip(), parts[1].strip(), dep_name, indep_name)

        elif tipo == "bernoulli":
            parts = equation.split("|")
            if len(parts) != 3:
                raise HTTPException(status_code=400, detail="Formato para bernoulli: P(x) | Q(x) | n")
            sol, sol_latex, steps = solve_bernoulli(parts[0].strip(), parts[1].strip(), parts[2].strip(), dep_name, indep_name)

        else:
            raise HTTPException(status_code=400, detail=f"Tipo de EDO no soportado: {tipo}")
        # 4. Manejo de PVI (si aplica)
        particular_sol = ""
        # (Lógica PVI iría aquí si se implementa)

        # 5. Devolver Respuesta
        return EDOResponse(
            success=True,
            solution=str(sol),
            solution_latex=f"\\( {sol_latex} \\)",
            
            particular_solution=particular_sol,
            steps=steps
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error en el endpoint /solve: {e}\n{traceback.format_exc()}")
        return EDOResponse(
            success=False,
            error=str(e),
            steps=[]
        )


# --- Endpoints Estáticos ---

@app.get("/")
async def root():
    return {
        "message": "API Calculadora EDO",
        "version": "1.1 (Refactorizada)",
        "tipos_soportados": ["separable", "exacta", "lineal", "bernoulli"]
    }

@app.get("/health")
async def health():
    return {"status": "ok"}


# --- Ejecución ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)