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
from typing import List, Any
from sympy import (
    symbols, Function, Eq, dsolve, diff, integrate, simplify, exp, latex, Symbol,
    Integral, solve, E, Derivative, Wild, Pow
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
)
import traceback
import re

transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))

app = FastAPI(title="EDO Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EDOResponse(BaseModel):
    success: bool
    solution: str = ""
    solution_latex: str = ""
    steps: List[str] = []
    error: str = ""
    detected_vars: str = ""

def format_step(title: str, math: str) -> str:
    return f"<strong>{title}</strong><br>\\( {math} \\)"

# --- Preprocesamiento ---
def preprocess_equation(eq_str: str):
    eq_clean = eq_str.strip()
    eq_clean = re.sub(r"e\^([+-]?\d+)([a-zA-Z])", r"e^(\1*\2)", eq_clean)
    eq_clean = re.sub(r"e\^([a-zA-Z])", r"e^(\1)", eq_clean)
    eq_clean = re.sub(r"(\d)([a-zA-Z\(])", r"\1*\2", eq_clean)
    eq_clean = re.sub(r"(\))([a-zA-Z0-9\(])", r"\1*\2", eq_clean)
    
    token_dydx = "DYDX"
    dep_str, indep_str = "y", "x"

    has_dx = bool(re.search(r"\bdx\b", eq_clean))
    has_dy = bool(re.search(r"\bdy\b", eq_clean))
    eq_sym_str = eq_clean

    if has_dx and has_dy:
        eq_sym_str = re.sub(r"\bdx\b", "1", eq_sym_str)
        eq_sym_str = re.sub(r"\bdy\b", token_dydx, eq_sym_str)
    else:
        match_frac = re.search(r"d([a-zA-Z]+)\s*/\s*d([a-zA-Z]+)", eq_clean)
        match_prime = re.search(r"([a-zA-Z]+)'", eq_clean)
        if match_frac:
            dep_str, indep_str = match_frac.group(1), match_frac.group(2)
            eq_sym_str = re.sub(r"d" + dep_str + r"\s*/\s*d" + indep_str, token_dydx, eq_sym_str)
        elif match_prime:
            dep_str = match_prime.group(1)
            temp = re.sub(r"(sin|cos|tan|exp|log|ln|sqrt|e|dx|dy)", "", eq_clean)
            vars_found = set(re.findall(r"[a-zA-Z]", temp))
            if dep_str in vars_found: vars_found.remove(dep_str)
            if 'x' in vars_found: indep_str = 'x'
            elif 't' in vars_found: indep_str = 't'
            elif vars_found: indep_str = list(vars_found)[0]
            eq_sym_str = eq_sym_str.replace(f"{dep_str}'", token_dydx)
        else:
            if not (has_dx or has_dy): raise Exception("No se detectó derivada (y', dy/dx) ni diferenciales.")

    if "=" in eq_sym_str:
        lhs, rhs = eq_sym_str.split("=", 1)
        eq_sym_str = f"({lhs}) - ({rhs})"
    
    x = symbols(indep_str)
    y = symbols(dep_str)
    dydx_sym = symbols(token_dydx)
    local_dict = {indep_str: x, dep_str: y, token_dydx: dydx_sym, 'e': E}
    
    try:
        implicit_expr = parse_expr(eq_sym_str, local_dict=local_dict, transformations=transformations)
    except Exception as e:
        raise Exception(f"Sintaxis inválida: {e}")
    
    solved = solve(implicit_expr, dydx_sym)
    readable_eq = ""
    if solved:
        readable_eq = latex(Eq(Derivative(Function(dep_str)(x), x), solved[0]))
    else:
        readable_eq = latex(Eq(implicit_expr, 0))

    return implicit_expr, dydx_sym, dep_str, indep_str, readable_eq

# --- SOLVERS (Separable, Exacta, Lineal NO TOCADOS) ---

def solve_separable(implicit_expr, dydx_sym, dep, indep):
    x, y = symbols(indep), symbols(dep)
    y_func = Function(dep)(x)
    C1 = Symbol("C")
    steps = []
    solved = solve(implicit_expr, dydx_sym)
    if not solved: raise Exception("No se pudo despejar la derivada.")
    rhs = solved[0]
    ode = Eq(diff(y_func, x), rhs)
    steps.append(f"1. Ecuación original: \\( \\frac{{d{dep}}}{{d{indep}}} = {latex(rhs)} \\)")
    steps.append(f"2. Ecuación en formato simbólico: \\( {latex(ode)} \\)")
    steps.append("3. Método: Separación de variables")
    g = Wild('g', exclude=[dep])
    h = Wild('h', exclude=[indep])
    g_part, h_part = 1, 1
    match = rhs.match(g * h)
    if match:
        g_part, h_part = match.get(g, 1), match.get(h, 1)
    else:
        factors = rhs.as_ordered_factors()
        for fac in factors:
            if fac.has(indep) and not fac.has(dep): g_part *= fac
            elif fac.has(dep) and not fac.has(indep): h_part *= fac
            else:
                if not fac.free_symbols: g_part *= fac 
                else: h_part *= fac
    steps.append(f"4. Separación (heurística): RHS ≈ g({indep}) · h({dep})")
    steps.append(f"   \\( g({indep}) \\approx {latex(simplify(g_part))} \\)")
    steps.append(f"   \\( h({dep}) \\approx {latex(simplify(h_part))} \\)")
    h_inv_part = simplify(1 / h_part)
    g_simp = simplify(g_part)
    steps.append(f"5. Separar variables: \\( {latex(h_inv_part)} \, d{dep} = {latex(g_simp)} \, d{indep} \\)")
    integral_y = Integral(h_inv_part, y)
    integral_x = Integral(g_simp, x)
    steps.append(format_step("6. Plantear integrales:", f"{latex(integral_y)} = {latex(integral_x)} + {latex(C1)}"))
    result_y = integral_y.doit()
    result_x = integral_x.doit()
    steps.append(format_step("7. Resolver integrales:", f"{latex(simplify(result_y))} = {latex(simplify(result_x))} + {latex(C1)}"))
    ode_final = Eq(diff(y_func, x), rhs.subs(y, y_func))
    sol = dsolve(ode_final, y_func)
    sol_latex = latex(sol)
    steps.append(format_step("8. Solución general (despejada):", sol_latex))
    return sol, sol_latex, steps

def solve_exact(implicit_expr, dydx_sym, dep, indep):
    x, y = symbols(indep), symbols(dep)
    C1 = Symbol("C")
    steps = []
    expr_expanded = implicit_expr.expand()
    N = simplify(expr_expanded.coeff(dydx_sym))
    M = simplify(expr_expanded.as_independent(dydx_sym)[0])
    steps.append(format_step("1. Verificar la exactitud", f"M = {latex(M)}, \\quad N = {latex(N)}"))
    My = diff(M, y)
    Nx = diff(N, x)
    msg_exact = "Si hay exactitud" if simplify(My - Nx) == 0 else "No es exacta"
    steps.append(format_step(f"Derivadas parciales ({msg_exact})", f"M_{dep} = {latex(My)} \\quad , \\quad N_{indep} = {latex(Nx)}"))
    if simplify(My - Nx) != 0: raise Exception("Las derivadas parciales no son iguales. No es exacta.")
    F_partial = integrate(M, x)
    steps.append(format_step(f"2. Integrar M respecto a {indep}", f"F = \\int ({latex(M)}) d{indep} = {latex(F_partial)} + h({dep})"))
    Fy = diff(F_partial, y)
    h_prime = simplify(N - Fy)
    steps.append(format_step(f"3. Derivar respecto a {dep} y comparar con N", f"\\frac{{\\partial F}}{{\\partial {dep}}} = {latex(Fy)} + h'({dep}) = {latex(N)}"))
    steps.append(format_step(f"Despejar h'({dep})", f"h'({dep}) = {latex(h_prime)}"))
    h_val = integrate(h_prime, y)
    steps.append(format_step(f"Integrar h'({dep})", f"h({dep}) = \\int ({latex(h_prime)}) d{dep} = {latex(h_val)}"))
    F_final = F_partial + h_val
    sol = Eq(F_final, C1)
    steps.append(format_step("4. Solución implícita F(x,y) = C", latex(sol)))
    return sol, latex(sol), steps

def solve_linear(implicit_expr, dydx_sym, dep, indep):
    x, y = symbols(indep), symbols(dep)
    y_func = Function(dep)(x)
    C1 = Symbol("C")
    steps = []
    solved = solve(implicit_expr, dydx_sym)
    if not solved: raise Exception("No se pudo llevar a forma estándar.")
    rhs = solved[0]
    expanded = simplify(rhs).expand()
    coeff_y = expanded.coeff(y, 1)
    coeff_const = expanded.coeff(y, 0)
    P = simplify(-coeff_y)
    Q = simplify(coeff_const)
    steps.append(format_step("1. Identificar P y Q", f"P({indep})={latex(P)}, \\quad Q({indep})={latex(Q)}"))
    int_P = integrate(P, x)
    mu = simplify(exp(int_P))
    steps.append(format_step("2. Factor Integrante", f"\\mu = e^{{\\int {latex(P)} d{indep}}} = {latex(mu)}"))
    steps.append(format_step("3. Resolver usando fórmula:", f"\\mu {dep} = \\int \\mu Q d{indep}"))
    rhs_int = simplify(mu * Q)
    res_int = integrate(rhs_int, x)
    steps.append(format_step(f"Integrar el lado derecho", f"\\int ({latex(mu)})({latex(Q)}) d{indep} = {latex(res_int)} + C"))
    sol_expr = simplify((res_int + C1) / mu)
    sol = Eq(y_func, sol_expr)
    steps.append(format_step(f"Despejar {dep} (Solución Final)", latex(sol)))
    return sol, latex(sol), steps

# --- BERNOULLI MODIFICADO ---

def solve_bernoulli(implicit_expr, dydx_sym, dep, indep):
    x, y = symbols(indep), symbols(dep)
    y_func = Function(dep)(x)
    C1 = Symbol("C")
    steps = []

    solved = solve(implicit_expr, dydx_sym)
    if not solved: raise Exception("Error despejando y'.")
    rhs = solved[0]

    expanded = simplify(rhs).expand()
    terms = expanded.as_ordered_terms()
    P_val, Q_val, n_val = 0, 0, 0
    
    for t in terms:
        ty = t.as_independent(y)[1]
        tx = t.as_independent(y)[0]
        if ty == y: P_val = simplify(-tx)
        elif ty == 1: Q_val = tx
        else:
            b, e = ty.as_base_exp()
            if b == y: n_val, Q_val = e, tx
    
    # 1. Ecuación General
    steps.append(format_step("1. Escribir la ecuación diferencial general", 
                             f"{dep}' + ({latex(P_val)}){dep} = ({latex(Q_val)}){dep}^{{{n_val}}}"))

    # 2. Variable u
    u_exp = simplify(1 - n_val)
    steps.append(format_step(f"2. Hacer u = y^(1-n) (n={n_val})", f"u = {dep}^{{{latex(u_exp)}}}"))

    # 3. Despejar y
    y_in_u = Symbol('u') ** simplify(1/u_exp)
    steps.append(format_step(f"3. Despejar {dep}", f"{dep} = u^{{1/{latex(u_exp)}}} = {latex(y_in_u)}"))

    # 4. Derivar
    u_sym = Function('u')(x)
    y_sub = u_sym ** simplify(1/u_exp)
    dy_sub = diff(y_sub, x)
    steps.append(format_step(f"4. Derivar '{dep}' y u", 
                             f"\\frac{{d{dep}}}{{d{indep}}} = {latex(dy_sub)}"))

    # 5. Sustituir
    steps.append(format_step("5. Sustituir en la ecuación original", 
                             f"({latex(dy_sub)}) + ({latex(P_val)})({latex(y_sub)}) = ({latex(Q_val)})({latex(y_sub)})^{{{n_val}}}"))

    # 6. Ordenar (Lineal)
    P_new = simplify((1 - n_val) * P_val)
    Q_new = simplify((1 - n_val) * Q_val)
    steps.append(format_step("6. Ordenar (Ecuación Lineal)", 
                             f"u' + ({latex(P_new)})u = {latex(Q_new)}"))

    # 7. Resolver Lineal
    steps.append(format_step("7. Utilizar el método de ecuaciones lineales", 
                             f"Identifica P({indep})={latex(P_new)} y Q({indep})={latex(Q_new)}"))
    
    mu_u = simplify(exp(integrate(P_new, x)))
    res_u = integrate(simplify(mu_u * Q_new), x)
    
    # Solución explícita de u (Esta es la que coincide con tu cuaderno)
    u_sol_val = (res_u + C1) / mu_u
    
    steps.append(format_step(f"Resolver para u (Factor integrante \\( \\mu = {latex(mu_u)} \\))", 
                             f"u = {latex(u_sol_val)}"))

    # Volver a y
    y_final = u_sol_val ** simplify(1/u_exp)
    sol = Eq(y_func, y_final)
    
    steps.append(format_step(f"Volver a variable original (Solución Final para {dep})", latex(sol)))

    return sol, latex(sol), steps

@app.post("/solve", response_model=EDOResponse)
async def solve_edo(request: Request):
    body = await request.json()
    if not body.get("equation") or not body.get("tipo"): 
        raise HTTPException(400, "Faltan datos")
    try:
        expr, dydx, dep, indep, interp_latex = preprocess_equation(body.get("equation"))
        msg = f"Var: {dep}, {indep}"
        if body.get("tipo") == "separable": sol, ltx, stp = solve_separable(expr, dydx, dep, indep)
        elif body.get("tipo") == "exacta": sol, ltx, stp = solve_exact(expr, dydx, dep, indep)
        elif body.get("tipo") == "lineal": sol, ltx, stp = solve_linear(expr, dydx, dep, indep)
        elif body.get("tipo") == "bernoulli": sol, ltx, stp = solve_bernoulli(expr, dydx, dep, indep)
        else: raise HTTPException(400, "Tipo inválido")
        
        return EDOResponse(success=True, solution=str(sol), solution_latex=f"\\( {ltx} \\)", steps=stp, detected_vars=msg)
    except Exception as e:
        return EDOResponse(success=False, error=str(e), steps=[])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)