#!/usr/bin/env python3
"""
02_solve_simulation.py
Loads the generated .msh file into DOLFINx and solves the transient multiphysics system:
  1. Incompressible Stokes Squeeze Flow (Taylor-Hood P2/P1 element formulation)
  2. Transient Energy Equation (Conduction + Exothermic Heat Generation)
  3. Arrhenius Crosslinking Kinetics & Cure-Dependent Dynamic Rheology
"""

import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem
from dolfinx.io import gmsh as gmshio, XDMFFile
from dolfinx.fem.petsc import LinearProblem


def _owned(f):
    """Owned (non-ghost) DOF values of a scalar Function on this MPI rank."""
    return f.x.array[: f.function_space.dofmap.index_map.size_local]


def global_mean(f):
    """Mean of a scalar Function's DOF values across all MPI ranks."""
    comm = f.function_space.mesh.comm
    vals = _owned(f)
    return comm.allreduce(vals.sum(), op=MPI.SUM) / comm.allreduce(vals.size, op=MPI.SUM)


def global_max(f):
    """Max of a scalar Function's DOF values across all MPI ranks."""
    comm = f.function_space.mesh.comm
    vals = _owned(f)
    return comm.allreduce(vals.max() if vals.size else -np.inf, op=MPI.MAX)


def log(msg):
    """Print from rank 0 only, so MPI runs don't duplicate output."""
    if MPI.COMM_WORLD.rank == 0:
        print(msg, flush=True)


def run_simulation(mesh_filename="die_bonding_mesh.msh"):
    if not os.path.exists(mesh_filename):
        raise FileNotFoundError(f"Mesh file '{mesh_filename}' not found. Please run 01_generate_mesh.py first.")

    log(f"[Solver] Loading mesh '{mesh_filename}' into DOLFINx...")
    
    # Read mesh and physical boundary topology tags:
    #   - facet_tags == 1 : Bottom Substrate Interface (Fixed, Heated)
    #   - facet_tags == 2 : Top Die Interface (Moving Bond Head)
    #   - facet_tags == 3 : Lateral Outlets (Atmospheric Pressure Release)
    mesh_data = gmshio.read_from_msh(mesh_filename, MPI.COMM_WORLD, rank=0, gdim=2)
    domain, cell_tags, facet_tags = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags

    # ==========================================================================
    # 1. MATERIAL & PHYSICAL PARAMETERS (SI/Metric Units: mm, g, s, J, K, MPa)
    # ==========================================================================
    # Thermophysical Properties of Epoxy:
    rho = 1.2e-3      # Density ρ [g/mm³]
    Cp = 1.5          # Specific heat capacity C_p [J/(g·K)]
    k_th = 0.8e-3     # Thermal conductivity k [W/(mm·K)]
    Delta_H = 300.0   # Total exothermic heat of reaction ΔH [J/g]

    # Reaction Kinetics (nth-Order Arrhenius Model):
    A_rate = 1.0e5    # Pre-exponential frequency factor A [1/s]
    E_a = 5.0e4       # Reaction activation energy E_a [J/mol]
    R_gas = 8.314     # Universal gas constant R [J/(mol·K)]
    n_rxn = 1.5       # Reaction order n [-]

    # Rheological Viscosity Constants:
    # (mm-N-s system: stress in MPa = N/mm², so viscosity in MPa·s; 10 Pa·s = 1.0e-5 MPa·s)
    mu_0 = 1.0e-5     # Uncured zero-shear dynamic viscosity μ_0 [MPa·s] (= 10 Pa·s)
    C_mu = 3.5        # Cure-thickening exponential coefficient C_μ [-]

    # Kinematic Process Operating Parameters:
    V_press = 0.01    # Bond Head downward pressing speed [mm/s]
    T_hot = 423.15    # Isothermal substrate heating stage temperature T_hot = 150°C [K]

    # ==========================================================================
    # 2. HYDRODYNAMICS: STOKES SQUEEZE FLOW FORMULATION
    #
    # Physical Rationale:
    #   In micro-gap die bonding (gap height H ~ 50 μm << die length L ~ 2 mm),
    #   the Reynolds number Re = (ρ * V * H) / μ << 1. Inertial forces are
    #   negligible compared to viscous forces, reducing Navier-Stokes to Stokes.
    #
    # Governing PDE (Strong Form):
    #   -∇ · [ 2μ(α) ε(u) ] + ∇p = 0    (Momentum Balance)
    #   ∇ · u = 0                        (Incompressibility / Mass Balance)
    #   where Strain Rate Tensor ε(u) = 1/2 * (∇u + (∇u)ᵀ)
    # ==========================================================================

    # Discrete Function Space Setup:
    # Taylor-Hood Mixed Element Pair: Quadratic Vector (P2) for Velocity u,
    # Linear Scalar (P1) for Pressure p.
    # Numerical Rationale: Satisfies the discrete Ladyzhenskaya-Babuška-Brezzi (LBB)
    # inf-sup condition to prevent spurious pressure oscillations.
    gdim = domain.geometry.dim
    v_elem = element("Lagrange", domain.basix_cell(), 2, shape=(gdim,))
    p_elem = element("Lagrange", domain.basix_cell(), 1)
    W = fem.functionspace(domain, mixed_element([v_elem, p_elem]))

    # --- Dirichlet Kinematic Boundary Conditions ---
    # Top Boundary (Die Surface, Tag 2): Prescribed downward velocity u_y = -V_press, u_x = 0
    # (Sub-space BCs require a Function on the collapsed velocity space V)
    V, _ = W.sub(0).collapse()
    fdim = domain.topology.dim - 1
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facet_tags.find(2))
    v_top = fem.Function(V)
    v_top.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), np.full(x.shape[1], -V_press))))
    bc_top = fem.dirichletbc(v_top, top_dofs, W.sub(0))

    # Bottom Boundary (Substrate Interface, Tag 1): Rigid No-Slip Condition u = (0, 0)
    bot_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facet_tags.find(1))
    v_bot = fem.Function(V)  # zero-initialized
    bc_bot = fem.dirichletbc(v_bot, bot_dofs, W.sub(0))

    stokes_bcs = [bc_top, bc_bot]

    # --- Variational (Weak) Formulation of Stokes System ---
    # Test Functions: v ∈ V, q ∈ Q; Trial Functions: u ∈ V, p ∈ Q
    # Weak Form Integration by Parts:
    #   a(u,v) = ∫ 2μ ε(u):ε(v) dx - ∫ p (∇·v) dx + ∫ q (∇·u) dx = 0
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f_zero = fem.Constant(domain, (0.0, 0.0))

    # Cure degree field α (P1), shared by the rheology and kinetics models.
    V_scalar = fem.functionspace(domain, ("Lagrange", 1))
    alpha_n = fem.Function(V_scalar, name="Cure_Degree")
    alpha_n.interpolate(lambda x: np.zeros(x.shape[1]))     # Uncured State (α = 0)

    # Cure-dependent viscosity μ(α) = μ_0 * exp(C_μ * α), evaluated from the live α field.
    # LinearProblem re-assembles on every solve(), so updating alpha_n updates μ.
    mu_eff = mu_0 * ufl.exp(C_mu * alpha_n)

    a_stokes = (
        2.0 * mu_eff * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx  # Viscous dissipation work
        - p * ufl.div(v) * ufl.dx                                                      # Hydrodynamic pressure work
        + q * ufl.div(u) * ufl.dx                                                      # Mass continuity constraint
    )
    L_stokes = ufl.inner(f_zero, v) * ufl.dx

    # Set up Linear PETSc Solver using Direct LU Decomposition (MUMPS backend)
    stokes_problem = LinearProblem(
        a_stokes, L_stokes, bcs=stokes_bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_",
    )

    def solve_stokes():
        w_sol = stokes_problem.solve()
        p_sol = w_sol.sub(1).collapse()
        p_sol.name = "Pressure"
        return w_sol.sub(0).collapse(), p_sol

    # ==========================================================================
    # 3. TRANSIENT HEAT TRANSPORT & ARRHENIUS CURE KINETICS
    #
    # Governing Energy PDE (Strong Form):
    #   ρ C_p (∂T/∂t) - ∇ · (k_th ∇T) = Q_reaction
    #   where Reaction Source Q_reaction = ρ * ΔH * (dα/dt)
    #
    # Reaction Kinetics ODE (Arrhenius nth-Order Model):
    #   dα/dt = S_α(T, α) = A * exp( -E_a / (R*T) ) * (1 - α)^n
    #   where α ∈ [0, 1] is the Degree of Cure (0 = Liquid, 1 = Fully Solidified)
    # ==========================================================================

    # Field Variable Initializations (t = 0); α was initialized with the Stokes setup above.
    T_n = fem.Function(V_scalar, name="Temperature")
    T_n.interpolate(lambda x: np.full(x.shape[1], 298.15))  # Room Temperature 25°C (298.15 K)

    # Thermal Boundary Condition: Bottom Substrate Isothermal Stage Heating (Tag 1)
    bot_thermal_dofs = fem.locate_dofs_topological(V_scalar, domain.topology.dim - 1, facet_tags.find(1))
    bc_heat = fem.dirichletbc(PETSc.ScalarType(T_hot), bot_thermal_dofs, V_scalar)

    # Time Discretization Parameters:
    dt = 0.1       # Time step Δt [s]
    t_total = 2.0  # Total process simulation duration [s]

    T_trial = ufl.TrialFunction(V_scalar)
    v_T = ufl.TestFunction(V_scalar)

    # --- Symbolic UFL Formulation for Reaction Source Term Q_reaction ---
    T_k = ufl.max_value(T_n, 273.15)                           # Ensure absolute temperature in Kelvin
    alpha_clamped = ufl.min_value(ufl.max_value(alpha_n, 0.0), 0.99)  # Numerical bound constraint α ∈ [0, 0.99)
    S_a_n = A_rate * ufl.exp(-E_a / (R_gas * T_k)) * ((1.0 - alpha_clamped) ** n_rxn)  # dα/dt
    Q_src = rho * Delta_H * S_a_n                          # Volumetric heat rate [W/mm³]

    # --- Weak Form of Transient Energy Equation (Implicit Backward Euler) ---
    #   ∫ ρ C_p [(Tⁿ⁺¹ - Tⁿ) / Δt] v_T dx + ∫ k_th ∇Tⁿ⁺¹ · ∇v_T dx = ∫ Q_src v_T dx
    F_thermal = (
        rho * Cp * ((T_trial - T_n) / dt) * v_T * ufl.dx                  # Unsteady thermal inertia
        + k_th * ufl.inner(ufl.grad(T_trial), ufl.grad(v_T)) * ufl.dx     # Fourier thermal conduction
        - Q_src * v_T * ufl.dx                                             # Exothermic cure heat release
    )
    a_thermal, L_thermal = ufl.lhs(F_thermal), ufl.rhs(F_thermal)
    thermal_problem = LinearProblem(a_thermal, L_thermal, bcs=[bc_heat], petsc_options_prefix="thermal_")

    # ==========================================================================
    # 4. TIME-STEPPING SOLVER LOOP & RHEOLOGY UPDATES
    #
    # Dynamic Viscosity Constitutive Model:
    #   μ(α) = μ_0 * exp( C_μ * α )
    #   Physical Rationale: As molecular crosslinking progresses (α → 1),
    #   fluid resistance grows exponentially until the adhesive solidifies.
    # ==========================================================================
    xdmf_file = XDMFFile(MPI.COMM_WORLD, "die_bonding_results.xdmf", "w")
    xdmf_file.write_mesh(domain)

    t = 0.0
    log("[Solver] Starting transient coupled physics iteration...")
    while t < t_total:
        t += dt

        # --- Step A: Solve Linearized Thermal Equation for Tⁿ⁺¹ ---
        T_h = thermal_problem.solve()

        # --- Step B: Explicit Forward Euler Update for Cure Degree αⁿ⁺¹ ---
        #   αⁿ⁺¹ = αⁿ + Δt * S_α(Tⁿ⁺¹, αⁿ)
        #   Hard bounds applied via np.clip to enforce physical domain α ∈ [0.0, 1.0]
        alpha_n.x.array[:] = np.clip(
            alpha_n.x.array + dt * (A_rate * np.exp(-E_a / (R_gas * T_h.x.array)) * ((1.0 - alpha_n.x.array) ** n_rxn)),
            0.0, 1.0
        )
        T_n.x.array[:] = T_h.x.array

        # --- Step C: Re-solve Stokes Squeeze Flow with Updated Viscosity μ(αⁿ⁺¹) ---
        u_sol, p_sol = solve_stokes()

        # Write Timestep Data for ParaView 3D Visualization
        xdmf_file.write_function(T_n, t)
        xdmf_file.write_function(alpha_n, t)
        xdmf_file.write_function(p_sol, t)

        # Output Field Monitors (Mean Temperature, Mean Cure Progress, Max Hydrodynamic Pressure),
        # reduced across all MPI ranks
        mean_alpha = global_mean(alpha_n)
        mean_T = global_mean(T_n) - 273.15
        max_p = global_max(p_sol)

        log(f"Time: {t:.1f}s | Mean Temp: {mean_T:.1f}°C | Cure: {mean_alpha*100:.2f}% | Max Hydro Pressure: {max_p:.4e} MPa")

    xdmf_file.close()
    log("[Solver] Simulation complete. Results saved to 'die_bonding_results.xdmf'.")


if __name__ == "__main__":
    run_simulation()