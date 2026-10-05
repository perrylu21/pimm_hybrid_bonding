#!/usr/bin/env python3
"""
03_main.py
Master pipeline launcher.
Automatically checks for existing .msh files, skips mesh generation if present,
and triggers the DOLFINx simulation solver.
"""

import os
import sys
import shutil
import importlib.util

DOLFINX_IMAGE = "dolfinx/dolfinx:stable"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def relaunch_in_docker_if_needed():
    """DOLFINx/petsc4py/mpi4py live only in the Docker image. If they are missing
    on this host, re-run this script inside the container with the project mounted."""
    if importlib.util.find_spec("dolfinx") is not None:
        return
    if shutil.which("docker") is None:
        sys.exit(f"[Error] DOLFINx not installed and 'docker' not found. Install Docker and pull {DOLFINX_IMAGE}.")

    cmd = ["docker", "run", "--rm"]
    if sys.stdin.isatty():
        cmd.append("-it")
    cmd += ["-v", f"{PROJECT_DIR}:/work", "-w", "/work", DOLFINX_IMAGE,
            "python3", os.path.basename(__file__), *sys.argv[1:]]
    print(f"[Pipeline Status] DOLFINx not found on host. Relaunching in Docker ({DOLFINX_IMAGE})...", flush=True)
    os.execvp("docker", cmd)  # replaces this process; exit code comes from the container


def check_and_import(module_name, file_path):
    """Helper to dynamically import module files."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    mesh_filename = "die_bonding_mesh.msh"
    mesh_script = "01_generate_mesh.py"
    solver_script = "02_solve_simulation.py"

    print("==========================================================")
    print("      Die Bonding Multiphysics Pipeline Launcher          ")
    print("==========================================================")

    # Step 1: Check Mesh File Existence
    if os.path.exists(mesh_filename):
        print(f"[Pipeline Status] Found existing mesh '{mesh_filename}'. Skipping mesh generation to save time.")
    else:
        print(f"[Pipeline Status] Mesh file '{mesh_filename}' not found. Triggering mesh generator...")
        if not os.path.exists(mesh_script):
            print(f"[Error] Required generator script '{mesh_script}' is missing!", file=sys.stderr)
            sys.exit(1)
        
        # Under mpirun, only rank 0 writes the mesh; other ranks wait for it.
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.rank == 0:
            mesh_module = check_and_import("generate_mesh_module", mesh_script)
            mesh_module.generate_mesh(mesh_filename)
        comm.Barrier()

    # Step 2: Execute DOLFINx Simulation
    print("\n[Pipeline Status] Launching DOLFINx Multiphysics Solver...")
    if not os.path.exists(solver_script):
        print(f"[Error] Required solver script '{solver_script}' is missing!", file=sys.stderr)
        sys.exit(1)

    solver_module = check_and_import("solve_simulation_module", solver_script)
    solver_module.run_simulation(mesh_filename)

    print("\n==========================================================")
    print("Pipeline finished successfully!")
    print("==========================================================")


if __name__ == "__main__":
    os.chdir(PROJECT_DIR)  # mesh/solver/result paths are relative to the project folder
    relaunch_in_docker_if_needed()
    main()