import typing
from typing import List

class Problem:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Subclass should implement this")
    
    def get_data(self):
        """Return problem data."""
        raise NotImplementedError("Subclass should implement this")
    
    def set_data(self, data):
        """Set problem data"""
        raise NotImplementedError("Subclass should implement this")
    
    def is_solution(self, sol) -> bool:
        """Check if sol is solution """
        raise NotImplementedError("Subclass should implement this")

        
class Solver:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Subclass should implement this")
    
    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def can_solve(self, problem: Problem) -> bool:
        """Return True if solver can solve this problem."""
        raise NotImplementedError("Subclass should implement this")
    
    def solve(self, problem: Problem, *args, **kwargs):
        raise NotImplementedError("Subclass should implement this")
    

class MicroProblem(Problem):
    def update(self, macro: Problem):
        """Update micro problem with macro data"""
        raise NotImplementedError("Subclass should implement this")

    
class MacroProblem(Problem): 
    def update(self, micros: List[MicroProblem]):
        """Update macro problem with micro data"""
        raise NotImplementedError("Subclass should implement this")
        
        
class HMMProblem(Problem):
    def __init__(self, macro: MacroProblem, micros: List[MicroProblem], *args, **kwargs):
        """Given a macro problem and a micro problem(s), construct a HMM solver."""
        self.macro = macro
        self.micros = micros
        self.convergence_checker = kwargs.get("convergence_checker", None)
        pass
        
    def update_macro(self, micro_sols):
        self.macro.update(micro_sols)
        pass
    
    def update_micros(self, macro_sol):
        for macro in self.micros:
            macro.update(macro_sol)
        pass
    
    def update(self, macro_sol, micro_sols):
        self.update_macro(micro_sols)
        self.update_micros(macro_sol)
        pass
    
    def get_data(self, *args, **kwargs):
        return (self.macro.get_data(*args, **kwargs),) + tuple(micro.get_data(*args, **kwargs) for micro in self.micros)
    
    
    def set_data(self, macro_data, micro_datas, *args, **kwargs):
        self.macro.set_data(macro_data, *args, **kwargs)
        for micro, data in zip(self.micros, micro_datas):
            micro.set_data(data, *args, **kwargs)
        pass
    
    def is_converged(self, macro_sol, micro_sols, *args, **kwargs):
        if self.convergence_checker is not None:
            return self.convergence_checker(macro_sol, micro_sols, *args, **kwargs)
        else:
            return False
    
    def is_solution(self, macro_sol, micro_sols, *args, **kwargs):
        """Check if solution"""
        self.update(macro_sol, micro_sols)
        return self.macro.is_solution(macro_sol, *args, **kwargs) and\
                all(micro.is_solution(sol, *args, **kwargs) for micro, sol in zip(self.micros, micro_sols))
    

class IterativeHMMSolver(Solver):
    def __init__(self, macro_solver: Solver, micro_solvers: List[Solver]):
        self.macro_solver = macro_solver
        self.micro_solvers = micro_solvers
        pass
        
    def can_solve(self, problem: Problem, *args, **kwargs)->bool:
        if HMMProblem.__subclasscheck__(type(problem)):
            if len(self.micro_solvers) == len(problem.micros):
                return self.macro_solver.can_solve(problem.macro) and \
                        all(solver.can_solve(micro) for solver, micro in zip(self.micro_solvers, problem.micros))
        else:
            return False
        

    def solve(self, hmm_problem: HMMProblem, macro_guess, *args, callback=None, verbose=False, logger=None, **kwargs):
        """Solve Multi Scale Problem"""
        assert self.can_solve(hmm_problem), "Unable to solve"
        maxiter = kwargs.pop("maxiter", 10)
        macro_sol = macro_guess
        for i in range(maxiter):
            if verbose:
                print(f"Step {i}/{maxiter}", end="\r")
            

            if logger is not None:
                logger.start_event("hmm_micro_update")
            
            # Solve micro problems
            hmm_problem.update_micros(macro_sol)
            
            if logger is not None:
                logger.end_event("hmm_micro_update")
                logger.start_event("hmm_micro_solve")
            
            micro_sols = [solver.solve(micro) for solver, micro in zip(self.micro_solvers, hmm_problem.micros)]
            

            # Solve macro problem
            if logger is not None:
                logger.end_event("hmm_micro_solve")
                logger.start_event("hmm_macro_update")

            hmm_problem.update_macro(micro_sols)

            if logger is not None:
                logger.end_event("hmm_macro_update")
                logger.start_event("hmm_macro_solve")

            macro_sol = self.macro_solver.solve(hmm_problem.macro)

            if logger is not None:
                logger.end_event("hmm_macro_solve")
                logger.start_event("hmm_solution_check")
            
            # Check if solved
            if hmm_problem.is_solution(macro_sol, micro_sols, *args, **kwargs):
                if verbose:
                    print(f"Convergence to solution at step {i}/{maxiter}")
                break
            
            # Check if converged
            if hmm_problem.is_converged(macro_sol, micro_sols, *args, **kwargs):
                if verbose:
                    print(f"Convergence (not necessarily to solution) at step {i}/{maxiter}")
                break
            
            if logger is not None:
                logger.end_event("hmm_solution_check")
            
            # Apply callback
            if callback is not None:
                callback(i, macro_sol, micro_sols)
                
        # Return solutions
        return macro_sol, micro_sols