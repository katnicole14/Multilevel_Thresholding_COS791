import numpy as np


def shade(
    dims,
    objective_fn,
    pop_size=30,
    bounds=(1, 254),
    max_generations=100,
    H=100,          # memory size
    p=0.1,          # top p% used for pbest (e.g. 0.1 = top 10%)
    maximize=True,
    seed=None,
):
    rng = np.random.default_rng(seed)
    lo, hi = bounds
 
    # --- init population ---
    population = rng.uniform(lo, hi, size=(pop_size, dims))
    fitness = np.array([objective_fn(ind) for ind in population])
 
    # --- init memory: start at 0.5 for both F and CR (neutral guess) ---
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    memory_pos = 0  # circular pointer into the memory arrays
 
    def is_better(a, b):
        return a > b if maximize else a < b
 
    best_idx = np.argmax(fitness) if maximize else np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    history = [best_fitness]
 
    for gen in range(max_generations):
        S_F, S_CR, S_weights = [], [], []  # successful (F,CR) pairs this gen
 
        new_population = population.copy()
        new_fitness = fitness.copy()
 
        # pbest pool: top ceil(p * pop_size) individuals of CURRENT pop
        n_pbest = max(1, int(round(p * pop_size)))
        pbest_pool = np.argsort(fitness)
        pbest_pool = pbest_pool[-n_pbest:] if maximize else pbest_pool[:n_pbest]
 
        for i in range(pop_size):
            # --- sample this individual's own F_i, CR_i from memory ---
            r = rng.integers(0, H)
            F_i = -1
            while F_i <= 0:  # Cauchy can produce negative/zero -> resample
                F_i = rng.standard_cauchy() * 0.1 + M_F[r]
            F_i = min(F_i, 1.0)
 
            CR_i = rng.normal(M_CR[r], 0.1)
            CR_i = np.clip(CR_i, 0.0, 1.0)
 
            # --- current-to-pbest/1 mutation ---
            pbest_idx = pbest_pool[rng.integers(0, len(pbest_pool))]
            others = [idx for idx in range(pop_size) if idx != i]
            r1, r2 = rng.choice(others, size=2, replace=False)
 
            mutant = (
                population[i]
                + F_i * (population[pbest_idx] - population[i])
                + F_i * (population[r1] - population[r2])
            )
            mutant = np.clip(mutant, lo, hi)
 
            # --- binomial crossover ---
            trial = population[i].copy()
            j_rand = rng.integers(0, dims)
            for j in range(dims):
                if rng.random() < CR_i or j == j_rand:
                    trial[j] = mutant[j]
 
            trial_fitness = objective_fn(trial)
 
            # --- greedy selection ---
            if is_better(trial_fitness, fitness[i]):
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                S_F.append(F_i)
                S_CR.append(CR_i)
                S_weights.append(abs(trial_fitness - fitness[i]))  # improvement size
 
        population, fitness = new_population, new_fitness
 
        # --- update memory (only if at least one success this generation) ---
        if S_F:
            S_F = np.array(S_F)
            S_CR = np.array(S_CR)
            w = np.array(S_weights)
            w = w / w.sum()  # normalize weights
 
            # weighted Lehmer mean for F (standard SHADE formula)
            mean_F = np.sum(w * S_F ** 2) / np.sum(w * S_F)
            # weighted arithmetic mean for CR
            mean_CR = np.sum(w * S_CR)
 
            M_F[memory_pos] = mean_F
            M_CR[memory_pos] = mean_CR
            memory_pos = (memory_pos + 1) % H
 
        gen_best_idx = np.argmax(fitness) if maximize else np.argmin(fitness)
        if is_better(fitness[gen_best_idx], best_fitness):
            best_solution = population[gen_best_idx].copy()
            best_fitness = fitness[gen_best_idx]
 
        history.append(best_fitness)
 
    return best_solution, best_fitness, history


def l_shade(
    dims,
    objective_fn,
    max_nfes,           # <-- budget in FUNCTION EVALUATIONS, not generations
    pop_size_init=None, # NP_init; defaults to 18*dims (common L-SHADE default)
    pop_size_min=4,      # NP_min; DE/current-to-pbest/1 needs at least 4
    bounds=(1, 254),
    H=6,                 # L-SHADE paper commonly uses a SMALLER memory than SHADE
    p=0.11,
    arch_rate=2.6,        # archive capacity = round(arch_rate * NP)
    maximize=True,
    seed=None,
):
    """
    L-SHADE: SHADE + Linear Population Size Reduction (LPSR) + external
    archive, driven by a FUNCTION-EVALUATION budget (not a fixed number
    of generations) -- this matches the assignment's stopping criterion
    ("Set maximum Function Evaluations (FEs) to be equal") and is
    necessary anyway, since population size shrinks over the run, so
    "one generation" no longer means a fixed number of evaluations.
 
    Returns
    -------
    best_solution, best_fitness, history
    history : list of (nfes_used, best_fitness_so_far) checkpoints,
              one per generation, so convergence plots use a common
              x-axis (NFES) across all 5 algorithms.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
 
    NP_init = pop_size_init if pop_size_init is not None else max(4, 18 * dims)
    NP = NP_init
 
    population = rng.uniform(lo, hi, size=(NP, dims))
    fitness = np.array([objective_fn(ind) for ind in population])
    nfes = NP  # initial population evaluation counts too
 
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    memory_pos = 0
 
    archive = np.empty((0, dims))
    arch_size_max = int(round(arch_rate * NP_init))
 
    def is_better(a, b):
        return a > b if maximize else a < b
 
    best_idx = np.argmax(fitness) if maximize else np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    history = [(nfes, best_fitness)]
 
    while nfes < max_nfes:
        S_F, S_CR, S_weights = [], [], []
        new_population = population.copy()
        new_fitness = fitness.copy()
 
        n_pbest = max(1, int(round(p * NP)))
        pbest_pool = np.argsort(fitness)
        pbest_pool = pbest_pool[-n_pbest:] if maximize else pbest_pool[:n_pbest]
 
        for i in range(NP):
            r = rng.integers(0, H)
            F_i = -1
            while F_i <= 0:
                F_i = rng.standard_cauchy() * 0.1 + M_F[r]
            F_i = min(F_i, 1.0)
 
            CR_i = np.clip(rng.normal(M_CR[r], 0.1), 0.0, 1.0)
 
            pbest_idx = pbest_pool[rng.integers(0, len(pbest_pool))]
 
            # r1 from population; r2 from population UNION archive
            others = [idx for idx in range(NP) if idx != i]
            r1 = rng.choice(others)
 
            union_pool = np.vstack([population, archive]) if len(archive) else population
            r2_idx = rng.integers(0, len(union_pool))
            x_r2 = union_pool[r2_idx]
 
            mutant = (
                population[i]
                + F_i * (population[pbest_idx] - population[i])
                + F_i * (population[r1] - x_r2)
            )
            mutant = np.clip(mutant, lo, hi)
 
            trial = population[i].copy()
            j_rand = rng.integers(0, dims)
            for j in range(dims):
                if rng.random() < CR_i or j == j_rand:
                    trial[j] = mutant[j]
 
            trial_fitness = objective_fn(trial)
            nfes += 1
 
            if is_better(trial_fitness, fitness[i]):
                # archive the replaced (losing) target before overwriting
                archive = np.vstack([archive, population[i]]) if len(archive) else population[i:i+1].copy()
                if len(archive) > arch_size_max:
                    drop = rng.integers(0, len(archive))
                    archive = np.delete(archive, drop, axis=0)
 
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                S_F.append(F_i)
                S_CR.append(CR_i)
                S_weights.append(abs(trial_fitness - fitness[i]))
 
            if nfes >= max_nfes:
                break
 
        population, fitness = new_population, new_fitness
 
        if S_F:
            S_F = np.array(S_F)
            S_CR = np.array(S_CR)
            w = np.array(S_weights)
            w = w / w.sum()
            mean_F = np.sum(w * S_F ** 2) / np.sum(w * S_F)
            mean_CR = np.sum(w * S_CR)
            M_F[memory_pos] = mean_F
            M_CR[memory_pos] = mean_CR
            memory_pos = (memory_pos + 1) % H
 
        # --- Linear Population Size Reduction (LPSR) ---
        NP_new = int(round(((pop_size_min - NP_init) / max_nfes) * nfes + NP_init))
        NP_new = max(pop_size_min, NP_new)
        if NP_new < NP:
            # drop the worst (NP - NP_new) individuals
            order = np.argsort(fitness)
            keep = order[-NP_new:] if maximize else order[:NP_new]
            population = population[keep]
            fitness = fitness[keep]
            NP = NP_new
            arch_size_max = int(round(arch_rate * NP))
            if len(archive) > arch_size_max:
                archive = archive[rng.choice(len(archive), arch_size_max, replace=False)]
 
        gen_best_idx = np.argmax(fitness) if maximize else np.argmin(fitness)
        if is_better(fitness[gen_best_idx], best_fitness):
            best_solution = population[gen_best_idx].copy()
            best_fitness = fitness[gen_best_idx]
 
        history.append((nfes, best_fitness))
 
    return best_solution, best_fitness, history

