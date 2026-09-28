from Objective_functions.Otsu import repair_thresholds


def crossover(
    current_solution,
    mutant,
    crossover_rate,
    random_generator,
    levels=256
):
    """
    Perform binomial crossover for Differential Evolution.

    The current solution and mutant are combined to produce
    a new trial threshold solution.
    """

    # Crossover cannot work with an empty current solution.
    if len(current_solution) == 0:
        raise ValueError(
            "Current solution array is empty."
        )

    # Crossover cannot work with an empty mutant.
    if len(mutant) == 0:
        raise ValueError(
            "Mutant array is empty."
        )

    # Both vectors must contain the same number of thresholds.
    if len(current_solution) != len(mutant):
        raise ValueError(
            "Current solution and mutant arrays must have "
            "the same number of thresholds."
        )

    # CR is a probability, so it must be between 0 and 1.
    if crossover_rate < 0 or crossover_rate > 1:
        raise ValueError(
            "Crossover rate must be between 0 and 1."
        )

    # The vector length corresponds to K in the assignment.
    number_of_thresholds = len(current_solution)

    # Select one position that must use the mutant value.
    # This is called j_rand in the DE crossover formula.
    compulsory_mutant_position = (
        random_generator.integers(
            0,
            number_of_thresholds
        )
    )

    # Store the crossover result.
    trial = []

    # Process every threshold position separately.
    for position in range(number_of_thresholds):

        # Generate a value between 0 and 1.
        random_value = random_generator.random()

        # Use the mutant threshold if the crossover test passes
        # or if this is the compulsory mutant position.
        use_mutant = (
            random_value <= crossover_rate
            or position == compulsory_mutant_position
        )

        if use_mutant:
            trial.append(
                mutant[position]
            )
        else:
            trial.append(
                current_solution[position]
            )

    # Convert the result into valid image thresholds.
    trial = repair_thresholds(
        trial,
        levels
    )

    return trial