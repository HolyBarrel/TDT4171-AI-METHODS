import random

# Returs a random birthday represented by an integer from 1 to 365 for each day of a year
def random_birthday():
    return random.randint(1, 365)

# Runs a simulation to find out how many persons are needed in Peters group the set to cover every day of the year
def run_simulate_group(n_simulations=10000):
    total_group_size = 0

    for _ in range(n_simulations):
        # Utilizes a set to track distinct birthdays
        covered_days = set()
        group_size = 0

        # Adds a random person untill all days are covered by the group
        while len(covered_days) < 365:
            covered_days.add(random_birthday())
            group_size += 1

        total_group_size += group_size

    # Computes the average group size across all simulations
    return total_group_size / n_simulations


def main():
    n_simulations = 100000
    expected_group_size = run_simulate_group(n_simulations)
    print(f"{n_simulations} simulations: expected group size, about {expected_group_size:.2f} people")


if __name__ == "__main__":
    main()
