import matplotlib.pyplot as plt
import random

# Returns a random integer representing a day of a 365-day year
def random_birthday():
    return random.randint(1, 365)

# Computes and returns the probability for at least two birthdays occurring the same day from n_persons
def compute_prob_birthdays_collide(n_persons, n_simulations=100000):
    #Tracks how many simulations result in at least one collision
    collisions_count = 0

    for _ in range(n_simulations):
        used_birthdays = set()
        collision_found = False

        for _ in range(n_persons):
            birthday = random_birthday()
            if birthday in used_birthdays:
                collision_found = True
                break
            used_birthdays.add(birthday)

        if collision_found:
            collisions_count += 1

    return collisions_count / n_simulations

# Computes probabilities for a range of n_persons-params in an interval from start to end
def compute_prob_collide_for_range(start=10, end=50, n_simulations=100000):
    for n in range(start, end + 1):
        prob = compute_prob_birthdays_collide(n, n_simulations)
        print(f"For N = {n}, probability of collision = {prob:.4f}")

# Plots the probability various n_persons-params for the range specified
def plot_prob_collide_for_range(start=10, end=50, n_simulations=100000):
    # Utilizes a list to store N values and corresponding probabilities
    n_values = list(range(start, end + 1))
    probabilities = []

    # Iterates through the range and computes probabilities
    for n in n_values:
        prob = compute_prob_birthdays_collide(n, n_simulations)
        probabilities.append(prob)

    # Generates the plot
    plt.figure(figsize=(8, 6))
    plt.plot(n_values, probabilities, marker='o', linestyle='-', color='blue', label='Collision Probability')

    # Draws a horizontal line at 0.5
    plt.axhline(y=0.5, color='green', linestyle='--', label='50% Threshold')

    # Finds the first N for which probability >= 0.5 (if any)
    crossing_n = None
    crossing_prob = None
    for n, prob in zip(n_values, probabilities):
        if prob >= 0.5:
            crossing_n = n
            crossing_prob = prob
            break

    # If any crossing point exists, then it is plotted on the graph
    if crossing_n is not None:
        # Plot a red dot at the crossing point
        plt.scatter(crossing_n, crossing_prob, color='red', zorder=5)
        # Annotate the crossing point
        plt.annotate(
            f'N={crossing_n}',
            (crossing_n, crossing_prob),
            xytext=(crossing_n + 1, crossing_prob),
            arrowprops=dict(arrowstyle='->', color='green'),
            color='red'
        )

    # Adds title and axis labels
    plt.title(f'Probability of Colliding Birthdays (N from {start} to {end})')
    plt.xlabel('Number of People (N)')
    plt.ylabel('Probability of at least one shared birthday')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    print("Probability of any of 2 people having the same birthday:")
    print(compute_prob_birthdays_collide(2))
    print("Amount of simulations: 100000")

    # Calls the new plotting function for N in [10..50]
    plot_prob_collide_for_range(10, 50, 100000)


if __name__ == "__main__":
    main()

