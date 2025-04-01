import random
import statistics

# Describes the different possible symbols for each slot of the slot machine
symbols = ["BAR", "BELL", "LEMON", "CHERRY"]

# Creates a simple slot machine-array with all the possible combos for each slot
possible_combos = [
    (s1, s2, s3)
    for s1 in symbols
    for s2 in symbols
    for s3 in symbols
]

# Computes the payouts for the different slot machine combinations
def compute_payout(slot_combo):
    # One wheel for each of the symbols of the combo to compute
    w1, w2, w3 = slot_combo

    if w1 == w2 == w3:
        if w1 == "BAR": return 20
        if w1 == "BELL": return 15
        if w1 == "LEMON": return 5
        if w1 == "CHERRY": return 3
    elif w1 == w2:
        if w1 == "CHERRY": return 2
    elif w1 == "CHERRY": return 1

    return 0

# Returns a random spin from the array of possible combos
def get_random_spin():
    num = random.random()
    index = int(num * len(possible_combos))

    return possible_combos[index]

# Simulates spinning the machine until broke (having no coins left)
def spin_until_broke(coins_remaining):
    counter = 0
    while coins_remaining > 0:
        coins_remaining -= 1
        spin = get_random_spin()
        payout = compute_payout(spin)
        coins_remaining += payout

        counter += 1

    return counter

def main():
    n_runs = 100000
    results = []

    for _ in range(n_runs):
        # The default for each round is stated to be 10 coins in the task
        init_coins = 10
        spins_taken = spin_until_broke(coins_remaining=init_coins)
        results.append(spins_taken)

    # Computes and prints some basic statistics
    mean_spins = statistics.mean(results)
    median_spins = statistics.median(results)
    print(f"Initial coins: {init_coins}")
    print(f"Number of simulations: {n_runs}")
    print(f"Mean spins before broke: {mean_spins:.2f}")
    print(f"Median spins before broke: {median_spins:.0f}")

if __name__=="__main__":
    main()


