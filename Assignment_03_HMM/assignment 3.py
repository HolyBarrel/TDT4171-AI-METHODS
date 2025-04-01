import numpy as np
import matplotlib.pyplot as plt

# Transition matrix
T = np.array([
    [0.7, 0.3],  # From Rain -> [Rain, Not Rain]
    [0.3, 0.7]   # From Not Rain -> [Rain, Not Rain]
])

# Forward operation to recursively calculate the forward message
def forward_operation(prev, evidence):
    O = create_obs(evidence)

    prediction = np.dot(O, np.dot(T, prev))

    # Normalizes the prediction during each iteration
    normalized = prediction / np.sum(prediction)

    return normalized

# Creates the observation matrix based on the observation provided
def create_obs(observation):
    if observation == "umbrella":
        # Utilizes the probabilities of observing an umbrella
        return np.diag([0.9, 0.2])
    elif observation == "no_umbrella":
        # Utilizes the probabilities of observing no umbrella
        return np.diag([0.1, 0.8])
    else:
        raise ValueError("Invalid observation provided. Use 'umbrella' or 'no_umbrella'.")


if __name__ == "__main__":
    # Initial forward message, meaning the probability of rain and no rain at day 0
    f0 = np.array([0.5, 0.5])

    # Observations for each day in the task
    observations = ["umbrella", "umbrella", "no_umbrella", "umbrella", "umbrella"]

    # Recording the forward messages for each day
    msgs= []

    forward_msg = f0
    for t, obs in enumerate(observations, start=1):
        forward_msg = forward_operation(forward_msg, obs)
        msgs.append(forward_msg)
        print(f"Forward message after observing {obs} at day {t}:", forward_msg)

    print(f"Probability of rain at day 2:",msgs[1][0])
    print(f"Probability of rain at day 5:",msgs[4][0])

    prob_rain = [m[0] for m in msgs]
    
    # Generates an array of day indices
    days = np.arange(1, len(prob_rain) + 1)
    
    # Plots the probability of rain
    plt.plot(days, prob_rain, marker='o', color='blue', label='Probability of Rain')
    plt.xlabel('Day')
    plt.ylabel('Probability')
    plt.title('Probability for Rain Over Days')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')

    # Annotates each point with its corresponding observation (for better visualization)
    for day, obs, p_rain in zip(days, observations, prob_rain):
        # Positions the text slightly above each marker
        plt.annotate(obs, xy=(day, p_rain), xytext=(0, 15), 
                     textcoords='offset points', ha='center')

    plt.show()