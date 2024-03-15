# Assuming num_episodes, random_a, and dqn_a are defined and updated within the loop
import torch

# num = 1
# # Example list of tensors, each with a single element
#
#
# # Convert each tensor to a number and format it
# formatted_numbers = ', '.join(f"{tensor.item():.0f}%" for tensor in random_a)
#
# text1 = f"The Model{num:.0f}k3 random is {formatted_numbers}"
# print(text1)



number_of_iterations = 5
# Open the file before the loop starts. Use 'a' mode to append if you want to keep adding to the file across multiple runs.
with open('model_accuracies.txt', 'w') as file:
    for i in range(number_of_iterations):  # Replace number_of_iterations with your actual loop range
        # Update num_episodes, random_a, and dqn_a as needed
        num_episodes = (i + 1) * 1000  # Example update logic
        # random_a = 75.5  # Example value, assume this changes every iteration
        dqn_a = 85.0  # Example value, assume this changes every iteration
        random_a = [torch.tensor([20.5]), torch.tensor([33.7]), torch.tensor([42.1])]
        # Construct the text you want to save
        formatted_numbers = ', '.join(f"{tensor.item():.0f}%" for tensor in random_a)
        text1 = f"The Model{num_episodes:.0f}k3 random is {formatted_numbers}"

        text2 = f"The model {num_episodes:.0f}k3 dqn accuracy is {dqn_a:.0f}%"

        # Write the text to the file
        file.write(text1 + "\n" + text2 + "\n")