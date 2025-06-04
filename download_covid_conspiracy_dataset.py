from datasets import load_dataset

# Download the dataset
dataset = load_dataset("webimmunization/COVID-19-conspiracy-theories-tweets")

# Save the train split to CSV
dataset["train"].to_csv("COVID-19-conspiracy-theories-tweets.csv", index=False)
print("Dataset downloaded and saved as COVID-19-conspiracy-theories-tweets.csv")