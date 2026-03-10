import matplotlib.pyplot as plt

def plot_feature_distribution(train_data, new_data, feature):

    plt.figure(figsize=(8,5))

    plt.hist(train_data[feature], bins=50, alpha=0.5, label="Training Data")
    plt.hist(new_data[feature], bins=50, alpha=0.5, label="Incoming Data")

    plt.title(f"Feature Distribution Comparison: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.legend()

    plt.savefig("drift_amount.png")
    plt.close()