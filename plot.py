from matplotlib import pyplot as plt

def plot_learning_curve(training_res, validation_res, metric, title, filename):
    fig, ax = plt.subplots()

    x = range(len(training_res))

    ax.plot(x, training_res, label="Training " + metric)
    ax.plot(x, validation_res, label="Validation " + metric)

    ax.legend()
    ax.set(xlabel="Epoch", ylabel=metric)
    ax.set_title(title)

    fig.savefig(filename)
    plt.show()
    plt.close()

def plot_learning_curves(performance):
    for hyper_set, metrics_dict in performance.items():
        for metric, train_val_dict in metrics_dict.items():
            title = f"lr={hyper_set[0]} #neurons/hidden layer={hyper_set[1]}"
            filename = f"{hyper_set[0]}_{hyper_set[1]}_{metric}.png"
            plot_learning_curve(train_val_dict["training"], train_val_dict["validation"], metric, title, filename)