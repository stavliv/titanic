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