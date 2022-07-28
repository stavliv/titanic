from matplotlib import pyplot as plt

def plot_learning_curve(training_res: list, validation_res: list, metric: str, title: str, filename: str):
    '''
    plots the learning curve

    Parameters
    ----------
    training_res : array_like (1d)
        the training points to plot
    validation_res : array_like (1d)
        the validation points to plot
    metric : str
        the metric that is being plotted 
    title : str
        the title of the plot
    filename : str
        the file to save the plot
    '''
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