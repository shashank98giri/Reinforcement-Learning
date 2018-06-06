from matplotlib import pyplot as plt

def plot_values(s,title):
    plt.figure(figsize=(7,4))
    plt.plot(s)
    plt.ylabel("delta")
    plt.xlabel("iterations")
    plt.title(title)

    plt.show()
