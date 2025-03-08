import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse


def solve_heat_equation(t : np.ndarray) -> (np.ndarray, np.array):
    """
    Solves the 2D heat equation over a given time period.

    Parameters:
    t (np.ndarray): A 3D numpy array representing the temperature grid at each time step.
                    The shape of the array is (max_iter_time, domain, domain).

    Returns:
    np.ndarray: The updated temperature grid after solving the heat equation.
    np.array: An array containing the error values at each time step.

    The function iterates over the time steps and updates the temperature grid based on the
    finite difference method. It also calculates the error between the solutions at consecutive
    time steps and stores these error values in an array.
    """
    errors = np.zeros(max_iter_time)
    for k in range(max_iter_time):
        for i in range(1, domain-1):
            for j in range(1, domain-1):
                t[k + 1, i, j] = gamma * (t[k, i + 1, j] + t[k, i - 1, j] + t[k, i, j + 1] + t[k, i, j - 1]) + (1 - 4 * gamma) * t[k, i, j]
        errors[k] = np.sqrt(np.sum(delta_t * (t[k + 1, 1:domain-1, 1:domain-1] - t[k, 1:domain-1, 1:domain-1]) ** 2))
    return t, errors

def plot_heat_map(t_k : float, k : int) -> plt:
    plt.clf()

    plt.title(f"Temperature at time t = {k*delta_t:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.pcolormesh(t_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

    return plt

def y_n_question(question : str) -> bool:
    while True:
        answer = input(question)
        if answer.lower() == "y":
            return True
        elif answer.lower() == "n":
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'")

def animate(k):
    plot_heat_map(T[k], k)

def get_float() -> float:
    while True:
        try:
            return float(input())
        except ValueError:
            print("Invalid input. Please enter a valid float.")

def get_initial_condition() -> (float, float, float, float, float):
    print("Enter initial base temperature: ")
    t_initial = get_float()
    print("Enter top temperature: ")
    t_top = get_float()
    print("Enter left temperature: ")
    t_left = get_float()
    print("Enter right temperature: ")
    t_right = get_float()
    print("Enter bottom temperature: ")
    t_bottom = get_float()

    return T_initial, T_top, T_left, T_right, T_bottom

def plot_errors(errors_array : np.ndarray) -> plt:
    plt.clf()
    plt.plot(np.arange(max_iter_time)*delta_t, errors_array)
    plt.title("Error over time")
    plt.xlabel("Time")
    plt.ylabel("Error")
    return plt


if __name__ == "__main__":
    print("2D heat equation solver")

    parser = argparse.ArgumentParser("2D heat equation solver")
    parser.add_argument("-d","--domain", type=int, default=100, help="Domain size")
    parser.add_argument("-m","--max_iter_time", type=int, default=750, help="Maximum iteration time")
    parser.add_argument("-a","--alpha", type=float, default=.1, help="Alpha value")
    parser.add_argument("-s","--mesh_size", type=int, default=1, help="Mesh size")
    args = parser.parse_args()

    domain = args.domain
    max_iter_time = args.max_iter_time

    alpha = args.alpha
    mesh_size = args.mesh_size

    print(f"Using Settings: domain={domain}, max_iter_time={max_iter_time}, alpha={alpha}, mesh_size={mesh_size}")

    delta_t = (mesh_size ** 2 / (4 * alpha))
    gamma = alpha * delta_t / (mesh_size ** 2)

    print(f"Delta t: {delta_t:.3f}, Gamma: {gamma:.3f}")

    # initialize solution : u(k, i, j) grid
    # k is time index
    # i, j are spatial indices
    T = np.zeros((max_iter_time, domain, domain))

    # set initial condition
    T_initial = 0

    T_top = 0
    T_left = 0
    T_right = 0
    T_bottom = 100

    if y_n_question("Do you want to set initial condition? (y/n)"):
        T_initial, T_top, T_left, T_right, T_bottom = get_initial_condition()

    # set boundary conditions
    T.fill(T_initial)

    T[:, (domain - 1):, :] = T_top
    T[:, :, :1] = T_left
    T[:, :, (domain - 1):] = T_right
    T[:, :1, :] = T_bottom

    print("Solving heat equation...")
    T, errors = solve_heat_equation(T)

    anim = FuncAnimation(plt.figure(), animate, interval=1, frames=max_iter_time, repeat=False)
    anim.save("output_animation/heat_equation.gif")

    error_plot = plot_errors(errors)
    error_plot.savefig("output_error/error_plot.png")

    print("Done")

