import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import norm, t
from scipy.stats import poisson
from scipy.stats import uniform
from scipy.stats import cauchy
import pandas
import numpy as np
from IPython.display import display

sns.set(style="whitegrid")

def generate_and_plot_normal_combined(sample_size, title, number):
    data = np.random.normal(loc=0, scale=1, size=sample_size)
    min_value_floor = np.floor(np.min(data))
    max_value_ceil = np.ceil(np.max(data))
    pdf = norm.pdf(np.linspace(min_value_floor, max_value_ceil, 1000), loc=0, scale=1)
    plt.subplot(1, 3, number)
    sns.histplot(data, kde=False, bins=int(math.log2(sample_size)) + 1, label='Гистограмма', stat='density', color='skyblue')
    plt.title(f'Выборка размером {sample_size}')
    plt.plot(np.linspace(min_value_floor, max_value_ceil, 1000), pdf, label='PDF', linestyle='dashed', color='orange')
    plt.suptitle(title, fontsize=16)

def generate_and_plot_student_combined(sample_size, title, number):
    data = np.random.standard_t(df=3, size=sample_size)
    min_value_floor = np.floor(np.min(data))
    max_value_ceil = np.ceil(np.max(data))
    pdf = t.pdf(np.linspace(min_value_floor, max_value_ceil, 1000), df=3)
    plt.subplot(1, 3, number)
    sns.histplot(data, kde=False, bins=int(math.log2(sample_size)) + 1, label='Гистограмма', stat='density', color='skyblue')
    plt.title(f'Выборка размером {sample_size}')
    plt.plot(np.linspace(min_value_floor, max_value_ceil, 1000), pdf, label='PDF', linestyle='dashed', color='orange')
    plt.suptitle(title, fontsize=16)

def generate_and_plot_poisson_combined(sample_size, title, number):
    data = np.random.poisson(lam=10, size=sample_size)
    min_value_floor = np.floor(np.min(data))
    max_value_ceil = np.ceil(np.max(data))
    pdf = poisson.pmf(np.arange(min_value_floor, max_value_ceil), mu=10)
    plt.subplot(1, 3, number)
    sns.histplot(data, kde=False, bins=int(math.log2(sample_size)) + 1, label='Гистограмма', stat='density', color='skyblue')
    plt.title(f'Выборка размером {sample_size}')
    plt.plot(np.arange(min_value_floor, max_value_ceil), pdf, label='PMF', linestyle='dashed', color='orange', marker='o')
    plt.suptitle(title, fontsize=16)

def generate_and_plot_uniform_combined(sample_size, title, number):
    data = np.random.uniform(-math.sqrt(3), math.sqrt(3), size=sample_size)
    pdf = uniform.pdf(np.linspace(-math.sqrt(3), math.sqrt(3), 1000), loc=-math.sqrt(3), scale=2*math.sqrt(3))
    plt.subplot(1, 3, number)
    sns.histplot(data, kde=False, bins=int(math.log2(sample_size)) + 1, label='Гистограмма', stat='density', color='skyblue')
    plt.title(f'Выборка размером {sample_size}')
    plt.plot(np.linspace(-math.sqrt(3), math.sqrt(3), 1000), pdf, label='PDF', linestyle='dashed', color='orange')
    plt.suptitle(title, fontsize=16)

def generate_and_plot_cauchy_combined(sample_size, title, number):
    data = np.random.standard_cauchy(size=sample_size)
    min_value_floor = np.floor(np.min(data))
    max_value_ceil = np.ceil(np.max(data))
    pdf = cauchy.pdf(np.linspace(min_value_floor, max_value_ceil, 1000), loc=0, scale=1)
    plt.subplot(1, 3, number)
    sns.histplot(data, kde=False, bins=int(np.sqrt(sample_size)), label='Гистограмма', stat='density', color='skyblue')
    plt.title(f'Выборка размером {sample_size}')
    plt.plot(np.linspace(min_value_floor, max_value_ceil, 1000), pdf, label='PDF', linestyle='dashed', color='orange')
    plt.suptitle(title, fontsize=16)

def print_characteristics(N: list) -> None:
    methods = [
        lambda n: np.random.normal(0.0, 1.0, n),
        lambda n: np.random.standard_cauchy(n),
        lambda n: np.random.standard_t(3.0, n),
        lambda n: np.random.poisson(10.0, n),
        lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
    ]

    names = ['normal', 'cauchy', "student's", 'poisson', 'uniform']
    repeats = 1000

    for i in range(len(methods)):
        for n in N:
            data = np.zeros([2, 5])
            for j in range(repeats):
                sample = methods[i](n)

                sample.sort()
                x = np.mean(sample)
                med_x = np.median(sample)
                z_r = (sample[0] + sample[-1]) / 2.0
                z_q = (sample[int(np.ceil(n / 4.0) - 1)] + sample[int(np.ceil(3.0 * n / 4.0) - 1)]) / 2.0
                r = int(np.round(n / 4.0))
                z_tr = (1.0 / (n - 2 * r)) * sum([sample[i] for i in range(r, n - r)])

                stats = [x, med_x, z_r, z_q, z_tr]
                for k in range(len(stats)):
                    data[0][k] += stats[k]
                    data[1][k] += stats[k] * stats[k]

            data /= repeats
            data[1] -= data[0] ** 2
            df = pandas.DataFrame(data, columns=["x", "med x", "z_R", "z_Q", "z_{tr}"], index=["E(z)", "D(z)"])
            print(f"{names[i]} n = {n}")
            display(df)

sample_sizes = [10, 50, 1000]
print_characteristics(sample_sizes)
'''
number = 1
plt.figure(figsize=(15, 5))
for size in sample_sizes:
    generate_and_plot_normal_combined(size, 'Нормальное распределение', number)
    number += 1
plt.show()

number = 1
plt.figure(figsize=(15, 5))
for size in sample_sizes:
    generate_and_plot_student_combined(size, 'Распределение Стьюдента', number)
    number += 1
plt.show()

number = 1
plt.figure(figsize=(15, 5))
for size in sample_sizes:
    generate_and_plot_poisson_combined(size, 'Распределение Пуассона (10)', number)
    number += 1
plt.show()
number = 1
plt.figure(figsize=(15, 5))
for size in sample_sizes:
    generate_and_plot_uniform_combined(size, 'Равномерное распределение (-√3, √3)', number)
    number += 1
plt.show()
number = 1
plt.figure(figsize=(15, 5))
for size in sample_sizes:
    generate_and_plot_cauchy_combined(size, 'Распределение Коши (0, 1)', number)
    number += 1
plt.show()
'''
