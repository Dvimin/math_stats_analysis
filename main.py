import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import norm, t
from scipy.stats import poisson
from scipy.stats import uniform
from scipy.stats import cauchy


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


sample_sizes = [10, 50, 1000]

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


