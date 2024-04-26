from scipy.stats import norm
from csv import writer
import numpy as np

def generate_points_A(num_points:int=2000):
    distribution_x = norm(loc=0, scale=50)
    distribution_y = norm(loc=0, scale=200)
    distribution_z = norm(loc=0, scale=0)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x,y,z) #tworzy listę krotek x,y,z
    return points

def generate_points_B(num_points:int=2000):
    distribution_x = norm(loc=0, scale=0)
    distribution_y = norm(loc=0, scale=50)
    distribution_z = norm(loc=0, scale=200)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x,y,z) #tworzy listę krotek x,y,z
    return points

def generate_points_C(num_points:int=2000, r:int=10):
    # Generowanie losowych wysokości
    distribution_z = norm(loc=0, scale=200)
    z = distribution_z.rvs(size=num_points)

    # Generowanie losowych kątów
    distribution_theta = norm(loc=0, scale=360)
    theta = distribution_theta.rvs(size=num_points)

    # Konwersja do współrzędnych kartezjańskich
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    points = zip(x,y,z) #tworzy listę krotek x,y,z
    return points


cloud_points_A = generate_points_A(2000)

cloud_points_B= generate_points_B(2000)

cloud_points_C = generate_points_C(2000,75)

with open('points.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
    csvwriter = writer(csvfile)
    # csvwriter.writerow('x','y','z') nagłówek
    for p in cloud_points_A:
         csvwriter.writerow(p)

    for p in cloud_points_B:
        csvwriter.writerow(p)

    for p in cloud_points_C:
        csvwriter.writerow(p)