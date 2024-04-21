from scipy.stats import norm
from csv import writer

distribution_x = norm(loc=0, scale=28)
distribution_y = norm(loc=0, scale=200)
distribution_z = norm(loc=0.2, scale=0.05)

num_points = 2000
x = distribution_x.rvs(size=num_points)
y = distribution_y.rvs(size=num_points)
z = distribution_z.rvs(size=num_points)

points = zip(x,y,z) #tworzy listę krotek x,y,z
with open('LidarData.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
    csvwriter = writer(csvfile)
    # csvwriter.writerow('x','y','z') nagłówek
    for p in points:
        csvwriter.writerow(p)
