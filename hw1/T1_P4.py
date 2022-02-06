#####################
# CS 181, Spring 2021
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20
        
    if part == 'a':
        arr = []
        for x in xx:
            arr.append([1, x, x**2, x**3, x**4, x**5])
        return np.array(arr)
    
    elif part == 'b':
        arr = []
        for x in xx:
            basis = [1]
            for j in range(1960, 2015, 5):
                basis.append(np.exp(-1 * (x - j) ** 2 / 25))
            arr.append(basis)
        return np.array(arr)
    
    elif part == 'c':
        arr = []
        for x in xx:
            basis = [1]
            for j in range(1, 6):
                basis.append(np.cos(x/j))
            arr.append(basis)
        return np.array(arr)
    
    elif part == 'd':
        arr = []
        for x in xx:
            basis = [1]
            for j in range(1, 26):
                basis.append(np.cos(x/j))
            arr.append(basis)
        return np.array(arr)
    
    else:
        return None

# Create the simplest basis, with just the time and an offset.
# X = np.vstack((np.ones(years.shape), years)).T
# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

for part in ['a', 'b', 'c', 'd']:
    X = make_basis(years, part=part, is_years=True)
    w = find_weights(X, Y)

    # Compute the regression line on a grid of inputs.
    # DO NOT CHANGE grid_years!!!!!
    grid_years = np.linspace(1960, 2005, 200)
    grid_X = make_basis(grid_years, part=part, is_years=True)
    grid_Yhat = np.dot(grid_X, w)

    # TODO: plot and report sum of squared error for each basis
    print(f"L2 for part {part.upper()}: {sum((np.dot(X, w) - Y) ** 2)}")

    # Plot the data and the regression line.
    plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.title(f"Part {part.upper()} (Year)")
    plt.savefig('year_' + str(part) + '.png')
    plt.show()

Y = republican_counts[years<last_year]
for part in ['a', 'c', 'd']:
    X = make_basis(sunspot_counts[years<last_year], part=part, is_years=False)
    w = find_weights(X, Y)
    
    # Compute the regression line on a grid of inputs.
    # DO NOT CHANGE grid_years!!!!!
    # grid_years = np.linspace(1960, 2005, 200)
    grid_sunspots = np.linspace(min(sunspot_counts[years<last_year]), max(sunspot_counts[years<last_year]), 200)
    grid_X = make_basis(grid_sunspots, part=part, is_years=False)
    grid_Yhat = np.dot(grid_X, w)

    # TODO: plot and report sum of squared error for each basis
    print(f"L2 for part {part.upper()}: {sum((np.dot(X, w) - Y) ** 2)}")

    # Plot the data and the regression line.
    plt.plot(sunspot_counts[years<last_year], Y, 'o', grid_sunspots, grid_Yhat, '-')
    plt.xlabel("Number of Sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.title(f"Part {part.upper()} (Sunspots)")
    plt.savefig('sunspots_' + str(part) + '.png')
    plt.show()