import numpy as np

def distance(dimensions = 0, array1 = [], array2 = []):
    sum = 0
    for i in range (0, dimensions):
        dif = array1[i] - array2[i]
        square = dif ** 2
        sum += square

    distance = np.sqrt(sum)
    return distance



if __name__ == '__main__':
    print distance(4, [0,0,0,0], [0,3,4,0])
