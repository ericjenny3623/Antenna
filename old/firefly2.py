#This file is the firefly algorithm.
#Hook into other files as needed for global variables/functions

import numpy as np
import math
import functions
import settings

pi = np.pi

x = [0 for i in range(0, settings.N_FF)]
R = [0 for i in range(0, settings.N_FF)]
error = [0 for i in range(0, settings.N_FF)]

#Initialize our constraint
constraint = functions.compute_constraint(settings.N_PHI, settings.BW_DES, settings.SLL_DES)

def firefly(alpha, gamma, N_FF, iterations):
    
    #Initialize the arrays for each firefly
    for i in range(0, settings.N_FF):
        x[i] = functions.compute_x(settings.N)
        x[i] = sorted(x[i])
        R[i] = functions.compute_R(settings.KHAT, settings.N_PHI, settings.N, x[i])
        #Don't forget to convert R to dB!
        error[i] = functions.compute_error(settings.N_PHI, 20 * np.log10(np.absolute(R[i])), constraint)

    for iteration in range(0, iterations):

        #Convergent movement
        #Unsynchronized
        #Compare each firefly to every other firefly
        for i in range(0, settings.N_FF):
            for j in range(0, settings.N_FF):
                #Don't move towards itself
                if(i != j and error[j] < error[i]):
                    distance = functions.compute_cartesian_distance(settings.N, x[i], x[j])
                    x[i] = [a + (b - a) * math.exp((-1.0 / gamma) * distance**2) for a, b in zip(x[i], x[j])]

        #Random movement
        #Uniformly distributed in [-0.5, 0.5]
        for i in range(0, settings.N_FF):
            #Each element needs a unique random movement
            for j in range(0, settings.N):

                #Shape random movement to avoid going outside aperture
                #Maintain zero mean random movement
                #Lower edge
                if(x[i][j] <= (alpha / 2)):
                    temp = np.random.uniform(low = -alpha / 2, high = alpha / 2)
                    if temp < 0:
                        x[i][j] = x[i][j] + (1 - float(iteration) / iterations) * np.random.uniform(low = -x[i][j], high = 0)
                    else:
                        x[i][j] = x[i][j] + (1 - float(iteration) / iterations) * temp
                #Upper edge
                elif(x[i][j] >= (1 - alpha / 2)):
                    temp = np.random.uniform(low = -alpha / 2, high = alpha / 2)
                    if temp > 0:
                        x[i][j] = x[i][j] + (1 - float(iteration) / iterations) * np.random.uniform(low = 0, high = (1 - x[i][j]))
                    else:
                        x[i][j] = x[i][j] + (1 - float(iteration) / iterations) * temp
                else:
                    x[i][j] = x[i][j] + (1 - float(iteration) / iterations) * np.random.uniform(low = -alpha / 2, high = alpha / 2)

            #Don't forget to sort x!
            x[i] = sorted(x[i])
            R[i] = functions.compute_R(settings.KHAT, settings.N_PHI, settings.N, x[i])
            #Don't forget to convert R to dB!
            error[i] = functions.compute_error(settings.N_PHI, 20 * np.log10(np.absolute(R[i])), constraint)

    #END ITERATIONS

    #After the last iteration compute the response and error again
    for i in range(0, settings.N_FF):
        R[i] = functions.compute_R(settings.KHAT, settings.N_PHI, settings.N, x[i])
        #Don't forget to convert R to dB!
        error[i] = functions.compute_error(settings.N_PHI, 20 * np.log10(np.absolute(R[i])), constraint)

    return x, R, error

