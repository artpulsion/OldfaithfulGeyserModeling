# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 18:26:38 2016

@author: Sofiane
"""

import numpy as np
from math import *
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn


def read_file(filename):
    
    "Lit le fichier contenant les données du geyser Old Faithful"
    
    # lecture de l'en-tête jusqu'à ...
    infile = open(filename, "r")
    for ligne in infile:
        if ligne.find("eruptions waiting") != -1:
            break

    # ici, on a la liste des temps d'éruption et des délais d'irruptions
    data = []
    for ligne in infile:
        nb_ligne, eruption, waiting = [float(x) for x in ligne.split()]
        data.append(eruption)
        data.append(waiting)
    infile.close()

    # transformation de la liste en tableau 2D
    data = np.asarray(data)
    data.shape = (data.size / 2, 2)
    
    return data


# Read and structurate the data
path = "/home/sofianembarki/Artintel/old_faithful/data/data.txt"
data = read_file(path)



# Define bidimensional normal function
def normale_bidim(x, z, (mu_x, mu_z, sig_x, sig_z, p)):

    A = 1 / (2 * np.pi * sig_x * sig_z * np.sqrt((1 - ((p) ** 2))))
    B = 1 / (2 * (1 - (p**2)))
    C = ((x - mu_x) / sig_x) ** 2
    D = 2 * p * ((x - mu_x) * (z - mu_z) / (sig_x * sig_z))
    E = ((z - mu_z) / sig_z) ** 2
    res = A * np.exp(- B * (C - D + E))

    return res


# test fonction normale_bidim"
normale_bidim(1, 0, (1.0, 2.0, 1.0, 2.0, 0.7))


def dessine_1_normale(params):

    mu_x, mu_z, sigma_x, sigma_z, rho = params

    x_min = mu_x - 2 * sigma_x
    x_max = mu_x + 2 * sigma_x
    z_min = mu_z - 2 * sigma_z
    z_max = mu_z + 2 * sigma_z

    x = np.linspace(x_min, x_max, 100)
    z = np.linspace(z_min, z_max, 100)
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm = X.copy()
    for i in range(x.shape[0]):
        for j in range(z.shape[0]):
            norm[i, j] = normale_bidim(x[i], z[j], params)

    # affichage
    fig = plt.figure()
    plt.contour(X, Z, norm, cmap=cm.autumn)
    plt.show()


dessine_1_normale((-3.0, -5.0, 3.0, 2.0, 0.7))
dessine_1_normale((-3.0, -5.0, 3.0, 2.0, 0.2))


def dessine_normales(data, params, weights, bounds, ax):

    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # on détermine les coordonnées des coins de la figure
    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]

    # création de la grille
    nb_x = nb_z = 100
    x = np.linspace(x_min, x_max, nb_x)
    z = np.linspace(z_min, z_max, nb_z)
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm0 = np.zeros((nb_x, nb_z))
    for j in range(nb_z):
        for i in range(nb_x):
            norm0[j, i] = normale_bidim(x[i], z[j], params[0])  # * weights[0]

    norm1 = np.zeros((nb_x, nb_z))
    for j in range(nb_z):
        for i in range(nb_x):
            norm1[j, i] = normale_bidim(x[i], z[j], params[1])  # * weights[1]

# affichages des normales et des points du dataset
    ax.contour(X, Z, norm0, cmap=cm.winter, alpha=0.5)
    ax.contour(X, Z, norm1, cmap=cm.autumn, alpha=0.5)
    for point in data:
        ax.plot(point[0], point[1], 'k+')


def find_bounds(data, params):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # calcul des coins
    x_min = min(mu_x0 - 2 * sigma_x0, mu_x1 - 2 * sigma_x1, data[:, 0].min())
    x_max = max(mu_x0 + 2 * sigma_x0, mu_x1 + 2 * sigma_x1, data[:, 0].max())
    z_min = min(mu_z0 - 2 * sigma_z0, mu_z1 - 2 * sigma_z1, data[:, 1].min())
    z_max = max(mu_z0 + 2 * sigma_z0, mu_z1 + 2 * sigma_z1, data[:, 1].max())

    return (x_min, x_max, z_min, z_max)

# affichage des données : calcul des moyennes et variances des 2 colonnes
mean1 = data[:, 0].mean()
mean2 = data[:, 1].mean()
std1 = data[:, 0].std()
std2 = data[:, 1].std()

# les paramètres des 2 normales sont autour de ces moyennes
params = np.array([(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                   (mean1 + 0.2, mean2 + 1, std1, std2, 0)])
weights = np.array([0.4, 0.6])
bounds = find_bounds(data, params)

# affichage de la figure
fig = plt.figure()
ax = fig.add_subplot(111)
dessine_normales(data, params, weights, bounds, ax)
plt.show()


""" Attribution des paramètres """
current_params = np.array([[3.28778309, 69.89705882, 1.13927121, 13.56996002, 0.],
                           [3.68778309, 71.89705882, 1.13927121, 13.56996002, 0.]])
current_weights = np.array([0.5, 0.5])


"""Calculation for expectation """
def Q_i(data, current_params, current_weight):


    results = np.zeros(shape=(data.shape[0], data.shape[1]))

    for i in range(0, data.shape[0]):
        A = 1 / (2 * np.pi * current_params[0][2] * current_params[
                 0][3] * np.sqrt((1 - ((current_params[0][4]) ** 2))))
        B = 1 / (2 * (1 - (current_params[0][4]**2)))
        C = ((data[i][0] - current_params[0][0]) / current_params[0][2]) ** 2
        D = 2 * current_params[0][4] * ((data[i][0] - current_params[0][0]) * (
            data[i][1] - current_params[0][1]) / (current_params[0][2] * current_params[0][3]))
        E = ((data[i][1] - current_params[0][1]) / current_params[0][3]) ** 2
        a0 = current_weights[0] * A * np.exp(- B * (C - D + E))

        A = 1 / (2 * np.pi * current_params[1][2] * current_params[
                 1][3] * np.sqrt((1 - ((current_params[1][4]) ** 2))))
        B = 1 / (2 * (1 - (current_params[1][4]**2)))
        C = ((data[i][0] - current_params[1][0]) / current_params[1][2]) ** 2
        D = 2 * current_params[1][4] * ((data[i][0] - current_params[1][0]) * (
            data[i][1] - current_params[1][1]) / (current_params[1][2] * current_params[1][3]))
        E = ((data[i][1] - current_params[1][1]) / current_params[1][3]) ** 2
        a1 = current_weights[1] * A * np.exp(- B * (C - D + E))

        qi_y0 = a0 / (a0 + a1)
        qi_y1 = a1 / (a0 + a1)

        results[i][0] = qi_y0
        results[i][1] = qi_y1

    return results

T = Q_i(data, current_params, current_weights)


""" ### TEST avec un autre jeu de paramètre ###
current_params = np.array([[ 3.2194684, 67.83748075, 1.16527301, 13.9245876, 0.9070348 ],
                           [ 3.75499261, 73.9440348, 1.04650191, 12.48307362, 0.88083712]])
current_weights = np.array ( [ 0.49896815, 0.50103185] )
T_2 = Q_i ( data, current_params, current_weights ) """


""" Calculation for maximisation """
""" Attribution de nouveaux paramètres """
current_params = array([(2.51460515, 60.12832316, 0.90428702, 11.66108819, 0.86533355),
                        (4.2893485, 79.76680985, 0.52047055, 7.04450242, 0.58358284)])
current_weights = array([0.45165145, 0.54834855])

Q = Q_i(data, current_params, current_weights)


def M_step(data, Q, current_params, current_weights):
    """Fonction pour calculer les Maximums """

    params_bidim_0 = params_bidim_1 = np.zeros(shape=(1, 5))
    params_weights = np.zeros(shape=(1, 2))
    params = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

    ## WEIGHTS ##
    qi_y0 = 0
    for i in range(0, data.shape[0]):
        qi_y0 += Q[i][0]

    qi_y1 = 0
    for i in range(0, data.shape[0]):
        qi_y1 += Q[i][1]

    pi_0 = qi_y0 / (qi_y0 + qi_y1)
    pi_1 = qi_y1 / (qi_y0 + qi_y1)

    ## MU ##
    qi_y0_xi = 0
    for i in range(0, data.shape[0]):
        qi_y0_xi += (Q[i][0] * data[i][0])

    qi_y1_xi = 0
    for i in range(0, data.shape[0]):
        qi_y1_xi += (Q[i][1] * data[i][0])

    qi_y0_zi = 0
    for i in range(0, data.shape[0]):
        qi_y0_zi += (Q[i][0] * data[i][1])

    qi_y1_zi = 0
    for i in range(0, data.shape[0]):
        qi_y1_zi += (Q[i][1] * data[i][1])

    mu_x_0 = qi_y0_xi / qi_y0
    params[0][0] = mu_x_0

    mu_x_1 = qi_y1_xi / qi_y1
    params[0][1] = mu_x_1

    mu_z_0 = qi_y0_zi / qi_y0
    mu_z_1 = qi_y1_zi / qi_y1

    ## SIGMA ##
    qi_Y0_xi_mu_x_0 = 0
    for i in range(0, data.shape[0]):
        qi_Y0_xi_mu_x_0 += Q[i][0] * ((data[i][0] - mu_x_0)**2)

    qi_Y1_xi_mu_x_1 = 0
    for i in range(0, data.shape[0]):
        qi_Y1_xi_mu_x_1 += Q[i][1] * ((data[i][0] - mu_x_1)**2)

    qi_Y0_zi_mu_z_0 = 0
    for i in range(0, data.shape[0]):
        qi_Y0_zi_mu_z_0 += Q[i][0] * ((data[i][1] - mu_z_0)**2)

    qi_Y1_zi_mu_z_1 = 0
    for i in range(0, data.shape[0]):
        qi_Y1_zi_mu_z_1 += Q[i][1] * ((data[i][1] - mu_z_1)**2)

    sig_x_0 = np.sqrt(qi_Y0_xi_mu_x_0 / qi_y0)
    sig_x_1 = np.sqrt(qi_Y1_xi_mu_x_1 / qi_y1)
    sig_z_0 = np.sqrt(qi_Y0_zi_mu_z_0 / qi_y0)
    sig_z_1 = np.sqrt(qi_Y1_zi_mu_z_1 / qi_y1)

    ## RHO ##
    qi_y0_sig_xz_0 = 0
    for i in range(0, data.shape[0]):
        qi_y0_sig_xz_0 += Q[i][0] * ((data[i][0] - mu_x_0)
                                     * (data[i][1] - mu_z_0) / (sig_x_0 * sig_z_0))

    qi_y1_sig_xz_1 = 0
    for i in range(0, data.shape[0]):
        qi_y1_sig_xz_1 += Q[i][1] * ((data[i][0] - mu_x_1)
                                     * (data[i][1] - mu_z_1) / (sig_x_1 * sig_z_1))

    rho_0 = qi_y0_sig_xz_0 / qi_y0
    rho_1 = qi_y1_sig_xz_1 / qi_y1

    params = np.array([(mu_x_0, mu_z_0, sig_x_0, sig_z_0, rho_0),
                       (mu_x_1, mu_z_1, sig_x_1, sig_z_1, rho_1)]), np.array([pi_0, pi_1])
    return params

test = M_step(data, Q, current_params, current_weights)


# 7. Algorithme EM : mise au point

"""Initialisation des paramètres """
data = read_file(path)

mean1 = data[:, 0].mean()
mean2 = data[:, 1].mean()
std1 = data[:, 0].std()
std2 = data[:, 1].std()

params = np.array([(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                   (mean1 + 0.2, mean2 + 1, std1, std2, 0)])

weights = np.array([0.5, 0.5])


def EM(data, params, weights, nbr_iterations):
    for i in range(0, nbr_iterations):

        Q = Q_i(data, params, weights)  # étape E
        M = M_step(data, Q, params, weights)

        params = M[0]
        weights = M[1]

        bounds = find_bounds(data, params)

        # affichage de la figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dessine_normales(data, params, weights, bounds, ax)
        plt.show()

## test_final = EM(data, params, weights, 15)


### Version finale ###
data = read_file(path)

mean1 = data[:, 0].mean()
mean2 = data[:, 1].mean()
std1 = data[:, 0].std()
std2 = data[:, 1].std()

params = np.array([(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                   (mean1 + 0.2, mean2 + 1, std1, std2, 0)])

weights = np.array([0.5, 0.5])


def EM_final(data, params, weights, nbr_iterations):

    res_EM = []

    for i in range(0, nbr_iterations):
        Q = Q_i(data, params, weights)  # étape E
        M = M_step(data, Q, params, weights)
        
        params = M[0]
        weights = M[1]

        res_EM.append((params, weights))
    return res_EM

res_EM = EM_final(data, params, weights, 18)




# calcul des bornes pour contenir toutes les lois normales calculées
def find_video_bounds ( data, res_EM ):
    bounds = np.asarray ( find_bounds ( data, res_EM[0][0] ) )
    for param in res_EM:
        new_bound = find_bounds ( data, param[0] )
        for i in [0,2]:
            bounds[i] = min ( bounds[i], new_bound[i] )
        for i in [1,3]:
            bounds[i] = max ( bounds[i], new_bound[i] )
    return bounds

bounds = find_video_bounds ( data, res_EM )


import matplotlib.animation as animation

# création de l'animation : tout d'abord on crée la figure qui sera animée
fig = plt.figure ()
ax = fig.gca (xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]))

# la fonction appelée à chaque pas de temps pour créer l'animation
def animate ( i ):
    ax.cla ()
    dessine_normales (data, res_EM[i][0], res_EM[i][1], bounds, ax)
    ax.text(5, 40, 'step = ' + str ( i ))
    print "step animate = %d" % ( i )

# exécution de l'animation
anim = animation.FuncAnimation(fig, animate, 
                               frames = len ( res_EM ), interval=500 )
plt.show ()

# éventuellement, sauver l'animation dans une vidéo
anim.save('old_faithful.avi', bitrate=4000)
