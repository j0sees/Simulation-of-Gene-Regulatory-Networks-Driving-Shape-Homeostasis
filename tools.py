import numpy as np
from scipy import linalg
import csv
import neat
import os
import pickle
import subprocess as sp
#from numba import jit

# Tools

# List without joint ends
# https://stackoverflow.com/questions/29710249/python-force-list-index-out-of-range-exception
class flatList(list):
    def __getitem__(self, index):
        if index < 0:
            raise IndexError("list index out of range")
        return super(flatList, self).__getitem__(index)

#@jit
def CheckifOccupied(xCoord, yCoord, grid):
    if grid[yCoord][xCoord] > 0:         # if value on grid is 1 (quiet), 2 (moved) or 3 (splitted) then spot is occupied
        return True
    else:                                   # else, value is 0 (empty)
        return False
# CheckifOccupied

#@jit
def CheckifPreferred(xOri, yOri, xCoord, yCoord):
    if xCoord == xOri and yCoord == yOri:
        return True
    else:
        return False
# CheckifPreferred

# SGF dynamics with matrix approach
#@jit #WARNING ON is good!!
def SGFDiffEq(s_matrix, sigma_matrix, deltaS, deltaT):
    updated_matrix = s_matrix + deltaT*(sigma_matrix - deltaS*s_matrix)
    return updated_matrix
# sgfDiffEq

# TODO use linalg solve to make it faster and numerically more stable
# LGF dynamics with matrix approach
#@jit # WARNING ON is good!!
def LGFDiffEq(i_matrix, t_matrix, l_matrix, lambda_matrix, deltaL, deltaT, deltaR, D):
    alpha = D*deltaT/(deltaR**2)                            # constant
    f = (deltaT/2.)*(lambda_matrix - deltaL*l_matrix)       # term that takes into account LFG production for half time step
    g = linalg.inv(i_matrix - (alpha/2.)*t_matrix)          # inverse of some intermediate matrix
    h = i_matrix + (alpha/2.)*t_matrix                      # some intermediate matrix
    #l_halftStep = g@(l_matrix@h + f)                        # half time step calculation for LGF values
    l_halftStep = np.matmul(g,(np.matmul(l_matrix,h) + f))                        # half time step calculation for LGF values
    #print('grid after half time step...\n' + str(l_halftStep))
    f = (deltaT/2.)*(lambda_matrix - deltaL*l_halftStep)    # updated term...
    l_tStep = np.matmul((np.matmul(h,l_halftStep) + f),g)                         # final computation
    return l_tStep
# sgfDiffEq

#@jit
def GenerateTMatrix(size):
    t_matrix = np.zeros([size,size])
    for ix in range(size - 1):
        t_matrix[ix,ix] = -2.
        t_matrix[ix,ix + 1] = 1.
        t_matrix[ix + 1,ix] = 1.
    t_matrix[0,0] = -1.
    t_matrix[size - 1, size - 1] = -1.
    return t_matrix
# GenerateTMatrix

# Identity matrix
#@jit
def GenerateIMatrix(size):
    I_matrix = np.zeros([size,size])
    for ix in range(size):
        I_matrix[ix,ix] = 1.
    return I_matrix
# GenerateIMatrix

#@jit #WARNING ON is good!
def RecurrentNeuralNetwork(inputs, wMatrix, V):             # Recurrent Neural Network dynamics
    #beta = 2
    # bj = wMatrix@V - inputs
    bj = np.matmul(wMatrix,V) - inputs
    # might be improved ussing list comprehension...
    for ix in range(len(bj)):
        V[ix] = 1./(1 + np.exp(-2*bj[ix]))   #TransferFunction(bj[ix],2)
    # V = [1./(1 + np.exp(-2*bj[ix])) for ix in range(len(bj))]
    return V
# NeuralNetwork

#@jit
def GetStructure(cell_array, nLattice):
    structure = np.zeros([nLattice,nLattice])
    for ik in range(nLattice):
        for jk in range(nLattice):
            if cell_array[ik,jk] != 0:
                structure[ik,jk] = 1
    return structure
# GetStructure

def GetrNN(csvFile, ind):
    #with open('successful_test.csv', 'r') as csvfile:
    with open(csvFile, 'r') as csvfile:
        #reader = csv.reader(csvfile)
        bestIndividuals = np.loadtxt(csvfile, delimiter = ',')
    # get nNodes from nGenes
    nNodes = int(np.sqrt(len(bestIndividuals[ind,:])))
    wMatrix = np.array(bestIndividuals[ind,:].reshape(nNodes,nNodes))
    return wMatrix

def GetPop(csvFile):
    with open(csvFile, 'r') as csvfile:
        #reader = csv.reader(csvfile)
        networkContainer = np.loadtxt(csvfile, delimiter = ',')
    return networkContainer

def GetNetwork(fileID):#, iGenome):
    """
    Script that quickly loads a rNN created by NEAT
    """
    config_file = 'genomes/{}_config'.format(fileID)
    fileName = 'genomes/{}_best_unique_genomes'.format(fileID)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)

    # Config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # load the winner
    with open(fileName, 'rb') as f:
        genomes = pickle.load(f)#, encoding = 'bytes')

    #return neat.nn.RecurrentNetwork.create(genomes[iGenome], config)
    return genomes, config
    
def GenerateStatus(output):
    """
    Generate cell status out of the network output
    """
    status_data = np.zeros([2], dtype = int)     # [status, polarisation]
    
    # Cellular states
    iStatus = output[0]             # Proliferate: Split
    jStatus = output[1]             # Migrate:     Move
    kStatus = output[2]             # Apoptosis:   Die
    # Values for SGF and LGF
    #status_data[2] = output[3]      # SGF Prod
    #status_data[3] = output[4]      # LGF Prod
    # Polarisation
    compass = output[5]

    xThreshold = 0.5
    yThreshold = 0.001

    # Orientation boundaries:
    nBoundary = 0.25
    sBoundary = 0.5
    eBoundary = 0.75

    # oriented according to numpy order v>, not usual >^
    if abs(compass - sBoundary) <= yThreshold:  # compass == 0.5
        status_data[1] = 0                      # no orientation
        #print('no orientation')
    elif compass < sBoundary:                   # compass != 0.5
        if compass < nBoundary:                 # 0 <= compass < 0.25
            status_data[1] = 1                  # orientation West
            #print('orientation west')
        else:                                   # 0.25 <= compass < 0.5
            status_data[1] = 2                  # orientation North
            #print('orientation north')
    else: 
        if compass <= eBoundary:               # 0.5 < compass <= 0.75
            status_data[1] = 3                  # orientation East
            #print('orientation east')            
        else:                                   # 0.75 < compass <= 1
            status_data[1] = 4                  # orientation South
            #print('orientation south')
    
    if iStatus < xThreshold and jStatus < xThreshold and kStatus < xThreshold:
        status_data[0] = 1          # 'Quiet'
    else:
        for ix in iStatus, jStatus, kStatus:
            if xThreshold < ix:
                xThreshold = ix
        if abs(xThreshold - iStatus) <= yThreshold:
            status_data[0] = 2      # 'Split'
        elif abs(xThreshold - jStatus) <= yThreshold:
            status_data[0] = 3      # 'Move'
        else:
            status_data[0] = 4      # 'Die'
    return status_data
# Generate state

def NetworkClustering(run_folder):
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import AffinityPropagation
    from sklearn import metrics
    import pandas as pd
    #from sklearn.datasets.samples_generator import make_blobs
    #from sklearn.preprocessing import StandardScaler

    GenomicDMatrix, g_distance, g_path = GenomicDistanceMatrix(run_folder)
    #HammingDMatrix, h_distance, h_path = tools.HammingDistanceMatrix(run_folder)
    
    #plt.close()
    
    #g_path.append('lala8')
    #assert (g_path == h_path), 'path lists are different!'
    
    #fig = plt.figure()
    
    #labels_matrix = np.zeros([len(g_path), 5])                   # matrix to save labels
    #labels_matrix[:,0] = h_path                                 # first column contains paths for networks
    list_ = [g_distance]#, h_distance]
    
    for dist in list_:
        if dist == 'genomic':
            iMatrix = GenomicDMatrix
            # eps_val should be
            eps_val = 1.0       # The maximum distance between two samples for them to be considered as in the same neighborhood.
            preference_val = 0.8
            distance_used = 'Genomic_distance'
        else:
            iMatrix = HammingDMatrix
            eps_val = 0.9
            preference_val = 0.9
            distance_used = 'Hamming_distance'

        print('\nRunning clustering algorithms using {}'.format(distance_used))

        #---------------------------#
        #   DBScan Implementation   #
        #---------------------------#
        for eps_val in np.arange(0.399,0.41,0.001):
            print('\teps val {}:'.format(eps_val))
            cluster_instance = DBSCAN(metric = 'precomputed', n_jobs = 2, eps = eps_val)
            db = cluster_instance.fit(iMatrix)
            labels_true = cluster_instance.fit_predict(iMatrix)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            DB_labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(DB_labels)) - (1 if -1 in DB_labels else 0)
            try:
                print('\t\t=> [DBScan] Calinski-Harabaz Coefficient: {0:.03f}'.format(metrics.calinski_harabaz_score(iMatrix, DB_labels)))
                print('\t\t=> [DBScan] Silhouette Coefficient: {0:.3f}'.format(metrics.silhouette_score(iMatrix, DB_labels)))
                print('\t\t=> [DBScan] Number of clusters: {}'.format(n_clusters_))
                print('\t\t=> Labels: {}'.format(DB_labels))
            except ValueError:
                #DB_labels = [0 for _ in range(len(g_path))]
                print('\t\t[DBScan] Error! Only {} cluster(s)!'.format(n_clusters_))
                continue

        ## DBScan plot
        #plt.clf()
        #unique_labels = set(DB_labels)
        #colors = [plt.cm.Spectral(each)
                #for each in np.linspace(0, 1, len(unique_labels))]
        #for k, col in zip(unique_labels, colors):
            #if k == -1:
                ## Black used for noise.
                #col = [0, 0, 0, 1]
            #class_member_mask = (DB_labels == k)
            #xy = iMatrix[class_member_mask & core_samples_mask]
            #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    #markeredgecolor='k', markersize=14)
            #xy = iMatrix[class_member_mask & ~core_samples_mask]
            #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    #markeredgecolor = 'k', markersize = 6)
        #plt.title('DBScan Clustering using {0}\nEstimated number of clusters: {1}'.format(distance_used, n_clusters_))
        #plt.savefig('plots/{0}/{1}_DBScan_clustering.eps'.format(run_folder, distance_used), format='eps', bbox_inches='tight')

        #---------------------------#
        #   Affinity Propagation    #
        #---------------------------#
        #for preference_val in np.arange(0.05,1,0.05):
            #print('\tpreference val: {}'.format(preference_val))
            #af = AffinityPropagation(affinity = 'precomputed', preference = preference_val).fit(iMatrix)
            #af = AffinityPropagation(affinity = 'precomputed').fit(iMatrix)#, preference = preference_val
            #cluster_centers_indices = af.cluster_centers_indices_
            #AP_labels = af.labels_
            #n_clusters_ = len(cluster_centers_indices)
            #try:
                #print('\t\t=> [AP] Silhouette Coefficient: {0:.3f}'.format(metrics.silhouette_score(iMatrix, AP_labels)))
                #print('\t\t=> [AP] Calinski-Harabaz Coefficient: {0:.03f}'.format(metrics.calinski_harabaz_score(iMatrix, AP_labels)))
                #print('\t\t=> [AP] Number of clusters: {}'.format(n_clusters_))
            #except ValueError:
                #print('\t\t[AP] Error! Only {} cluster!'.format(n_clusters_))
                #continue

        ## Affinity Propagation plot
        #plt.clf()
        #from itertools import cycle
        #colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        #for k, col in zip(range(n_clusters_), colors):
            #class_members = AP_labels == k
            #cluster_center = iMatrix[cluster_centers_indices[k]]
            #plt.plot(iMatrix[class_members, 0], iMatrix[class_members, 1], col + '.')
            #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
            #for x in iMatrix[class_members]:
                #plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
        #plt.title('Affinity Propagation Clustering using {0}\nEstimated number of clusters: {1}'.format(distance_used, n_clusters_))
        #plt.savefig('plots/{0}/{1}_AF_clustering.eps'.format(run_folder, distance_used), format='eps', bbox_inches='tight')

        ## Clustering measures
        #print('Estimated number of clusters: {}'.format(n_clusters_))
        ## This measures only work if the true labels (clusters) are known...
        ##print('Homogeneity: {0:.3f}'.format(metrics.homogeneity_score(labels_true, labels)))
        ##print('Completeness: {0:.3f}'.format(metrics.completeness_score(labels_true, labels)))
        ##print('V-measure: {0:.3f}'.format(metrics.v_measure_score(labels_true, labels)))
        ##print('Adjusted Rand Index: {0:.3f}'.format(metrics.adjusted_rand_score(labels_true, labels)))
        ##print('Adjusted Mutual Information: {0:.3f}'.format(metrics.adjusted_mutual_info_score(labels_true, labels)))
        #print('Silhouette Coefficient: {0:.3f}'.format(metrics.silhouette_score(iMatrix, labels)))
        #print('Calinski-Harabaz Coefficient: {0:.3f}'.format(metrics.calinski_harabaz_score(iMatrix, labels)))
        #print('Labels:\n{}'.format(labels))

        #if dist == 'genomic':
            #g_DB_labels = DB_labels  # Second column DBScan labels for genomic distance
            #g_AP_labels = AP_labels  # Third column Affinity Propagation labels for genomic distance
        #else:
            #h_DB_labels = DB_labels  # Fourth column DBScan labels for hamming distance
            #h_AP_labels = AP_labels  # Fifth column Affinity Propagation labels for hamming distance

    #datafile = {'network':h_path, 'DB_gDist':g_DB_labels, 'AP_gDist':g_AP_labels, 'DB_hDist':h_DB_labels, 'AP_hDist':h_AP_labels}
    #df = pd.DataFrame(datafile, columns = ['network', 'DB_gDist', 'AP_gDist', 'DB_hDist', 'AP_hDist'])
    #df.to_csv('plots/{}/clustering_labels.txt'.format(run_folder), sep='\t')
    #print('\nData file created!')    

# Clustering

def GenomicDistanceMatrix(run_file):
    #-------------------------------#
    #       Generate histogram      #
    #-------------------------------#
    #listName = 'files_list'                                                             # name of list 
    #IDFilesList_command = 'ls plots/{0} | egrep 2018 > {1}'.format(run_file, listName)    # command to generate such file
    #sp.call(IDFilesList_command, shell = True)
    #fileList = open(listName).read().splitlines()                                       # store names in a python list for later use
    #sp.call('rm {}'.format(listName), shell = True)                                     # remove temporary file    
    fileList = [run_file]
    genomeList = []
    configList = []
    names_List = []

    for iFile in fileList:                              # Iterate over file names
        genomes, config = GetNetwork(iFile)             # get genomes and config files for a specific folder in the run folder
        nGen = 0
        for iGenome in genomes:                         # iterate over genomes
            nGen += 1
            genomeList.append(iGenome)                  # save genomes in a single list
            configList.append(config)                   # save config files as well
            #path = 'plots/{0}/{1}/{1}_best_unique_network_{2}'.format(run_file, iFile, nGen)
            path = 'plots/{0}/{0}_best_unique_network_{2}'.format(run_file, iFile, nGen)
            names_List.append(path.split('/')[-1])

    #print('total length of genomes list: {}'.format(len(genomeList)))

    nGenomes = len(genomeList)
    GDMatrix = np.zeros([nGenomes, nGenomes], dtype = np.float64)
    avg_distance = 0.0
    n_distances = 0
    for iy in range(nGenomes):
        for ix in range(iy, nGenomes):
            GDMatrix[iy,ix] = genomeList[iy].distance(genomeList[ix], configList[ix].genome_config)
            if GDMatrix[iy,ix] > 0.0:
                avg_distance += GDMatrix[iy,ix]
                n_distances += 1

    assert (n_distances == (nGenomes*(nGenomes - 1))/2), 'Number of distances is incorrect!!'

    print('Number of distances:{}, avg distance: {}'.format(n_distances, avg_distance/n_distances))
    #print('genomic distance matrix:\n{}'.format(GDMatrix))
    #print('number of entries higher than 1: {}'.format(counter))
    return GDMatrix, 'genomic', names_List, avg_distance/n_distances

def GetSpeciesDistances(species, run_file):
    GDMatrix, _, names_List, avg_distance = GenomicDistanceMatrix(run_file)
    m, _ = np.shape(GDMatrix)
    
    n_inds = 0
    for ix in species:
        n_inds += len(ix)

    assert (m == n_inds), 'Species list is incorrect!'
    print('Population avg distance:{}'.format(avg_distance))

    species_distances = np.zeros([len(species)])
    for iSpec in range(len(species)):                       # iteration over species in list
        n_ind_spec = len(species[iSpec])                    # amount of individuals in species 
        n_distances = (n_ind_spec*(n_ind_spec - 1))/2
        temp_distance = 0.0
        for iInd in range(n_ind_spec - 1):                  # iteration over individuals in species
            for jInd in range(iInd + 1, n_ind_spec):            # iteration over the rest of individuals in species
                iIndex = species[iSpec][iInd]               # get indexes of individuals
                jIndex = species[iSpec][jInd]
                temp_distance += GDMatrix[iIndex, jIndex]
                print('g_distance between inds {0} and {1}: {2}'.format(iIndex + 1, jIndex + 1, GDMatrix[iIndex, jIndex]))
        species_distances[iSpec] = temp_distance/n_distances
        print('avg dist for species {}: {}\n'.format(iSpec + 1, species_distances[iSpec]))
    
    print(''.format())

def ReadDigraph(DiGraphFile):
    fileList = open(DiGraphFile).read().splitlines()        # load file as list of strings for each line
    del fileList [-1]                                       # delete last item: '}'
    fileList.reverse()                                      # reverse list to delete items
    for _ in range(10):                                     # iterate and delete
        del fileList [-1]
    fileList.reverse()                                      # turn list to original state
    #print('final list:')

    fileList = [string.split('[')[0].strip() for string in fileList]

    # code snippet taken from:
    #https://stackoverflow.com/questions/46458128/generate-adjacency-matrix-in-graphviz

    pairs = [line.replace(" ", "").split("->") for line in fileList]

    keys_inputs = {'SGF':0, 'LGF':1}
    keys_outputs = {'Proliferate':2, 'Migrate':3, 'Apoptosis':4, '"SGFProd"':5, '"LGFProd"':6, 'Polarisation':7}

    adjacency_matrix = np.zeros([8,8], dtype = np.int)

    for iKey, iPos in keys_inputs.iteritems():
        for jKey, jPos in keys_outputs.iteritems():
            for ip in pairs:
                if ip[0] == iKey and ip[1] == jKey:
                    adjacency_matrix[iPos,jPos] = 1
                    adjacency_matrix[jPos,iPos] = 1
                elif ip[0] == jKey:
                    adjacency_matrix[jPos,jPos] = 1

    return adjacency_matrix
    #print('{}'.format(adjacency_matrix))
    #keys_list = ['SGF', 'LGF', 'Proliferate', 'Migrate', 'Apoptosis', '"SGFProd"', '"LGFProd"', 'Polarisation']
    #unique_edges = set(all_edges)
    #matrix = {origin: {dest: 0 for dest in all_edges} for origin in all_edges}
    #for p in pairs:
        #matrix[p[0]][p[1]] += 1
    #import pprint
    #pprint.pprint(matrix)
    ##for ix in range(len(matrix)):
    ##print('{}'.format(matrix))
    #import pandas as pd
    #a = pd.DataFrame(matrix)
    ##print('{}'.format(a.to_string(na_rep='0')))

def GetHammingDistance(matrix_a, matrix_b):
    dim_a, dim_b = np.shape(matrix_a)
    distance = 0.
    scale = 1.#3*(dim_a - 2)
    
    for ix in range(2,dim_a):
        for iy in range(dim_a):
            if matrix_a[iy,ix] != matrix_b[iy,ix]:
                distance += 1
    return distance/scale

def HammingDistanceMatrix(run_file):
    #listName = 'files_list'                                                             # name of list 
    #IDFilesList_command = 'ls plots/{0} | egrep 2018 > {1}'.format(run_file, listName)  # command to generate such file
    #sp.call(IDFilesList_command, shell = True)
    #fileList = open(listName).read().splitlines()                                       # store names in a python list for later use
    #sp.call('rm {}'.format(listName), shell = True)                                     # remove temporary file    
    fileList = [run_file]
    
    adjacencyMatrixList = []
    names_List = []
    
    for iFile in fileList:                                  # Iterate over file names
        genomes, _ = GetNetwork(iFile)                      # get genomes and config files for a specific folder in the run folder
        nNetworks = len(genomes)
        for iNet in range(nNetworks):                       # iterate over genomes
            #path = 'plots/{0}/{1}/{1}_best_unique_network_{2}'.format(run_file, iFile, iNet + 1)
            path = 'plots/{0}/{0}_best_unique_network_{2}'.format(run_file, iFile, iNet + 1)
            adjacencyMatrixList.append(ReadDigraph(path))   # save genomes in a single list
            names_List.append(path.split('/')[-1])

    #print('total length of genomes list: {}'.format(len(adjacencyMatrixList)))
    
    nMatrices = len(adjacencyMatrixList)
    HDMatrix = np.zeros([nMatrices, nMatrices], dtype = np.float64)

    for iy in range(nMatrices):
        for ix in range(nMatrices):
            HDMatrix[iy,ix] = GetHammingDistance(adjacencyMatrixList[iy], adjacencyMatrixList[ix])
    #print('genomic distance matrix:\n{}'.format(GDMatrix))
    return HDMatrix, 'hamming', names_List

def PrintGenomicDistMatrix(run_folder):
    GDMatrix, g_distance, g_path = GenomicDistanceMatrix(run_folder)
    print('{}'.format(GDMatrix))

def Speciate(run_folder, comp_threshold):
    iRepr = 0               # First member of the list is the representative of the species, could be chosen randomly
    
    #listName = 'files_list'                                                             # name of list 
    #IDFilesList_command = 'ls plots/{0} | egrep 2018 > {1}'.format(run_file, listName)  # command to generate such file
    #sp.call(IDFilesList_command, shell = True)
    #fileList = open(listName).read().splitlines()                                       # store names in a python list for later use
    #sp.call('rm {}'.format(listName), shell = True)                                     # remove temporary file    
    fileList = [run_folder]
    
    names_List = []
    genomes_List = []
    config_List = []
    
    # Get and organise genomes info 
    for iFile in fileList:                                  # Iterate over file names
        genomes, config = GetNetwork(iFile)                 # get genomes and config files for a specific folder in the run folder
        nNetworks = len(genomes)
        for iNet in range(nNetworks):                       # iterate over genomes
            #path = 'plots/{0}/{1}/{1}_best_unique_network_{2}'.format(run_file, iFile, iNet + 1)
            path = 'plots/{0}/{0}_best_unique_network_{2}'.format(run_folder, iFile, iNet + 1)
            names_List.append(path.split('/')[-1])
            genomes_List.append(genomes[iNet])
            config_List.append(config)

    # Get ready to "speciate"
    nGenomes = len(genomes_List)
    species_list = [[]]                                     # list containing species
    species_list[0].append(0)                               # first individual belongs to the first species

    for ix in range(1, nGenomes):                           # iterteration through individuals
        new_species = True
        for iy in range(len(species_list)):                 # iteration through species
            repr_index = species_list[iy][iRepr] 
            dist = genomes_List[ix].distance(genomes_List[repr_index], config_List[repr_index].genome_config)
            print('genomic distance between {} and {}: {}'.format(ix, iy, dist))
            if dist <= comp_threshold:
                species_list[iy].append(ix)
                new_species = False
                break
            else:
                #print('distance between {0} and {1} higher than 1! dist = {2}'.format(ix, iy, dist))
                continue
        if new_species:
            species_list.append([ix])
            print('new species!')

    for ix in range(len(species_list)):
        print('Species #{}:'.format(ix + 1))
        for iy in range(len(species_list[ix])):
            print('\tInd #{0}: {1}'.format(iy + 1, names_List[species_list[ix][iy]]))
