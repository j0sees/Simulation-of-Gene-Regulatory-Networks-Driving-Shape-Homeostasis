import os
import sys
import numpy as np
import matplotlib
# WARNING to use in ozzy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# to check for matplotlib backend: >> matplotlib.get_backend()
from matplotlib.colors import ListedColormap
import subprocess as sp
import neat
import tools

def FitnessMapPlot():
    fileName = 'filesList'                                                  # name of the temporary file with names
    createList = 'ls plots/ | egrep 20180209 > {}'.format(fileName)     # command to generate such file
    sp.call(createList, shell = True)                                       # create the list
    fileList = open(fileName).read().splitlines()                           # store names in a python list for later use
    sp.call('rm {}'.format(fileName), shell = True)                         # remove temporary file
    
    print('=> List created...')
    statsArray = np.zeros([10,5,5])
    cCounter = 0
    for statsFile in fileList:
        with open('stats/{}_fitness_history.csv'.format(statsFile), 'r') as f:
            tempArray = np.loadtxt(f,delimiter=' ')

        for ix in range(10):
            statsArray[ix, cCounter//5, cCounter%5] = tempArray[ix,0]
        cCounter += 1

    print('=> Data stored...')
    print('=> Generating figures...')

    for ix in range(10):
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #fig.suptitle('')
        xticks = [ '{0}'.format(iy) for iy in np.linspace(0.1,1,10)]
        yticks = [ '{0:.02f}'.format(iy) for iy in np.linspace(0.01,0.1,10)]
        ticks = np.linspace(1,10,10)-1

        fig.canvas.draw()

        xlabels = [item.get_text() for item in ax.get_xticklabels()]
        xlabels = xticks

        ylabels = [item.get_text() for item in ax.get_yticklabels()]
        ylabels = yticks

        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        ax.set_xlabel('$P_{connection}$')
        ax.set_ylabel('$P_{node}$')

        ax.legend(loc = 'best')

        ax.set_title('Max fitness map for generation #{0}'.format(ix+1))
        mapPlot = ax.imshow(statsArray[ix,:,:], origin = 'lower', cmap = 'Blues', interpolation = 'none', vmin = 0, vmax = 1)
        cbar1 = fig.colorbar(mapPlot, ax = ax, orientation='vertical')#, shrink=0.75)
        #print('current data array:\n{}'.format(statsArray[ix,:,:]))
        plt.savefig('max_fitness_map_gen_{0}.eps'.format(ix+1), format='eps', bbox_inches='tight')
        #plt.show()
    
def FitnessperGenMap():
    fileName = 'filesList'                                                  # name of the temporary file with names
    folder = '20180214_pconnection_20_gen'
    createList = 'ls plots/{0} > {1}'.format(folder, fileName)              # command to generate such file
    sp.call(createList, shell = True)                                       # create the list
    fileList = open(fileName).read().splitlines()                           # store names in a python list for later use
    sp.call('rm {}'.format(fileName), shell = True)                         # remove temporary file

    print('=> Fitness vs generations map')    
    print('\tList created...')
    statsArray = np.zeros([20,10])
    cCounter = 0
    for statsFile in fileList:
        with open('stats/{}_fitness_history.csv'.format(statsFile), 'r') as f:
            tempArray = np.loadtxt(f, delimiter = ' ')
        statsArray[:,cCounter] = tempArray[:,0]
        cCounter += 1

    print('\tData stored...')
    print('\tGenerating map...')

    #-------------------------------#
    #       Generate histogram      #
    #-------------------------------#
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #fig.suptitle('')
    fig.canvas.draw()

    xticks = ['{0:.01f}'.format(iy) for iy in np.linspace(0.1,1,10)]
    yticks = ['{0}'.format(int(iy)) for iy in np.linspace(1,20,20)]
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    xlabels = xticks
    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    ylabels = yticks
    ticks = np.linspace(1,20,20) - 1
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(xlabels, rotation = -90)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel('Generation')
    ax.set_xlabel('$P_{connection}$')

    ax.set_title('Max Fitness map per generation')   
    mapPlot = ax.imshow(statsArray, origin = 'lower', cmap = 'Greens', interpolation = 'none', vmin = 0, vmax = 1)
    cbar1 = fig.colorbar(mapPlot, ax = ax, orientation='vertical')
    plt.savefig('max_fitness_map_per_gen.eps', format = 'eps', bbox_inches = 'tight')

def NetworkBehaviourMap(location, fileID):#, iGenome):
    """
    Function that plots a map showing the fixed points for the network status, GF production and polarisation
    """
    #print('=> Map plot')
    print('\nPlotting maps for networks in folder {}'.format(fileID))
    #location = 'plots/20182214_pconnection_20_gen'
    scale = 100
    SGF_size = 100                             # Seems to be enough
    LGF_size = 100
    marker_size = 0.1
    reps = 200                              # Same number of time steps of the cellular system
    nOutputs = 2
    SGF_max = 1#1
    LGF_max = 1
    
    #SGF_range = np.linspace(0, SGF_max, SGF_size + 1)
    #LGF_range = np.linspace(0, LGF_max, LGF_size + 1)

    SGF_range = np.arange(0, SGF_max + (1./scale), (1./scale))
    LGF_range = np.arange(0, LGF_max + (1./scale), (1./scale))


    GF_map = np.full([len(SGF_range), len(LGF_range), nOutputs], -1)
    GF_prod_map = np.zeros([len(SGF_range), len(LGF_range), 2], dtype = np.float64)
    
    #network = tools.GetNetwork(fileID, iGenome)
    genomes, config = tools.GetNetwork(fileID)#, iGenome)
    
    for iGenome in range(len(genomes)):
        network = neat.nn.RecurrentNetwork.create(genomes[iGenome], config)
        #print('\tGetting & processing data...')
        # Get data and process it...
        #for ix in LGF_range:
        for ix in range(len(LGF_range)):
            #for iy in SGF_range:
            for iy in range(len(SGF_range)):
                network.reset()
                inputs = [iy/float(scale), ix/float(scale)]
                #print('pos {}'.format(inputs))
                for _ in range(reps):
                    output = network.activate(inputs)
                GF_map[iy, ix, :] = tools.GenerateStatus(output)
                GF_prod_map[iy, ix, 0] = output[3]
                GF_prod_map[iy, ix, 1] = output[4]

        #-----------------------------------#
        #       Generate Behaviuor map      #
        #-----------------------------------#
        plt.close()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
        
        pad_dist = 0.02
        title_font_size = 8
        label_font_size = 5
        text_font_size = 5
        aspect = 1

        xticks = ['{0:.01f}'.format(iy) for iy in np.linspace(0,1,10 + 1)]
        yticks = ['{0:.01f}'.format(iy) for iy in np.linspace(0,1,10 + 1)]
        ticks = np.linspace(0,scale,11)
        
        cBehavMap = ListedColormap(['g', 'r', 'b', 'w'])

        ylabels = [item.get_text() for item in ax1.get_yticklabels()]
        ylabels = yticks

        ax1.set_yticks(ticks)
        ax1.set_yticklabels(ylabels, fontsize=label_font_size)
        ax1.set_ylabel('SGF', fontsize=label_font_size)
        ax1.set_title('Status map', fontsize=title_font_size)

        behaviourMap = ax1.imshow(GF_map[:,:,0], origin = 'lower', cmap = cBehavMap, interpolation = 'none', vmin = 1, vmax = 4)
        ax1.set_aspect(aspect='equal', adjustable='box-forced')
        cbar1 = fig.colorbar(behaviourMap, ax = ax1, ticks = [], orientation='vertical', pad=pad_dist)
        cbar1.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(['$quiescent$', '$proliferation$', '$migration$', '$apoptosis$']):
            cbar1.ax.text(0.5, (2 * j + 1) / 8.0, lab, ha = 'center', va = 'center', rotation=270, fontsize=text_font_size)
        cbar1.ax.get_yaxis().labelpad = 8
        cbar1.ax.set_ylabel('states', rotation = 270, fontsize=label_font_size)

        #--------------------------------------#
        #       Generate polarisation map      #
        #--------------------------------------#
        cPolMap = ListedColormap(['w', 'r', 'c', 'm', 'y'])

        ax2.set_title('Polarisation map', fontsize=title_font_size)
        
        polMap = ax2.imshow(GF_map[:,:,1], origin = 'lower', cmap = cPolMap, interpolation = 'none', vmin = 0, vmax = 4)
        ax2.set_aspect(aspect='equal', adjustable='box-forced')
        cbar2 = fig.colorbar(polMap, ax = ax2, ticks = [], orientation='vertical', pad=pad_dist)
        for j, lab in enumerate(['$none$', '$west$','$north$','$east$','$south$']):
            cbar2.ax.text(0.5, (2 * j + 1) / 10.0, lab, ha = 'center', va = 'center', rotation=270, fontsize=text_font_size)
        cbar2.ax.get_yaxis().labelpad = 8
        cbar2.ax.set_ylabel('polarisation', rotation = 270, fontsize=label_font_size)

        #----------------------------------------#
        #       Generate SGF production map      #
        #----------------------------------------#
        xlabels = [item.get_text() for item in ax3.get_xticklabels()]
        xlabels = xticks
        ylabels = [item.get_text() for item in ax3.get_yticklabels()]
        ylabels = yticks

        ax3.set_yticks(ticks)
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(xlabels, fontsize=label_font_size)#, rotation=-90)
        ax3.set_yticklabels(ylabels, fontsize=label_font_size)

        ax3.set_ylabel('SGF', fontsize=label_font_size)
        ax3.set_xlabel('LGF', fontsize=label_font_size)
        ax3.set_title('SGF production map', fontsize=title_font_size)

        SGFmax = np.amax(GF_prod_map[:,:,0])
        sgfProdMap = ax3.imshow(GF_prod_map[:,:,0], origin = 'lower', cmap = 'Reds', interpolation = 'none', vmin = 0, vmax = SGFmax)
        ax3.set_aspect(aspect='equal', adjustable='box-forced')
        ax3.text(5, 95, 'Max SGF value = {0:.02f}'.format(SGFmax), bbox={'facecolor': 'white', 'pad': 0.4, 'boxstyle':'round'}, fontsize=label_font_size)
        cbar3 = fig.colorbar(sgfProdMap, ax = ax3, orientation='vertical', pad=pad_dist)
        cbar3.ax.tick_params(labelsize=label_font_size)

        #----------------------------------------#
        #       Generate LGF production map      #
        #----------------------------------------#
        xlabels = [item.get_text() for item in ax4.get_xticklabels()]
        xlabels = xticks

        ax4.set_xticks(ticks)
        ax4.set_xticklabels(xlabels, fontsize=label_font_size)#, rotation=-90)
        ax4.set_xlabel('LGF', fontsize=label_font_size)
        ax4.set_title('LGF production map', fontsize=title_font_size)

        LGFmax = np.amax(GF_prod_map[:,:,1])
        lgfProdMap = ax4.imshow(GF_prod_map[:,:,1], origin = 'lower', cmap = 'Blues', interpolation = 'none', vmin = 0, vmax = LGFmax)
        ax4.set_aspect(aspect = 'equal', adjustable = 'box-forced')
        ax4.text(5, 95, 'Max LGF value = {0:.02f}'.format(LGFmax), bbox={'facecolor': 'white', 'pad': 0.4, 'boxstyle':'round'}, fontsize=label_font_size)        
        cbar4 = fig.colorbar(lgfProdMap, ax = ax4, orientation = 'vertical', pad=pad_dist)
        cbar4.ax.tick_params(labelsize=label_font_size)

        fig.tight_layout(w_pad=-5)

        # Save figure with four subplots
        plt.savefig('{0}/{1}/{1}_best_genome_{2}_maps.eps'.format(location, fileID, iGenome+1), format='eps', bbox_inches='tight')

def GF_AverageMap(SGF_mean, LGF_mean, location, iGenome):
    """
    Function that plots the average distribution of GF in the grid
    """
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,5))

    pad_dist = 0.02

    SGF_max = np.amax(SGF_mean)
    LGF_max = np.amax(LGF_mean)

    ax1.set_title('SGF average distribution over time')
    ax2.set_title('LGF average distribution over time')

    SGF_dist = ax1.imshow(SGF_mean, origin = 'lower', cmap = 'Oranges', interpolation = 'none', vmin = 0.0, vmax = SGF_max)
    ax1.set_aspect(aspect = 'equal')
    LGF_dist = ax2.imshow(LGF_mean, origin = 'lower', cmap = 'Oranges', interpolation = 'none', vmin = 0.0, vmax = LGF_max)
    ax2.set_aspect(aspect = 'equal')

    cbar1 = fig.colorbar(SGF_dist, ax = ax1, orientation = 'vertical', shrink = 0.76, pad = pad_dist)
    cbar2 = fig.colorbar(LGF_dist, ax = ax2, orientation = 'vertical', shrink = 0.76, pad = pad_dist)    

    # hide ticks
    SGF_dist.axes.xaxis.set_ticklabels([])
    SGF_dist.axes.yaxis.set_ticklabels([])
    SGF_dist.axes.get_xaxis().set_visible(False)
    SGF_dist.axes.get_yaxis().set_visible(False)
    
    LGF_dist.axes.xaxis.set_ticklabels([])
    LGF_dist.axes.yaxis.set_ticklabels([])
    LGF_dist.axes.get_xaxis().set_visible(False)
    LGF_dist.axes.get_yaxis().set_visible(False)

    plt.savefig('{0}/GF_average_dist_best_genome_{1}.eps'.format(location, iGenome + 1), format='eps', bbox_inches='tight')

def NetworkClustering(run_folder):
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import AffinityPropagation
    from sklearn import metrics
    import pandas as pd
    #from sklearn.datasets.samples_generator import make_blobs
    #from sklearn.preprocessing import StandardScaler

    GenomicDMatrix, g_distance, g_path = tools.GenomicDistanceMatrix(run_folder)
    HammingDMatrix, h_distance, h_path = tools.HammingDistanceMatrix(run_folder)
    
    plt.close()
    
    #g_path.append('lala8')
    assert (g_path == h_path), 'path lists are different!'
    
    #fig = plt.figure()
    
    #labels_matrix = np.zeros([len(g_path), 5])                   # matrix to save labels
    #labels_matrix[:,0] = h_path                                 # first column contains paths for networks
    list_ = [g_distance, h_distance]
    
    for dist in list_:
        if dist == 'genomic':
            iMatrix = GenomicDMatrix
            # eps_val should be
            eps_val = 0.6       # The maximum distance between two samples for them to be considered as in the same neighborhood.
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
        #for eps_val in np.arange(0.05,5,0.05):
            #print('\teps val {}:'.format(eps_val))
            #cluster_instance = DBSCAN(metric = 'precomputed', n_jobs = 2, eps = eps_val)
            #db = cluster_instance.fit(iMatrix)
            #labels_true = cluster_instance.fit_predict(iMatrix)
            #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            #core_samples_mask[db.core_sample_indices_] = True
            #DB_labels = db.labels_
            ## Number of clusters in labels, ignoring noise if present.
            #n_clusters_ = len(set(DB_labels)) - (1 if -1 in DB_labels else 0)
            #try:
                #print('\t\t=> [DBScan] Calinski-Harabaz Coefficient: {0:.03f}'.format(metrics.calinski_harabaz_score(iMatrix, DB_labels)))
                #print('\t\t=> [DBScan] Silhouette Coefficient: {0:.3f}'.format(metrics.silhouette_score(iMatrix, DB_labels)))
                #print('\t\t=> [DBScan] Number of clusters: {}'.format(n_clusters_))
            #except ValueError:
                ##DB_labels = [0 for _ in range(len(g_path))]
                #print('\t\t[DBScan] Error! Only {} cluster(s)!'.format(n_clusters_))
                #continue

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
        for preference_val in np.arange(0.05,5,0.05):
            print('\tpreference val: {}'.format(preference_val))
            af = AffinityPropagation(affinity = 'precomputed', preference = preference_val).fit(iMatrix)
            af = AffinityPropagation(affinity = 'precomputed').fit(iMatrix)#, preference = preference_val
            cluster_centers_indices = af.cluster_centers_indices_
            AP_labels = af.labels_
            n_clusters_ = len(cluster_centers_indices)
            try:
                print('\t\t=> [AP] Silhouette Coefficient: {0:.3f}'.format(metrics.silhouette_score(iMatrix, AP_labels)))
                print('\t\t=> [AP] Calinski-Harabaz Coefficient: {0:.03f}'.format(metrics.calinski_harabaz_score(iMatrix, AP_labels)))
                print('\t\t=> [AP] Number of clusters: {}'.format(n_clusters_))
            except ValueError:
                print('\t\t[AP] Error! Only {} cluster!'.format(n_clusters_))
                continue

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
#

if __name__ == '__main__':
    location = sys.argv[1]
    dataFile = sys.argv[2]
    #iGenome = int(sys.argv[2])
    #FitnessMapPlot()
    #FitnessperGenMap()
    NetworkBehaviourMap(location, dataFile)#, iGenome)
    #NetworkClustering(run_folder)
