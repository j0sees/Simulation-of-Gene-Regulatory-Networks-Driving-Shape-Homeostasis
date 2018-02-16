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
import pickle

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
            #statsArray[0,pCounter,cCounter] = tempArray[0,0]
            #statsArray[1,pCounter,cCounter] = tempArray[1,0]
            #statsArray[2,pCounter,cCounter] = tempArray[2,0]
            #statsArray[3,pCounter,cCounter] = tempArray[3,0]
            #statsArray[4,pCounter,cCounter] = tempArray[4,0]
            #statsArray[5,pCounter,cCounter] = tempArray[5,0]
            #statsArray[6,pCounter,cCounter] = tempArray[6,0]
            #statsArray[7,pCounter,cCounter] = tempArray[7,0]
            #statsArray[8,pCounter,cCounter] = tempArray[8,0]
            #statsArray[9,pCounter,cCounter] = tempArray[9,0]
        cCounter += 1
        #print('reading file: {}'.format(statsFile))
        #print('tempArray: \n{}'.format(tempArray))
        
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

        #xticks = {0:'0.1', 1:'0.2', 2:'0.3', 3:'0.4', 4:'0.5', 5:'0.6', 6:'0.7', 7:'0.8', 8:'0.9', 9:'1.0'}
        #yticks = {0:'0.01', 1:'0.02', 2:'0.03', 3:'0.04', 4:'0.05', 5:'0.06', 6:'0.07', 7:'0.08', 8:'0.09', 9:'0.10'}

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
    
    print('=> List created...')
    statsArray = np.zeros([20,10])
    cCounter = 0
    for statsFile in fileList:
        with open('stats/{}_fitness_history.csv'.format(statsFile), 'r') as f:
            tempArray = np.loadtxt(f,delimiter=' ')
        statsArray[:,cCounter] = tempArray[:,0]
        cCounter += 1

    print('=> Data stored...')
    print('=> Generating map...')

    #with open(fitArray, 'r') as dataFile:
        #fitData = np.loadtxt(dataFile, delimiter = ',')
    #-------------------------------#
    #       Generate histogram      #
    #-------------------------------#
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #fig.suptitle('')
    fig.canvas.draw()

    xticks = [ '{0:.01f}'.format(iy) for iy in np.linspace(0.1,1,10)]
    yticks = [ '{0}'.format(int(iy)) for iy in np.linspace(1,20,20)]
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    xlabels = xticks
    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    ylabels = yticks
    ticks = np.linspace(1,20,20)-1
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(xlabels, rotation=-90)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel('Generation')
    ax.set_xlabel('$P_{connection}$')

    ax.set_title('Max Fitness map per generation')   
    mapPlot = ax.imshow(statsArray, origin = 'lower', cmap = 'Greens', interpolation = 'none', vmin = 0, vmax = 1)
    cbar1 = fig.colorbar(mapPlot, ax = ax, orientation='vertical')
    plt.savefig('max_fitness_map_per_gen.eps', format='eps', bbox_inches='tight')

def NetworkBehaviourMap(fileID):
    print('=> Map plot')
    scale = 10
    reps = 100
    nOutputs = 4
    SGF_range = np.linspace(0,1,scale + 1)
    LGF_range = np.linspace(0,1,scale + 1)
    #network_output = np.zeros([len(SGF_range), len(LGF_range), nOutputs]) 
    GF_map = np.zeros([len(SGF_range), len(LGF_range), nOutputs])
    
    #fileID = '20180214_...'.format(folder)
    config_file = 'genomes/{}_config'.format(fileID)
    fileName = 'genomes/{}_best_unique_genomes'.format(fileID)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)
    iGenome = 1

    # Config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # load the winner
    with open(fileName, 'rb') as f:
        genomes = pickle.load(f)#, encoding = 'bytes')

    network = neat.nn.RecurrentNetwork.create(genomes[iGenome], config)
    print('\tGetting & processing data...')
    # Get data and process it...
    for ix in SGF_range:
        for iy in LGF_range:
            network.reset()
            inputs = [ix, iy]
            for _ in range(reps):
                output = network.activate(inputs)
            GF_map[int(iy*scale), int(ix*scale),:] = GenerateStatus(output)
    
    #-------------------------------#
    #       Generate histogram      #
    #-------------------------------#
    print('\tDrawing plot...')
    cMap = ListedColormap(['w', 'g', 'b', 'r'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #fig.suptitle('')
    fig.canvas.draw()

    #xticks = [ '{0:.01f}'.format(iy) for iy in np.linspace(0.1,1,10)]
    #yticks = [ '{0}'.format(int(iy)) for iy in np.linspace(1,20,20)]
    #xlabels = [item.get_text() for item in ax.get_xticklabels()]
    #xlabels = xticks
    #ylabels = [item.get_text() for item in ax.get_yticklabels()]
    #ylabels = yticks
    #ticks = np.linspace(1,20,20)-1
    #ax.set_yticks(ticks)
    #ax.set_xticks(ticks)
    #ax.set_xticklabels(xlabels, rotation=-90)
    #ax.set_yticklabels(ylabels)

    ax.set_ylabel('SGF')
    ax.set_xlabel('LGF')
    ax.set_title('Max Fitness map per generation')   
    
    ax.imshow(GF_map[:,:,0], origin = 'lower', cmap = cMap, interpolation = 'none', vmin = 0, vmax = 3)
    #for 
    ax.scatter((SGF_range)*scale, (LGF_range)*scale, s = 8, marker ='>', c = 'w', label = 'test')

    #cbar1 = fig.colorbar(mapPlot, ax = ax, orientation='vertical')
    plt.savefig('behaviour_map.eps', format='eps', bbox_inches='tight')
    

def GenerateStatus(output):
    #arrow_list = ['^', 'v', '>', '<', 'o']
    #cMap = ListedColormap(['y', 'g', 'r', 'b', 'w'])
    #color_list = ['red', 'blue', 'green', 'white']
    status_data = np.zeros([4])     # [status, polarisation, sgf_amount, lgf_amount]
    
    # Cellular states
    iStatus = output[0]            # Proliferate: Split
    jStatus = output[1]            # Migrate:     Move
    kStatus = output[2]            # Apoptosis:   Die
    # Values for SGF and LGF
    status_data[2] = output[3]
    status_data[3] = output[4]
    # Polarisation
    compass = output[5]

    xThreshold = 0.5
    yThreshold = 0.01

    # ORIENTATION:
    nBoundary = 0.25
    sBoundary = 0.5
    eBoundary = 0.75

    # oriented according to numpy order v>, not usual >^
    #if abs(compass - sBoundary) <= yThreshold:
    #    continue                    # value is already zero anyway
    if compass < sBoundary:
        if compass < nBoundary:
            status_data[1] = 1      # orientation North
        else:
            status_data[1] = 2      # orientation South
    else: 
        if compass < eBoundary:
            status_data[1] = 3      # orientation East
        else:
            status_data[1] = 4      # orientation West
    
    #if iStatus < xThreshold and jStatus < xThreshold and kStatus < xThreshold:
    #    continue                    # value is already zero anyway
    #else:
    for ix in iStatus, jStatus, kStatus:
        if xThreshold < ix:
            xThreshold = ix
    if abs(xThreshold - iStatus) <= yThreshold:
        status_data[0] = 1      # 'Split'
    elif abs(xThreshold - jStatus) <= yThreshold:
        status_data[0] = 2      # 'Move'
    else:
        status_data[0] = 3      # 'Die'
    return status_data
    # Generate state

if __name__ == '__main__':
    dataFile = sys.argv[1]
    #FitnessMapPlot()
    #FitnessperGenMap()
    NetworkBehaviourMap(dataFile)
