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
    print('\nPlotting map for file {}'.format(fileID))
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
        cbar4 = fig.colorbar(lgfProdMap, ax = ax4, orientation = 'vertical', pad = pad_dist)
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

if __name__ == '__main__':
    location = sys.argv[1]
    dataFile = sys.argv[2]
    #iGenome = int(sys.argv[2])
    #FitnessMapPlot()
    #FitnessperGenMap()
    NetworkBehaviourMap(location, dataFile)#, iGenome)
