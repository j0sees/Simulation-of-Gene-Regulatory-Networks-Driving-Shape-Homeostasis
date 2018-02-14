import sys
import numpy as np
import matplotlib
# WARNING to use in ozzy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# to check for matplotlib backend: >> matplotlib.get_backend()
import subprocess as sp

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
    
def Histogram():
    fileName = 'filesList'                                                  # name of the temporary file with names
    createList = 'ls plots/20180209_pConnection_20gen_run > {}'.format(fileName)     # command to generate such file
    sp.call(createList, shell = True)                                       # create the list
    fileList = open(fileName).read().splitlines()                           # store names in a python list for later use
    sp.call('rm {}'.format(fileName), shell = True)                         # remove temporary file
    
    print('=> List created...')
    statsArray = np.zeros([20,20])
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

    xticks = [ '{0:.02f}'.format(iy) for iy in np.linspace(0.05,1,20)]
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
    cbar1 = fig.colorbar(mapPlot, ax = ax, orientation='vertical')#, shrink=0.75)
    plt.savefig('max_fitness_map_per_gen.eps', format='eps', bbox_inches='tight')

if __name__ == '__main__':
    #dataFile = sys.argv[1]
    #FitnessMapPlot()
    Histogram()
