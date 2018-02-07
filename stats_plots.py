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
    createList = 'ls plots/ | egrep 2018020[6-7] > {}'.format(fileName)     # command to generate such file
    sp.call(createList, shell = True)                                       # create the list
    fileList = open(fileName).read().splitlines()                           # store names in a python list for later use
    sp.call('rm {}'.format(fileName), shell = True)                         # remove temporary file
    
    print('=> List created...')
    statsArray = np.zeros([10,10,10])
    cCounter = 0
    for statsFile in fileList:
        with open('stats/{}_fitness_history.csv'.format(statsFile), 'r') as f:
            tempArray = np.loadtxt(f,delimiter=' ')
        
        for ix in range(10):
            statsArray[ix, cCounter//10, cCounter%10] = tempArray[ix,0]
            #print('statsArray[{0},{1},{2}] = {3} =?= tempArray[{0},0] = {4}'.format(ix, cCounter//10, cCounter%10, statsArray[ix, cCounter//10, cCounter%10], tempArray[ix,0]))
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
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #fig.suptitle('')
    xticks = np.linspace(0.1,1,10)
    yticks = np.linspace(0.01,0.1,10)

    ax.set_xlabel('$P_{connection}$')
    ax.set_ylabel('$P_{node}$')
    #ax.set_xticks(xticks)
    #ax.set_yticks(yticks)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
#    cbar1 = fig.colorbar(mapPlot, ax = ax, ticks = [], orientation='vertical')#, shrink=0.75)
    ax.legend(loc = 'best')

    for ix in range(10):
        ax.set_title('Max fitness map for generation #{0}'.format(ix+1))
        mapPlot = ax.imshow(statsArray[ix,:,:], origin = 'lower', interpolation = 'none', vmin = 0, vmax = 1)
        #print('current data array:\n{}'.format(statsArray[ix,:,:]))
        plt.savefig('max_fitness_map_gen_{0}.eps'.format(ix+1), format='eps', bbox_inches='tight')
    #plt.show()

if __name__ == '__main__':
    #dataFile = sys.argv[1]
    FitnessMapPlot()
