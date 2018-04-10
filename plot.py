import sys
import numpy as np
import matplotlib
# WARNING to use in ozzy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# to check for matplotlib backend: >> matplotlib.get_backend()

from matplotlib.colors import ListedColormap, colorConverter
#import networkx as nx
import tools
import matplotlib.colors as colors
import matplotlib.cm as cmx
#import pylab
#from mpl_toolkits.axes_grid1 import ImageGrid

class PlotEnv:
    def __init__(self, fieldSize):
        plt.close()
        
        pad_dist = 0.01
        shrink_pct = 0.78

        #discrete color scheme
        cMap = ListedColormap(['w', 'g', 'b', 'r'])

        self.cellsFigure, (self.cellsSubplot,self.sgfSubplot,self.lgfSubplot) = plt.subplots(1, 3, figsize = (15,5))
        #plt.tick_params(axis='x', left='off', bottom='off', labelleft='off', labelbottom='off')

        self.cellsSubplot.set_aspect('equal')                        # TODO does this work?
        self.sgfSubplot.set_aspect('equal')
        self.lgfSubplot.set_aspect('equal')

        self.cellsFigure.suptitle('Cellular System Visualisation', fontweight='bold')

        self.cellGrid = np.zeros([fieldSize, fieldSize])             # may need a new name, same as in main...
        self.sgfGrid = np.zeros([fieldSize, fieldSize])
        self.lgfGrid = np.zeros([fieldSize, fieldSize])

        self.cellsSubplot.set_title('Cellular Grid')
        self.sgfSubplot.set_title('SGF spatial distribution')
        self.lgfSubplot.set_title('LGF spatial distribution')

        self.cellPlot = self.cellsSubplot.imshow(self.cellGrid, origin = 'lower', cmap = cMap, interpolation = 'none', vmin = 0, vmax = 3)
        cbar1 = self.cellsFigure.colorbar(self.cellPlot, ax = self.cellsSubplot, ticks = [], orientation='horizontal', pad = pad_dist, shrink = shrink_pct)
        #cbar1.ax.set_yticklabels(['dead', 'quiet', 'moving', 'splitting'])
        ##legend
        #cbar = plt.colorbar(cellPlot)

        # hide ticks
        self.cellsSubplot.axes.xaxis.set_ticklabels([])
        self.cellsSubplot.axes.yaxis.set_ticklabels([])
        self.cellsSubplot.axes.get_xaxis().set_visible(False)
        self.cellsSubplot.axes.get_yaxis().set_visible(False)
        #cellPlot.axes.xaxis.set_ticklabels([])
        #cellPlot.axes.yaxis.set_ticklabels([])
        
        cbar1.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(['$empty$','$quiescent$','$migrate$','$proliferate$']):
            cbar1.ax.text((2 * j + 1) / 8.0, .5, lab, ha = 'center', va = 'center')#, rotation=270)
        #cbar1.ax.get_yaxis().labelpad = -25
        #cbar1.ax.get_xaxis().labelpad = -5
        cbar1.ax.set_xlabel('state')#, rotation = 90)

        self.sgfPlot = self.sgfSubplot.imshow(self.sgfGrid, origin = 'lower', cmap = 'Reds', interpolation = 'none', vmin = 0, vmax = 1)
        cbar2 = self.cellsFigure.colorbar(self.sgfPlot, ax = self.sgfSubplot, orientation = 'horizontal', pad = pad_dist, shrink = shrink_pct)

        # hide ticks
        self.sgfPlot.axes.xaxis.set_ticklabels([])
        self.sgfPlot.axes.yaxis.set_ticklabels([])
        self.sgfPlot.axes.get_xaxis().set_visible(False)
        self.sgfPlot.axes.get_yaxis().set_visible(False)

        self.lgfPlot = self.lgfSubplot.imshow(self.lgfGrid, origin = 'lower', cmap = 'Blues', interpolation = 'none', vmin = 0, vmax = 1)
        cbar3 = self.cellsFigure.colorbar(self.lgfPlot, ax = self.lgfSubplot, orientation = 'horizontal', pad = pad_dist, shrink = shrink_pct)

        # hide ticks
        self.lgfPlot.axes.xaxis.set_ticklabels([])
        self.lgfPlot.axes.yaxis.set_ticklabels([])
        self.lgfPlot.axes.get_xaxis().set_visible(False)
        self.lgfPlot.axes.get_yaxis().set_visible(False)

        plt.ion()
        #plt.pause(0.001)
        self.cellsFigure.canvas.draw()
        plt.ioff()
        plt.subplots_adjust(wspace=-0.1)
   
    #def CellGridPlot(self, env, nLattice, tStep, location, iGenome):
        #cell_data = cellGrid         # slice the grid to get the layer with the cell positions
        #sgf_data = chemGrid[:,:,0]          # slice the grid to get the layer with the SGF profile
        #lgf_data = chemGrid[:,:,1]          # slice the grid to get the layer with the LGF profile

        #UpdatePlot( self, cell_data, sgf_data, lgf_data, tStep, location, iGenome)
    
    def UpdatePlot( self, env, tStep, location, iGenome):
        #
        self.cellPlot.set_data(env.cellGrid)
        self.sgfPlot.set_data(env.chemGrid[:,:,0])
        self.lgfPlot.set_data(env.chemGrid[:,:,1])
        #
        self.cellsSubplot.draw_artist(self.cellsSubplot.patch)
        self.cellsSubplot.draw_artist(self.cellPlot)
        self.sgfSubplot.draw_artist(self.sgfPlot)
        self.lgfSubplot.draw_artist(self.lgfPlot)

        plt.savefig('{0}/best_unique_genome_{1}/cell_system-{2:03d}.png'.format(location, iGenome+1, tStep,), format='png', bbox_inches='tight')
# UpdatePlot

class DC_PlotEnv:
    def __init__(self, fieldSize):
        plt.close()
        
        pad_dist = 0.01
        shrink_pct = 0.78
        text_font_size = 8

        #discrete color scheme
        cMap = ListedColormap(['w', 'g', 'b', 'r', 'k'])
        #DC_cMap = ListedColormap(['k', 'w'])
        
        color1 = colorConverter.to_rgba('white',alpha=0.0)
        color2 = colorConverter.to_rgba('black',alpha=0.8)
        DC_cMap = colors.LinearSegmentedColormap.from_list('DC_cMap',[color2,color1],256)
        
        self.cellsFigure, (self.cellsSubplot,self.sgfSubplot,self.lgfSubplot) = plt.subplots(1, 3, figsize = (15,5))
        #plt.tick_params(axis='x', left='off', bottom='off', labelleft='off', labelbottom='off')

        self.cellsSubplot.set_aspect('equal')                        # TODO does this work?
        self.sgfSubplot.set_aspect('equal')
        self.lgfSubplot.set_aspect('equal')

        self.cellsFigure.suptitle('Cellular System Visualisation', fontweight='bold')

        self.cellGrid = np.zeros([fieldSize, fieldSize])             # may need a new name, same as in main...
        self.DC_cellGrid = np.zeros([fieldSize, fieldSize])             # may need a new name, same as in main...
        self.sgfGrid = np.zeros([fieldSize, fieldSize])
        self.lgfGrid = np.zeros([fieldSize, fieldSize])

        self.cellsSubplot.set_title('Cellular Grid')
        self.sgfSubplot.set_title('SGF spatial distribution')
        self.lgfSubplot.set_title('LGF spatial distribution')

        self.cellPlot = self.cellsSubplot.imshow(self.cellGrid, origin = 'lower', cmap = cMap, interpolation = 'none', vmin = 0, vmax = 4)
        self.DC_cellPlot = self.cellsSubplot.imshow(self.DC_cellGrid, origin = 'lower', cmap = DC_cMap, interpolation = 'none', vmin = -1, vmax = 0)#, alpha = 0.4)
        cbar1 = self.cellsFigure.colorbar(self.cellPlot, ax = self.cellsSubplot, ticks = [], orientation='horizontal', pad = pad_dist, shrink = shrink_pct)
        #cbar1.ax.set_yticklabels(['dead', 'quiet', 'moving', 'splitting'])
        ##legend
        #cbar = plt.colorbar(cellPlot)

        # hide ticks
        self.cellsSubplot.axes.xaxis.set_ticklabels([])
        self.cellsSubplot.axes.yaxis.set_ticklabels([])
        self.cellsSubplot.axes.get_xaxis().set_visible(False)
        self.cellsSubplot.axes.get_yaxis().set_visible(False)
        #cellPlot.axes.xaxis.set_ticklabels([])
        #cellPlot.axes.yaxis.set_ticklabels([])
        
        cbar1.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(['$empty$','$quiescent$','$migrate$','$proliferate$', '$death cell$']):
            cbar1.ax.text((2 * j + 1) / 10.0, .5, lab, ha = 'center', va = 'center', fontsize=text_font_size, color = 'y', fontweight='bold')#, rotation=270)
        #cbar1.ax.get_yaxis().labelpad = -25
        #cbar1.ax.get_xaxis().labelpad = -5
        cbar1.ax.set_xlabel('state')#, rotation = 90)

        self.sgfPlot = self.sgfSubplot.imshow(self.sgfGrid, origin = 'lower', cmap = 'Reds', interpolation = 'none', vmin = 0, vmax = 1)
        cbar2 = self.cellsFigure.colorbar(self.sgfPlot, ax = self.sgfSubplot, orientation = 'horizontal', pad = pad_dist, shrink = shrink_pct)

        # hide ticks
        self.sgfPlot.axes.xaxis.set_ticklabels([])
        self.sgfPlot.axes.yaxis.set_ticklabels([])
        self.sgfPlot.axes.get_xaxis().set_visible(False)
        self.sgfPlot.axes.get_yaxis().set_visible(False)

        self.lgfPlot = self.lgfSubplot.imshow(self.lgfGrid, origin = 'lower', cmap = 'Blues', interpolation = 'none', vmin = 0, vmax = 1)
        cbar3 = self.cellsFigure.colorbar(self.lgfPlot, ax = self.lgfSubplot, orientation = 'horizontal', pad = pad_dist, shrink = shrink_pct)

        # hide ticks
        self.lgfPlot.axes.xaxis.set_ticklabels([])
        self.lgfPlot.axes.yaxis.set_ticklabels([])
        self.lgfPlot.axes.get_xaxis().set_visible(False)
        self.lgfPlot.axes.get_yaxis().set_visible(False)

        plt.ion()
        #plt.pause(0.001)
        self.cellsFigure.canvas.draw()
        plt.ioff()
        plt.subplots_adjust(wspace=-0.1)
       
    def UpdatePlot( self, env, tStep, location, iGenome):
        #
        self.cellPlot.set_data(env.cellGrid)
        self.DC_cellPlot.set_data(env.deathCellGrid)
        self.sgfPlot.set_data(env.chemGrid[:,:,0])
        self.lgfPlot.set_data(env.chemGrid[:,:,1])
        #
        self.cellsSubplot.draw_artist(self.cellsSubplot.patch)
        self.cellsSubplot.draw_artist(self.cellPlot)
        self.cellsSubplot.draw_artist(self.DC_cellPlot)
        self.sgfSubplot.draw_artist(self.sgfPlot)
        self.lgfSubplot.draw_artist(self.lgfPlot)

        plt.savefig('{0}/best_unique_genome_{1}/cell_system-{2:03d}.png'.format(location, iGenome+1, tStep,), format='png', bbox_inches='tight')
    # UpdatePlot
    
def CellsGridFigure(fieldSize, mode):
    # mode = True: cell_system as fitness function
    # mode = False: cell_system as display system
    plt.close()
    
    pad_dist = 0.01
    shrink_pct = 0.78

    #discrete color scheme
    cMap = ListedColormap(['w', 'g', 'b', 'r'])

    cellsFigure, (cellsSubplot,sgfSubplot,lgfSubplot) = plt.subplots(1, 3, figsize = (15,5))
    #plt.tick_params(axis='x', left='off', bottom='off', labelleft='off', labelbottom='off')

    cellsSubplot.set_aspect('equal')                        # TODO does this work?
    sgfSubplot.set_aspect('equal')
    lgfSubplot.set_aspect('equal')

    cellsFigure.suptitle('Cellular System Visualisation', fontweight='bold')

    cellGrid = np.zeros([fieldSize, fieldSize])             # may need a new name, same as in main...
    sgfGrid = np.zeros([fieldSize, fieldSize])
    lgfGrid = np.zeros([fieldSize, fieldSize])

    cellsSubplot.set_title('Cellular Grid')

#        cellsSubplot.axis('off')

    sgfSubplot.set_title('SGF spatial distribution')
#        sgfSubplot.axis('off')

    lgfSubplot.set_title('LGF spatial distribution')
#       lgfSubplot.axis('off')

    cellPlot = cellsSubplot.imshow(cellGrid, origin = 'lower', cmap = cMap, interpolation = 'none', vmin = 0, vmax = 3)
    cbar1 = cellsFigure.colorbar(cellPlot, ax = cellsSubplot, ticks = [], orientation='horizontal', pad = pad_dist, shrink = shrink_pct)
    #cbar1.ax.set_yticklabels(['dead', 'quiet', 'moving', 'splitting'])
    ##legend
    #cbar = plt.colorbar(cellPlot)

    # hide ticks
    cellsSubplot.axes.xaxis.set_ticklabels([])
    cellsSubplot.axes.yaxis.set_ticklabels([])
    cellsSubplot.axes.get_xaxis().set_visible(False)
    cellsSubplot.axes.get_yaxis().set_visible(False)
    #cellPlot.axes.xaxis.set_ticklabels([])
    #cellPlot.axes.yaxis.set_ticklabels([])
    
    cbar1.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['$empty$','$quiescent$','$migrate$','$proliferate$']):
        cbar1.ax.text((2 * j + 1) / 8.0, .5, lab, ha = 'center', va = 'center')#, rotation=270)
    #cbar1.ax.get_yaxis().labelpad = -25
    #cbar1.ax.get_xaxis().labelpad = -5
    cbar1.ax.set_xlabel('state')#, rotation = 90)

    sgfPlot = sgfSubplot.imshow(sgfGrid, origin = 'lower', cmap = 'Reds', interpolation = 'none', vmin = 0, vmax = 1)
    cbar2 = cellsFigure.colorbar(sgfPlot, ax = sgfSubplot, orientation = 'horizontal', pad = pad_dist, shrink = shrink_pct)

    # hide ticks
    sgfPlot.axes.xaxis.set_ticklabels([])
    sgfPlot.axes.yaxis.set_ticklabels([])
    sgfPlot.axes.get_xaxis().set_visible(False)
    sgfPlot.axes.get_yaxis().set_visible(False)

    lgfPlot = lgfSubplot.imshow(lgfGrid, origin = 'lower', cmap = 'Blues', interpolation = 'none', vmin = 0, vmax = 1)
    cbar3 = cellsFigure.colorbar(lgfPlot, ax = lgfSubplot, orientation = 'horizontal', pad = pad_dist, shrink = shrink_pct)

    # hide ticks
    lgfPlot.axes.xaxis.set_ticklabels([])
    lgfPlot.axes.yaxis.set_ticklabels([])
    lgfPlot.axes.get_xaxis().set_visible(False)
    lgfPlot.axes.get_yaxis().set_visible(False)

    #if mode == False:
    #    plt.show(block = False)

    plt.ion()
    #plt.pause(0.001)
    cellsFigure.canvas.draw()
    plt.ioff()
    
    plt.subplots_adjust(wspace=-0.1)
    #cellsFigure.tight_layout(w_pad=-5)

    # function returns the figure, subplots and plots
    return cellsFigure, cellsSubplot, sgfSubplot, lgfSubplot, cellPlot, sgfPlot, lgfPlot
# CellsGridFigure

def CellGridPlot(cellGrid,
                chemGrid,
                nLattice,
                cellsFigure,
                cellsSubplot,
                sgfSubplot,
                lgfSubplot,
                cellPlot,
                sgfPlot,
                lgfPlot,
                tStep,
                mode,
                location,
                iGenome):

    cell_data = cellGrid         # slice the grid to get the layer with the cell positions
    sgf_data = chemGrid[:,:,0]          # slice the grid to get the layer with the SGF profile
    lgf_data = chemGrid[:,:,1]          # slice the grid to get the layer with the LGF profile

    #Environment.UpdatePlot( cellsFigure,
                            #cellsSubplot,
                            #sgfSubplot,
                            #lgfSubplot,
                            #cellPlot,
                            #sgfPlot,
                            #lgfPlot,
                            #cell_data,
                            #sgf_data,
                            #lgf_data,
                            #tStep,
                            #mode)
    UpdatePlot( cellsFigure,
                cellsSubplot,
                sgfSubplot,
                lgfSubplot,
                cellPlot,
                sgfPlot,
                lgfPlot,
                cell_data,
                sgf_data,
                lgf_data,
                tStep,
                mode,
                location,
                iGenome)

def UpdatePlot( cellsFigure,
                cellsSubplot,
                sgfSubplot,
                lgfSubplot,
                cellPlot,
                sgfPlot,
                lgfPlot,
                cell_data,
                sgf_data,
                lgf_data,
                tStep,
                mode,
                location,
                iGenome):
    #
    cellPlot.set_data(cell_data)
    sgfPlot.set_data(sgf_data)
    lgfPlot.set_data(lgf_data)
    #
    cellsSubplot.draw_artist(cellsSubplot.patch)
    cellsSubplot.draw_artist(cellPlot)
    sgfSubplot.draw_artist(sgfPlot)
    lgfSubplot.draw_artist(lgfPlot)
    
    #
    #cellsFigure.canvas.update()
    #cellsFigure.canvas.flush_events()
    
    if mode == False:
        plt.savefig('{0}/best_unique_genome_{1}/cell_system-{2:03d}.png'.format(location, iGenome+1, tStep,), format='png', bbox_inches='tight')
# UpdatePlot

def FitvsNnodesPlot(statsFile):
    varSpace = 5
    nGen = 10
    with open('stats/{}'.format(statsFile), 'r') as dataFile:
        statsArray = np.loadtxt(dataFile,delimiter=',')
        statsArray = statsArray.reshape(varSpace, nGen, 2)

    #dataFigure, (dataSubplot) = plt.subplots(1, 1, figsize = (15,5))

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #fig.suptitle('')
    #genList = np.arange(1, nGen+1) #np.arange(nGen)
    nodeList = [8, 10, 15, 20, 25]
    ax.set_xlabel('number of nodes')
    ax.set_ylabel('fitness of best individual')
    ax.set_xticks(nodeList)
    ax.set_ylim([0,1])
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    
    ax.set_title('Max fitness vs number of nodes')   
    ax.plot(nodeList, statsArray[:,0,0], color='b', linestyle='dashed', label = '1st gen')
    #ax.plot(nodeList, statsArray[:,1,0], color='0.9', linestyle='dashed', label = '2nd gen')
    #ax.plot(nodeList, statsArray[:,2,0], color='0.8', linestyle='dashed', label = '3rd gen')
    #ax.plot(nodeList, statsArray[:,3,0], color='0.7', linestyle='dashed', label = '4th gen')
    #ax.plot(nodeList, statsArray[:,4,0], color='0.6', linestyle='dashed', label = '5th gen')
    #ax.plot(nodeList, statsArray[:,5,0], color='0.5', linestyle='dashed', label = '6th gen')
    #ax.plot(nodeList, statsArray[:,6,0], color='0.4', linestyle='dashed', label = '7th gen')
    #ax.plot(nodeList, statsArray[:,7,0], color='0.3', linestyle='dashed', label = '8th gen')
    #ax.plot(nodeList, statsArray[:,8,0], color='0.2', linestyle='dashed', label = '9th gen')
    ax.plot(nodeList, statsArray[:,9,0], color='g', linestyle='dashed', label = '10th gen')
    ax.legend(loc = 'best')
    #plt.savefig(''.format())
    plt.show()

def FitvsnGenPlot(statsFile):
    varSpace = 5
    nGen = 10
    with open('stats/{}'.format(statsFile), 'r') as dataFile:
        statsArray = np.loadtxt(dataFile,delimiter=',')
        statsArray = statsArray.reshape(varSpace, nGen, 2)

    #dataFigure, (dataSubplot) = plt.subplots(1, 1, figsize = (15,5))

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #fig.suptitle('')
    genList = np.arange(1, nGen+1) #np.arange(nGen)
    # xticks = 
    ax.set_xlabel('number of generations')
    ax.set_ylabel('fitness')
    ax.set_xticks(genList)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    
    ax.set_title('change in fitness over generations')   
    ax.plot(genList, statsArray[0,:,1], 'r--', label = 'average fitness')
    ax.plot(genList, statsArray[1,:,1], 'b--', label = '20 nodes')
    ax.plot(genList, statsArray[2,:,1], 'k--', label = '15 nodes')
    ax.plot(genList, statsArray[3,:,1], 'y--', label = '10 nodes')
    ax.plot(genList, statsArray[4,:,1], 'g--', label = '8 nodes')

    ax.scatter(genList, statsArray[0,:,0], marker =',', c = 'r', label = 'max fitness')
    ax.scatter(genList, statsArray[1,:,0], marker ='o', c = 'b', label = 'max fitness 20')
    ax.scatter(genList, statsArray[2,:,0], marker ='*', c = 'k', label = 'max fitness 15')
    ax.scatter(genList, statsArray[3,:,0], marker ='+', c = 'y', label = 'max fitness 10')
    ax.scatter(genList, statsArray[4,:,0], marker ='_', c = 'g', label = 'max fitness 8')
    ax.legend(loc = 'best')
    #plt.savefig(''.format())
    plt.show()
    
def benchPlot(benchFile):
    varSpace = 5
    #nGen = 10
    with open('benchmarks/{}'.format(benchFile), 'r') as dataFile:
        benchArray = np.loadtxt(dataFile,delimiter=',')
#        benchArray = benchArray.reshape(2, varSpace)

    #dataFigure, (dataSubplot) = plt.subplots(1, 1, figsize = (15,5))

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #fig.suptitle('')
    runList = np.arange(0,varSpace)
    x_ticks = np.arange(0, 26, 5)#, nGen+1) #np.arange(nGen)
    # xticks = 
    width = 0.25
    ax.set_xlabel('number of generations')
    ax.set_ylabel('time (s)')
    ax.set_xticks(runList + (width/2.))
    ax.set_xticklabels(x_ticks)
    #ax.set_xscale('log')
    #ax.set_yscale('log') 
    
    ax.set_title('benchmarks for each generation number')   
    #ax.hist(genList, benchArray[0,:], 'r--', label='8 nodes')
    #ax.hist(genList, benchArray[0,:], 'r--', label='8 nodes')
    ax.bar(runList, benchArray[runList, 1], width, align = 'center', alpha = 0.85, color = 'g', label = 'avg time per generation')
    ax.bar(runList + width, benchArray[runList, 0], width, align = 'center', alpha = 0.85, color = 'r', label = 'total time per generation')
    #ax.plot(genList, benchArray[1,:], 'b-', label='25 nodes')
    #ax.scatter(genList, benchArray[0,:], label='max fitness')
    #ax.scatter(genList, benchArray[1,:], label='max fitness')
    # ax.plot(syntheticForest,normRank,label='Synthetic data')
    ax.legend(loc = 'best')
    #plt.savefig(''.format())
    plt.show()
    
def ShowNetwork(dataFile):
    csvFile = 'populations/{}.csv'.format(dataFile)
    ind = 460
    nNodes = 25
    wMatrix = tools.GetrNN(csvFile, ind)
    G = nx.from_numpy_matrix(wMatrix)
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle('Recurrent Neural Network')
    #ax.set_xlabel('Node degree k')
    #ax.set_ylabel('P(k)')
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #ax.set_title('Degree distribution plot')   
    ax.set_title('subtitle')   
    #ax.plot(expPk,label='Exp pred')
    #ax.plot(theoPk,label='Theo pred',linewidth=3)
    #ax.legend(loc='best')
    pos = nx.spring_layout(G)
    nx.draw(G, ax=ax)
    #nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color = values, node_size = 500)
    #nx.draw_networkx_labels(G, pos)
    #nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
    #draw_networkx_edges(G, pos, edgelist=None, width=1.0, style='solid', alpha = 1.0, edge_cmap = plt.cm.Blues, edge_vmin = -1, edge_vmax = 1, ax=ax, arrows=True, label='label!')
    
    #edgesArray = list(range(nx.number_of_edges(G)))
    #print('{}'.format(edgesArray))
    # These values could be seen as dummy edge weights

    #jet = cm = plt.get_cmap('jet') 
    #cNorm  = colors.Normalize(vmin = -1, vmax = 1)
    #scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = jet)
    #colorList = []

    #for i in edgesArray:
    #    colorVal = scalarMap.to_rgba(edgesArray[i])
    #    colorList.append(colorVal)
    #    nx.draw_spring(G, ax = ax, edge_color = colorList)
    #print('{}'.format(colorList))
    
    #ax.legend(loc='best')
    #plt.savefig('network.eps', format='eps', bbox_inches='tight')
    plt.show()
    
def MapPlot(mapFile):
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fileName = 'maps/{}.csv'.format(mapFile)
    with open(fileName, 'r') as dataFile:
        networkMap = np.loadtxt(dataFile, delimiter = ',')
    #discrete color scheme
    cMap = ListedColormap(['y', 'g', 'r', 'b', 'w'])
    mapPlot = ax.imshow(networkMap, origin = 'lower', cmap = cMap, interpolation = 'none', vmin = 0, vmax = 4)
    cbar1 = fig.colorbar(mapPlot, ax = ax, ticks = [], orientation='horizontal')#, shrink=0.75)
    plt.show()

if __name__ == '__main__':
    dataFile = sys.argv[1]
    # FitvsNnodesPlot(dataFile)
    # FitvsnGenPlot(dataFile)
    # benchPlot(dataFile)
    # ShowNetwork(dataFile)
    # Histogram(dataFile)
    MapPlot(dataFile)
