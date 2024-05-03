import matplotlib.pyplot as plt
import math
from . import parametersBC
import numpy as np
from . import subroutines
import matplotlib, time
class BCplotter:
    '''
    class for BC activity plotting
    '''
    def __init__(self):
        plt.ion()
        self.egocentricActivity = np.zeros((51, 16))
        self.bvcActivity = np.zeros((51, 16))
        #self.tr1activity = np.zeros((51, 16))
        #self.tr2activity = np.zeros((51, 16))
        #self.tr3activity = np.zeros((51, 16))



        self.angular = np.linspace(0, 2 * math.pi, 51)  # - parametersBC.polarAngularResolution
        self.radial = subroutines.radialScaling()

        R, P = np.meshgrid(self.radial, self.angular)
        self.decodedDir = 0.0
        self.fig = plt.figure(figsize=(8, 6.6))
        self.ax1 = self.fig.add_subplot(211, projection='3d')
        self.ax1.title.set_text("egocentric Layer, decoded Direction: " + str(self.decodedDir))
        self.ax1.axis('off')
        self.ax2 = self.fig.add_subplot(212, projection='3d')
        self.ax2.title.set_text("BVC Layer")
        self.ax2.axis('off')
        '''
        self.ax3 = self.fig.add_subplot(223, projection='3d')
        self.ax3.title.set_text("transformation Layer 6 = " + str(round((180/math.pi) * parametersBC.transformationAngles[5], 2)) + "°")
        self.ax3.axis('off')
        self.ax4 = self.fig.add_subplot(224, projection='3d')
        self.ax4.title.set_text("transformation Layer 16 = " + str(round((180/math.pi) * parametersBC.transformationAngles[15], 2)) + "°")
        self.ax4.axis('off')
        #self.ax5 = self.fig.add_subplot(236, projection='3d')
        #self.ax5.title.set_text("transformation Layer 15")
        '''

        # Express the mesh in the cartesian system.
        self.X, self.Y = R * np.cos(P), R * np.sin(P)
        # Plot the surfaces.
        self.surface1 = self.ax1.plot_surface(self.X, self.Y, self.egocentricActivity, cmap="magma")
        self.surface2 = self.ax2.plot_surface(self.X, self.Y, self.bvcActivity, cmap='magma' )
        #self.surface3 = self.ax3.plot_surface(self.X, self.Y, self.tr1activity, cmap="magma")
        #self.surface4 = self.ax4.plot_surface(self.X, self.Y, self.tr2activity, cmap='magma')
        #self.surface5 = self.ax1.plot_surface(self.X, self.Y, self.tr3activity, cmap="magma", antialiased=False)

        self.ax1.set_zlim(0, 1)
        self.ax2.set_zlim(0, 1)
        #self.ax3.set_zlim(0, 1)
        #self.ax4.set_zlim(0, 1)
        self.ax1.view_init(90, 180)
        self.ax2.view_init(90, 180)
        #self.ax3.view_init(80, 180)
        #self.ax4.view_init(80, 180)
        #self.ax5.view_init(70, 180)
        plt.show()

    def bcPlotting(self, egocentricAct, bvcAct, tr1act, tr2act, tr3act, decodedDir):

        self.surface1.remove()
        self.surface2.remove()
        #self.surface3.remove()
        #self.surface4.remove()
        #self.surface5.remove()
        self.decodedDir = decodedDir
        #self.egocentricActivity = np.random.shuffle(egocentricAct)
        self.egocentricActivity = np.reshape(egocentricAct, (51, 16))
        self.bvcActivity = np.reshape(bvcAct, (51, 16))
        self.tr1activity = np.reshape(tr1act, (51, 16))
        self.tr2activity = np.reshape(tr2act, (51, 16))
        #self.tr3activity = np.reshape(tr3act, (51, 16))

        self.surface1 = self.ax1.plot_surface(self.X, self.Y, self.egocentricActivity, cmap="magma")
        self.ax1.title.set_text("egocentric Layer, decoded Direction: " + str(round((180/math.pi) * self.decodedDir, 1)) + "°")
        self.surface2 = self.ax2.plot_surface(self.X, self.Y, self.bvcActivity, cmap="magma")
        #self.surface3 = self.ax3.plot_surface(self.X, self.Y, self.tr1activity, cmap="magma")
        #self.surface4 = self.ax4.plot_surface(self.X, self.Y, self.tr2activity, cmap="magma")
        #self.surface5 = self.ax5.plot_surface(self.X, self.Y, self.tr3activity, cmap="magma")


        plt.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.01)
