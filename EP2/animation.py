import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from test_b import Trelica

FILE = 'inputs/input-c'

class TrelicaAnimada(Trelica):
  def __init__(self, file_name):
    Trelica.__init__(self, file_name)
    self.fig, self.ax = plt.subplots()
    self.X_0 = np.ndarray(2 * self.total_nodes, dtype=float)

    # Posição inicial do primeiro nó nós
    self.X_0[0] = 5
    self.X_0[1] = 40

    node_defined = np.zeros(self.total_nodes, dtype=int)
    node_defined[0] = 1

    for beam in self.beams:
      multiplier = 1
      if (node_defined[beam['nodes'][0]-1] == 0 and node_defined[beam['nodes'][1]-1] == 1):
        node_id, other_node_id = beam['nodes'][0]-1, beam['nodes'][1]-1

      elif (node_defined[beam['nodes'][1]-1] == 0 and node_defined[beam['nodes'][0]-1] == 1):
        node_id, other_node_id = beam['nodes'][1]-1, beam['nodes'][0]-1
        multiplier *= -1

      else:
        continue

      print(node_id+1, other_node_id+1)

      self.X_0[2*node_id] = self.X_0[2*other_node_id] + multiplier * beam['length']*np.cos(np.radians(beam['angle']))
      self.X_0[2*node_id+1] = self.X_0[2*other_node_id+1] + multiplier * beam['length']*np.sin(np.radians(beam['angle']))

      node_defined[node_id] = 1
    # for node_id in range(1, self.total_nodes):

    #   print(f'\n\nPROCURANDO NODE {node_id+1}')

    #   for beam in self.beams:

    #     if node_id == beam['nodes'][0]-1:
    #       other_node_id = beam['nodes'][1]-1

    #     elif node_id == beam['nodes'][1]-1:
    #       other_node_id = beam['nodes'][0]-1

    #     else:
    #       continue

    #     print(other_node_id+1)

    #     if node_defined[other_node_id] == 1:
    #       print('ACHEI')
    #       self.X_0[2*node_id] = self.X_0[2*other_node_id] + beam['length']*np.cos(np.radians(beam['angle']))
    #       self.X_0[2*node_id+1] = self.X_0[2*other_node_id+1] + beam['length']*np.sin(np.radians(beam['angle']))
    #       break

    #   node_defined[node_id] = 1

    self.animated_beams = []

    i = 0
    for beam in self.beams:
      point1 = (self.X_0[2*(beam['nodes'][0]-1)], self.X_0[2*(beam['nodes'][0]-1)+1])
      point2 = (self.X_0[2*(beam['nodes'][1]-1)], self.X_0[2*(beam['nodes'][1]-1)+1])
      ln, = self.ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=2)
      self.animated_beams.append(ln)

      # i += 1
      # if i > 1:
      #   break


    plt.show()


script_dir = os.path.dirname(__file__) # caminho absoluto do script
trelica = TrelicaAnimada(os.path.join(script_dir, FILE))
