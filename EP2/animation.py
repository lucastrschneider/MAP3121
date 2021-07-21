import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import utils

from test_b import Trelica

from householder import Householder
from qr_method import QRMethod

FILE = 'inputs/input-c'
EPSILON = 1e-6
SCALE = 1e2

class TrelicaAnimada(Trelica):
  def __init__(self, file_name, modes_weights):
    Trelica.__init__(self, file_name)
    self.modes_amount = len(modes_weights)
    self.modes_weights = np.array(modes_weights, dtype=float)

    self.fig, self.ax = plt.subplots()
    self.X_0 = np.ndarray(2 * self.total_nodes, dtype=float)

    # Posição inicial do primeiro nó nós
    self.X_0[0] = 5
    self.X_0[1] = 40

    node_defined = np.zeros(self.total_nodes, dtype=int)
    node_defined[0] = 1

    # Calcula a posição de todos os nós que de alguma forma estão conectados ao nó inicial
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


    # Cria os elementos gráficos que representam as barras
    self.animated_beams = []

    for beam in self.beams:
      point1 = (self.X_0[2*(beam['nodes'][0]-1)], self.X_0[2*(beam['nodes'][0]-1)+1])
      point2 = (self.X_0[2*(beam['nodes'][1]-1)], self.X_0[2*(beam['nodes'][1]-1)+1])
      ln, = self.ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=2)
      self.animated_beams.append(ln)      

    # Calcula as frequencias e modos de vibração dos nós que estão soltos
    method_h = Householder(Trelica.get_Ktil(self), print_steps=False)
    result_h = method_h.iterate()

    method_qr = QRMethod(result_h['T']['alfa'], result_h['T']['beta'], result_h['Ht'], spectral=True)
    eigen_values, eigen_vectors = method_qr.iterate(EPSILON)

    eigen_values, eigen_vectors = utils.sort_eigen_values_vectors(eigen_values, eigen_vectors)
    
    # Pega as 5 menores frequências e seus respectivos modos de vibração  
    self.vibration_frequencies = np.sqrt(eigen_values[:self.modes_amount])
    self.vibration_modes = np.multiply(np.power(self.m, -1/2).reshape(self.m.shape[0], 1), eigen_vectors[:, :self.modes_amount])

    print(self.vibration_frequencies)
    print(self.vibration_modes)

    self.anim = animation.FuncAnimation(self.fig, self._update, frames=np.linspace(0, 1, 200),
                                        interval=5)


  def _update(self, time):
    # Calcula a posição atual de cada vértice
    X_t = self.X_0.copy()

    for i in range(self.lose_nodes):
      for j in range(self.modes_amount):
        X_t[2*i] += self.modes_weights[j] * self.vibration_modes[2*i, j] * np.cos(self.vibration_frequencies[j] * time)
        X_t[2*i+1] += self.modes_weights[j] * self.vibration_modes[2*i+1, j] * np.cos(self.vibration_frequencies[j] * time)

    
    # Atualiza as barras da treliça
    for i, beam in enumerate(self.beams):
      point1 = (X_t[2*(beam['nodes'][0]-1)], X_t[2*(beam['nodes'][0]-1)+1])
      point2 = (X_t[2*(beam['nodes'][1]-1)], X_t[2*(beam['nodes'][1]-1)+1])
      self.animated_beams[i].set_data([point1[0], point2[0]], [point1[1], point2[1]])

    return self.animated_beams[0],


  def show(self):
    plt.show()

  
  def save(self, name):
    self.anim.save(name)


def run():
  script_dir = os.path.dirname(__file__) # caminho absoluto do script
  trelica = TrelicaAnimada(os.path.join(script_dir, FILE), [SCALE, 0, 0, 0, 0])
  trelica.save('trelica_modo_0.mp4')
  # trelica.show()

if __name__ == '__main__':
  run()