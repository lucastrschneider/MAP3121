import os

import numpy as np
from qr_method import QRMethod
from householder import Householder
import utils

FILE = 'inputs/input-c'

EPSILON = 1e-6

class Trelica:
  """
  Classe que representa uma treliça bidimensional

  Args:
      file_name (string): Nome do arquivo de onde os dados da treliça devem ser lidos

  Attributes:
    total_nodes (int): Número de nós da treliça
    lose_nodes (int): Número de nós não engastados da treliça.
    beams_amount (int): Número de barras da treliça.
    rho (float): Densidade das barras da treliça em kg/m³.
    A (float): Área da seção transversal das barras da treliça em m².
    E (float): Módulo de elasticidade das barras da treliça GPa.
    beams (list[dict]): Lista em que cada item é um dicionário que representa uma barra da treliça.
    K (numpy.ndarray): Matriz de rigidez do sistema.
    m (numpy.ndarray): Array que representa a diagonal principal da matriz de massas do sistema.
  """
  def __init__(self, file_name):
    f = open(file_name, 'r')

    # primeira linha
    self.total_nodes, self.lose_nodes, self.beams_amount = \
      [int(x) for x in f.readline().split()]

    # segunda linha
    self.rho, self.A, self.E = \
      [float(x) for x in f.readline().split()]

    # linhas restantes
    self.beams = []
    for i in range(self.beams_amount):
      line = f.readline().split()
      actual_beam = {
        'nodes': (int(line[0]), int(line[1])),
        'angle': float(line[2]),
        'length': float(line[3])
      }
      self.beams.append(actual_beam)

    f.close()

    self.K = np.zeros((2*self.lose_nodes, 2*self.lose_nodes), dtype=float)
    self.m = np.zeros(2*self.lose_nodes, dtype=float)

    for beam in self.beams:
      self._fill_beam(beam)


  def _fill_beam(self, beam):
    """
    Método que recebe uma barra e adiciona a sua contribuição para a matriz de
    rigidez e para a matriz de massas do sistema

    Args:
        beam (dict): [description]
    """
    # Contribuicao para matriz de rigidez
    Kij_ = np.ndarray((2,2), dtype=float)
    C = np.cos(np.radians(beam['angle']))
    S = np.sin(np.radians(beam['angle']))
    i = beam['nodes'][0]
    j = beam['nodes'][1]
    Kij_[0,0] = C**2
    Kij_[0,1] = C*S
    Kij_[1,0] = C*S
    Kij_[1,1] = S**2

    Kij_ *= self.A * self.E * 10**(9) / beam['length']

    if i <= self.lose_nodes:
      self.K[2*i - 2 : 2*i, 2*i - 2 : 2*i] += Kij_

    if i <= self.lose_nodes and j <= self.lose_nodes:
      self.K[2*j - 2 : 2*j, 2*i - 2 : 2*i] -= Kij_
      self.K[2*i - 2 : 2*i, 2*j - 2 : 2*j] -= Kij_

    if j <= self.lose_nodes:
      self.K[2*j - 2 : 2*j, 2*j - 2 : 2*j] += Kij_

    # Contribuicao para matriz de massas
    beam_mass = self.rho * self.A * beam['length']

    if i <= self.lose_nodes:
      self.m[2*i-2] += beam_mass / 2
      self.m[2*i-1] += beam_mass / 2

    if j <= self.lose_nodes:
      self.m[2*j-2] += beam_mass / 2
      self.m[2*j-1] += beam_mass / 2


  def get_Ktil(self):
    """
    Gera a matriz K_til a partir da matriz K do sistema

    Returns:
        numpy.ndarray: Matriz K multiplicada a direita e a esquerda por M^(-1/2)
    """
    Ktil = self.K.copy()
    for i in range(Ktil.shape[0]):
      for j in range(Ktil.shape[1]):
        Ktil[i][j] /= (np.sqrt(self.m[i]) * np.sqrt(self.m[j]))

    return Ktil


def run():
  print('#############################################')
  print('COMEÇANDO O TESTE B)')
  print('#############################################\n')

  script_dir = os.path.dirname(__file__) # caminho absoluto do script
  trelica = Trelica(os.path.join(script_dir, FILE))

  method_h = Householder(trelica.get_Ktil(), print_steps=False)
  result_h = method_h.iterate()

  method_qr = QRMethod(result_h['T']['alfa'], result_h['T']['beta'], result_h['Ht'], spectral=True)
  eigen_values, eigen_vectors = method_qr.iterate(EPSILON)

  eigen_values, eigen_vectors = utils.sort_eigen_values_vectors(eigen_values, eigen_vectors)

  # Imprimindo 5 menores frequencias e seus modos de vibracao
  print('\nMenores frequencias de vibracao da trelica:')
  for i in range(5):
    print(f'\tFrequencia[{i}] = {np.sqrt(eigen_values)[i]} rad/s')

  print('\nModos de vibracao correspondentes:')
  vibration_modes = np.multiply(np.power(trelica.m, -1/2).reshape(trelica.m.shape[0], 1), eigen_vectors[:, 0:5])
  for i in range(5):
    print(f'\tModo[{i}] = {vibration_modes[:,i].T}')
  print()


if __name__ == '__main__':
  run()
