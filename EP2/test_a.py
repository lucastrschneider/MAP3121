import os

import numpy as np
from qr_method import QRMethod
from householder import Householder
import utils

EPSILON = 1e-6


def fill_matrix(file_name):
  f = open(file_name, 'r')
  
  #Tamanho da matriz 
  n = int(f.readline())

  #Matriz
  A = np.ndarray((n, n), dtype=float)

  for i in range(n):
      line = f.readline().split()
      for j in range(n):
          A[j][i] = line[j]
  
  f.close()
  
  return A

def run():
  print('#############################################')
  print('COMEÇANDO O TESTE A)')
  print('#############################################\n')
  
  while True:
    try:
      mode = input('Digite o nome do arquivo para o teste: ')

      FILE = 'inputs/' + mode

      script_dir = os.path.dirname(__file__) # caminho absoluto do script
      A = fill_matrix(os.path.join(script_dir, FILE))
      break

    except:
      print('\nNão existe este arquivo!\nExemplo de arquivo existente: input-a\n')

  print(f'A =\n{A}')


  method_h = Householder(A, print_steps=False)

  result_h = method_h.iterate()

  method_qr = QRMethod(result_h['T']['alfa'], result_h['T']['beta'], result_h['Ht'], spectral=True, print_steps=False)

  eigen_values, eigen_vectors = method_qr.iterate(EPSILON)

  print()
  utils.print_eigen_values_vectors(eigen_values, eigen_vectors)

  # Checagem de cada autovalor com seu respectivo autovetor
  eigen_errors = np.ndarray(eigen_values.shape, dtype=float)

  for i in range(eigen_values.shape[0]):
    vi = eigen_vectors[:,[i]]
    Av = np.matmul(A, vi)
    lambdav = eigen_values[i] * vi

    eigen_errors[i]  = np.linalg.norm(Av - lambdav, ord=np.inf)
  
  print(f'\nMaximo de ||A.v - lambda.v|| = {np.max(eigen_errors)}')

  # Checar se a matriz de autovetores e ortogonal
  identity_matrix = np.matmul(eigen_vectors.T,eigen_vectors)
  
  ortho_error  = np.linalg.norm(np.identity(eigen_vectors.shape[0]) - identity_matrix, ord=np.inf)
  
  print(f'\n||I - V_T.V|| = {ortho_error}\n')

if __name__ == '__main__':
  run()
