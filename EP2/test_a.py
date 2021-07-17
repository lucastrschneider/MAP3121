import numpy as np
from qr_method import QRMethod
from householder import Householder

EPSILON = 1e-6
A1 = np.array([[2, 4, 1, 1],
              [4, 2, 1, 1],
              [1, 1, 1, 2],
              [1, 1, 2, 1]], dtype=float)

def fill_A_test_2(n):
  A2 = np.ndarray((n, n), dtype=float)

  for i in range(n):
    for j in range(n):
      if i <= j:
        A2[i][j] = n - j
      else:
        A2[i][j] = n - i

  return A2

def run():
  print('#############################################')
  print('COMEÇANDO O TESTE A)')
  print('#############################################\n')

  print('\nOpções disponíveis:')
  print(f'1) A =\n{A1}\n')
  print(f'2) A =')
  print('[[ n  n-1 n-2 ...  2   1 ]')
  print(' [n-1 n-1 n-2 ...  2   1 ]')
  print(' [n-2 n-2 n-2 ...  2   1 ]')
  print(' [ :   :   :  ...  :   : ]')
  print(' [ 2   2   2   2   2   1 ]')
  print(' [ 1   1   1   1   1   1 ]]\n')

  try:
    mode = int(input('Número da opção escolhida (entradas inválidas levam à opção 1): '))
    if mode > 2 or mode <= 0:
      raise ValueError
  except:
    mode = 1

  if mode == 1:
    A = A1
  else:
    try:
      n = int(input('Digite o valor de n (padrão é 20): '))
      if n <= 0:
        raise ValueError
    except:
      n = 20

    A = fill_A_test_2(n)

  print('\n')
  print(f'A =\n{A}')


  method_h = Householder(A, print_steps=False)

  result_h = method_h.iterate()

  method_qr = QRMethod(result_h['T']['alfa'], result_h['T']['beta'], result_h['Ht'], spectral=True, print_steps=False)

  eigen_values, eigen_vectors = method_qr.iterate(EPSILON)

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
  
  print(f'\n||I - V_T.V|| = {ortho_error}')

if __name__ == '__main__':
  run()
