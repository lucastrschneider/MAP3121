import numpy as np
from qr_method import QRMethod

EPSILON = 1e-6
COMPLETE_N_LIST = [4, 8, 16, 32]

def run():
  print('#############################################')
  print('COMEÇANDO O TESTE A)')
  print('#############################################\n')

  try:
    defined_n = int(input('Digite enter para realizar o teste completo (n = 4, 8, 16 e 32) ou\nespecifique um valor de n: '))
    if defined_n > 0:
      n_list = [defined_n]
    else:
      raise ValueError
  except:
    print('\nO valor fornecido não pode ser convertido para um inteiro positivo.\nO teste completo será executado')
    n_list = COMPLETE_N_LIST


  print('\n')
  for n in n_list:
    print('#############################################')
    print(f'n = {n}')
    print('#############################################')

    alfa = 2*np.ones(n)
    beta = -1*np.ones(n-1)

    exp_eigen_values = np.ndarray(n, dtype=float)
    exp_eigen_vectors = np.ndarray((n,n), dtype=float)

    print('\nValores esperados\n')
    for j in range(n):
      exp_eigen_values[n-j-1] = 2 * (1 - np.cos((j+1) * np.pi / (n+1)))
      for i in range(n):
        exp_eigen_vectors[i][n-j-1] = np.sin((j+1) * (i+1) * np.pi / (n+1))

      exp_eigen_vectors[:, n-j-1] = exp_eigen_vectors[:, n-j-1] / np.linalg.norm(exp_eigen_vectors[:, n-j-1])

    print(f'Auto valores:\n{exp_eigen_values}')
    print(f'Auto vetores:\n{exp_eigen_vectors}')

    print('\nSem deslocamento espectral')
    method = QRMethod(alfa, beta, spectral=False, print_steps=False)
    qr_eigen_values, qr_eigen_vectors = method.iterate(EPSILON)

    # TODO calcular erro

    print('\nCom deslocamento espectral')
    spectral_method = QRMethod(alfa, beta, spectral=True, print_steps=False)
    qr_spectral_eigen_values, qr_spectral_eigen_vectors = spectral_method.iterate(EPSILON)

    # TODO calcular erro
    
    print('\n')


if __name__ == '__main__':
  run()
