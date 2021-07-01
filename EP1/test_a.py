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

    print('\nSem deslocamento espectral')
    method = QRMethod(alfa, beta, spectral=False, print_steps=False)
    method.iterate(EPSILON)

    print('\nCom deslocamento espectral')
    spectral_method = QRMethod(alfa, beta, spectral=True, print_steps=False)
    spectral_method.iterate(EPSILON)

    print('\n')


if __name__ == '__main__':
  run()
