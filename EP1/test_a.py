import numpy as np
from qr_method import QRMethod

EPSILON = 1e-6
n_list = [4, 8, 16, 32]

def main():
  print('#########################################')
  print('COMEÇANDO O TESTE A)')
  print('#########################################\n')

  for n in n_list:
    print('#########################################')
    print(f'N = {n}')
    print('#########################################')

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
  main()
