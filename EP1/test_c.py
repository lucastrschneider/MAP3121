import numpy as np
import matplotlib.pyplot as plt
from qr_method import QRMethod

N = 10
MASSA = 2
EPSILON = 1e-6
TOTAL_TIME = 10 #s
dt = 0.025 #s

def plot(tempo, posicao, subplot=False):
  if subplot == False:
    for i in range(posicao.shape[0]):
      plt.plot(tempo, posicao[i,:], label=f'x{i+1}')
    plt.legend(loc=1)
    plt.title('Deslocamento das massas no tempo')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Posição (m)')

  else:
    fig, axs = plt.subplots(5, 2,  sharex=True, sharey=True)
    for i in range(5):
      for j in range(2):
        axs[i, j].plot(tempo, posicao[2*i + j,:])
        axs[i, j].set_title(f'x{2*i+j+1}(t)')

    for ax in axs.flat:
        ax.set(xlabel='Tempo (s)', ylabel='Posição (m)')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
       ax.label_outer()


def run():
  print('#############################################')
  print('COMEÇANDO O TESTE C)')
  print('#############################################\n')

  k = np.array([40 + 2 * (-1)**(i+1) for i in range(N+1)])

  alfa = (k[0:N] + k[1:N+1]) / MASSA
  beta = -k[1:N] / MASSA

  print('Aplicando o método QR na matriz A do sistema')

  method = QRMethod(alfa, beta, spectral=True, print_steps=False)
  eigen_values, eigen_vectors = method.iterate(EPSILON)

  T = np.linspace(0, TOTAL_TIME, int(TOTAL_TIME / dt + 1))
  X = np.ndarray((N, T.shape[0]), dtype=float)

  X0 = np.zeros(N)

  print('\nOpções disponíveis:')
  print('\t1 - X(0) = -2, -3, -1, -3, -1, -2, -3, -1, -3, -1')
  print('\t2 - X(0) = 1, 10, -4, 3, -2, 1, 10, -4, 3, -2')
  print('\t3 - X(0) correspondente ao modo de maior frequência')
  print('\t4 - Digite seu próprio valor de X(0)\n')

  try:
    mode = int(input('Número da opção escolhida (entradas inválidas levam à opção 4): '))
  except:
    mode = None

  if mode == 1:
    X0 = np.array([-2, -3, -1, -3, -1, -2, -3, -1, -3, -1], dtype=float)

  elif mode == 2:
    X0 = np.array([1, 10, -4, 3, -2, 1, 10, -4, 3, -2], dtype=float)

  elif mode == 3:
    i_max = 0
    for i in range(1, len(eigen_values)):
      if eigen_values[i] > eigen_values[i_max]:
        i_max = i

    X0 = eigen_vectors[:, i_max]

  else:
    print('Digite os valores iniciais para cada massa:')
    for i in range(N):
      X0[i] = float(input(f'\tx{i+1}(0) = '))

  print(f'X(0) = {X0}')

  Y0 = np.matmul(eigen_vectors.T, X0)
  X[:, 0] = X0

  for i in range(1, len(T)):
    t = T[i]
    Y = np.multiply(Y0, np.cos(np.sqrt(eigen_values)*t))
    X[:, i] = np.matmul(eigen_vectors, Y)

  plot(T, X)
  plot(T, X, subplot = True)

  plt.show()


if __name__ == '__main__':
  run()
