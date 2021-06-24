import numpy as np

class QRMethod:
  def __init__(self, a_alfa, a_beta, spectral=True, print_steps=False):
    self.n = a_alfa.shape[0]
    self.spectral = spectral
    self.k = 0
    self.print_steps = print_steps

    assert a_alfa.shape[0] == self.n
    assert a_beta.shape[0] == self.n-1

    self.last_alfa = np.ndarray(self.n, dtype=float)
    self.last_beta = np.ndarray(self.n-1, dtype=float)
    self.actual_alfa = np.ndarray(self.n, dtype=float)
    self.actual_beta = np.ndarray(self.n-1, dtype=float)
    self.eigen_vec = np.identity(self.n, dtype=float)

    np.copyto(self.actual_alfa, a_alfa)
    np.copyto(self.actual_beta, a_beta)


  def _wilkinson_coeficient(self):
    dk = (self.actual_alfa[self.n-2] - self.actual_alfa[self.n-1]) / 2
    return self.actual_alfa[self.n-1] + dk - np.sign(dk)*np.sqrt(dk**2 + (self.actual_beta[self.n-2])**2)

  def _givens_coeficients(self, k):
    alfa = self.actual_alfa[k]
    beta = self.actual_beta[k]

    # Método normal
    # div = np.sqrt(alfa**2 + beta**2)
    # ck = alfa / div
    # sk = -beta / div

    # Método mais estável
    if abs(alfa) > abs(beta):
      tau = -beta/alfa
      ck = 1 / np.sqrt(1 + tau ** 2)
      sk = ck * tau
    else:
      tau = -alfa/beta
      sk = 1 / np.sqrt(1 + tau ** 2)
      ck = sk * tau

    if self.print_steps:
      print(f'{(ck, sk)}')

    return (ck, sk)


  def _get_Qi(self, ci, si):
    Qi = np.identity(2) * ci
    Qi[0,1] = -si
    Qi[1,0] = si

    return Qi


  def _iterate_once(self):
    # Euristica de Wilkinson (coeficiente uk)
    if self.spectral and self.k > 0:
      uk = self._wilkinson_coeficient()
    else:
      uk = 0

    np.copyto(self.last_alfa, self.actual_alfa)
    np.copyto(self.last_beta, self.actual_beta)
    gama = self.actual_beta.copy()

    self.actual_alfa -= uk

    # Coeficientes de Givens (ck e sk)
    ck = np.ndarray(self.n-1, dtype=float)
    sk = np.ndarray(self.n-1, dtype=float)

    if self.print_steps:
      print(f'\n\nIteração {self.k}:')
      print(f'\nDeslocamento espectral: u{self.k} = {uk}')
      print(f'\nCoeficientes de Givens:')

    # Decomposição para calcular Q(k) e R(k)
    for i in range(self.n-1):
      # Calculo dos coefientes da iteração atual
      ck[i], sk[i] = self._givens_coeficients(i)

      alfa_i = self.actual_alfa[i] * ck[i] - self.actual_beta[i] * sk[i]
      gama_i = gama[i] * ck[i] - self.actual_alfa[i+1] * sk[i]
      alfa_j = gama[i] * sk[i] + self.actual_alfa[i+1] * ck[i]

      self.actual_alfa[i] = alfa_i
      gama[i] = gama_i
      self.actual_alfa[i+1] = alfa_j

      if i < self.n-2:
        gama[i+1] = gama[i+1] * ck[i]

    if self.print_steps:
      print(f'\nMatriz R({self.k})')
      print(f'alfa: {self.actual_alfa}')
      print(f'gama: {gama}')

    # Matriz A(k+1)
    for i in range(self.n-1):
      self.actual_alfa[i] = self.actual_alfa[i] * ck[i] - gama[i] * sk[i]
      self.actual_beta[i] = -self.actual_alfa[i+1] * sk[i]
      self.actual_alfa[i+1] = self.actual_alfa[i+1] * ck[i]

    # Correção por deslocamente espectral
    self.actual_alfa += uk

    # Atualiza autovetores V(k+1)
    for i in range(self.n-1): # itera pelas matrizes Q1, Q2...
      self.eigen_vec[:,i:i+2] = np.matmul(self.eigen_vec[:,i:i+2], self._get_Qi(ck[i], sk[i]).T)

    if self.print_steps:
      print(f'\nMatriz A({self.k+1})')
      print(f'alfa: {self.actual_alfa}')
      print(f'beta: {self.actual_beta}')
      print(f'\nAutovetores V({self.k+1})')
      print(f'{self.eigen_vec}')

    self.k += 1


  def iterate(self, epsilon):
    for m in range(self.n-1, 0, -1):
      while abs(self.actual_beta[m-1]) >= epsilon:
        self._iterate_once()
      self.n -= 1

    print(f'\nResultados finais:')
    print(f'Numero de iteracoes: {self.k+1}')
    print(f'Auto valores:\n{self.actual_alfa}')
    print(f'Auto vetores:\n{self.eigen_vec}')



def test1():
  alfa = 4*np.ones(3)
  beta = 3*np.ones(2)

  method = QRMethod(alfa, beta, spectral=True, print_steps=True)
  method.iterate(1e-20)


  eigen_vectors = []
  eigen_vectors.append(np.array([1, np.sqrt(2), 1], dtype=float))
  eigen_vectors.append(np.array([-1, 0, 1], dtype=float))
  eigen_vectors.append(np.array([1, -np.sqrt(2), 1], dtype=float))

  for i in range(len(eigen_vectors)):
    norm = np.linalg.norm(eigen_vectors[i])
    if norm != 0:
      eigen_vectors[i] /= norm

  print(f'\n{np.array(eigen_vectors).T}')


def test2():
  alfa = 4*np.ones(8)
  beta = 3*np.ones(7)

  method = QRMethod(alfa, beta, spectral=True, print_steps=False)
  method.iterate(1e-20)

  eigen_vectors = []
  eigen_vectors.append(np.array([1, 1.87939, 2.53209, 2.87939, 2.87939, 2.53209, 1.87939, 1 ], dtype=float))
  eigen_vectors.append(np.array([-1, -1.53209, -1.3473, -0.532089, 0.532089, 1.3473, 1.53209, 1 ], dtype=float))
  eigen_vectors.append(np.array([1, 1, 0, -1, -1, 0, 1, 1], dtype=float))
  eigen_vectors.append(np.array([-1, -0.347296, 0.879385, 0.652704, -0.652704, -0.879385, 0.347296, 1 ], dtype=float))
  eigen_vectors.append(np.array([1, -0.347296, -0.879385, 0.652704, 0.652704, -0.879385, -0.347296, 1], dtype=float))
  eigen_vectors.append(np.array([-1, 1.87939, -2.53209, 2.87939, -2.87939, 2.35209, -1.87939, 1], dtype=float))
  eigen_vectors.append(np.array([-1, 1, 0, -1, 1, 0, -1, 1], dtype=float))
  eigen_vectors.append(np.array([1, -1.53209, 1.3473, -0.532089, -0.532089, 1.3473, -1.53209, 1], dtype=float))

  for i in range(len(eigen_vectors)):
    norm = np.linalg.norm(eigen_vectors[i])
    if norm != 0:
      eigen_vectors[i] /= norm

  print(f'\n{np.array(eigen_vectors).T}')

if __name__ == '__main__':
  test1()

