import numpy as np

class QRMethod:
  def __init__(self, a_alfa, a_beta, spectral=True, print_steps=False, print_matrices=False):
    self.n = a_alfa.shape[0]
    self.spectral = spectral
    self.k = 0
    self.print_steps = print_steps
    self.print_matrices = print_matrices

    assert a_alfa.shape[0] == self.n
    assert a_beta.shape[0] == self.n-1

    self.last_alfa = np.ndarray(self.n, dtype=float)
    self.last_beta = np.ndarray(self.n-1, dtype=float)
    self.actual_alfa = np.ndarray(self.n, dtype=float)
    self.actual_beta = np.ndarray(self.n-1, dtype=float)
    self.auto_vec = np.identity(self.n, dtype=float)

    np.copyto(self.actual_alfa, a_alfa)
    np.copyto(self.actual_beta, a_beta)

  def _wilkinson_coeficient(self):
    pass

  def _givens_coeficients(self, k):
    alfa = self.actual_alfa[k]
    beta = self.actual_beta[k]

    # Método normal
    div = np.sqrt(alfa**2 + beta**2)
    ck = alfa / div
    sk = -beta / div

    # Método mais estável
    # if abs(alfa) > abs(beta):
    #   tau = -beta/alfa
    #   ck = 1 / np.sqrt(1 + tau ** 2)
    #   sk = ck * tau
    # else:
    #   tau = -alfa/beta
    #   sk = 1 / np.sqrt(1 + tau ** 2)
    #   ck = sk * tau
    
    if self.print_steps:
      print(f'{(ck, sk)}')
    return (ck, sk)

  def _get_R(self, gama):
    R = np.zeros((self.n, self.n))
    for i in range(self.n-1):
      R[i,i] = self.actual_alfa[i]
      R[i,i+1] = gama[i]

    R[self.n-1, self.n-1] = self.actual_alfa[self.n-1]

    return R

  def get_A(self):
    A = np.zeros((self.n, self.n))
    for i in range(self.n-1):
      A[i,i] = self.actual_alfa[i]
      A[i,i+1] = self.actual_beta[i]
      A[i+1,i] = self.actual_beta[i]
    A[self.n-1, self.n-1] = self.actual_alfa[self.n-1]

    return A

  def get_Qi(self, i, ci, si):
    Qi = np.identity(self.n)
    Qi[i,i] = ci
    Qi[i+1,i+1] = ci
    Qi[i,i+1] = -si
    Qi[i+1,i] = si

    return Qi

  def get_Q(self, ck, sk):
    Q = np.identity(self.n)
    for i in range(self.n-1):
      Q = np.matmul(Q, self.get_Qi(i, ck[i], sk[i]).T)

    return Q

  def _iterate_once(self, epsilon):

    if self.print_matrices:
      Ai = self.get_A()

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
      print(f'\n\nIteração {self.k+1}:')
      print(f'\nDeslocamento espectral: uk = {uk}')
      print(f'\nCoeficientes de Givens:')
      

    # Alfa e beta da matriz R(k)
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
      print(f'\nMatriz R(k)')
      print(f'alfa: {self.actual_alfa}')
      print(f'gama: {gama}')

    if self.print_matrices:
      Ri = self._get_R(gama)

    # Alfa e beta da matriz A(k+1)
    for i in range(self.n-1):
      self.actual_alfa[i] = self.actual_alfa[i] * ck[i] - gama[i] * sk[i]
      self.actual_beta[i] = -self.actual_alfa[i+1] * sk[i]
      self.actual_alfa[i+1] = self.actual_alfa[i+1] * ck[i]

    #   print(self.actual_alfa[i], self.actual_beta[i])
    #   print(self.actual_alfa[i+1])

    # Correção por deslocamente espectral
    self.actual_alfa += uk

    # Atualiza autovetores

    if self.print_steps:
      print(f'\nMatriz A(k+1)')
      print(f'alfa: {self.actual_alfa}')
      print(f'beta: {self.actual_beta}')
      print(f'\nAutovetores V(k+1)\n{self.auto_vec}')

    if self.print_matrices:
      Ai_plus_1 = self.get_A()
      Qi = self.get_Q(ck, sk)
    
    self.k += 1

    if self.print_matrices:
      print('\n', Ai)
      print(Ri)
      print(Qi)
      print(Ai_plus_1)
    # return (Ai, Ri, Qi, Ai_plus_1)

  def iterate(self, epsilon):
    pass


if __name__ == '__main__':
    alfa = 4*np.ones(3)
    beta = 3*np.ones(2)

    method = QRMethod(alfa, beta, spectral=False, print_steps=True, print_matrices=False)
    method._iterate_once(0)