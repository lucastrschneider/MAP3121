import numpy as np

class QRMethod:
  """
  Classe que representa o método QR para cálculo de autovalores e autovetores.

  Args:
    a_alfa (numpy.ndarray): Array de uma dimensão contendo os 'n' elementos da
                            diagonal principal da matriz A do sistema.
    a_beta (numpy.ndarray): Array de uma dimensão contendo os 'n-1' elementos da
                            subdiagonal da matriz A do sistema.
    spectral (bool, optional): Indica se deve usar ou não deslocamento espectral. Padrão é True.
    print_steps (bool, optional): Indica se deve imprimir informações de cada iteração na tela. Padrão é False.

  Attributes:
    n (int): Variável que indica a dimensão atual da matriz do sistema.
    k (int): Guarda o número da iteração atual do algorítmo.
    spectral (bool): Inidica se deve ser usado ou não o deslocamento espectral.
    print_steps (bool): Indica se deve imprimir informações de cada iteração na tela.
    actual_alfa (numpy.ndarray): Guarda o valor atual da diagonal principal de A.
    actual_beta (numpy.ndarray): Guarda o valor atual da subdiagonal de A.
    eigen_vec (numpy.ndarray): Matriz 'n x n' que guarda os autovetores do sistema.
  """
  def __init__(self, a_alfa, a_beta, Ht, spectral=True, print_steps=False, print_final=True):
    self.n = a_alfa.shape[0]
    self.spectral = spectral
    self.k = 0
    self.print_steps = print_steps
    self.print_final = print_final

    assert a_alfa.shape[0] == self.n
    assert a_beta.shape[0] == self.n-1

    self.last_alfa = np.ndarray(self.n, dtype=float)
    self.last_beta = np.ndarray(self.n-1, dtype=float)
    self.actual_alfa = np.ndarray(self.n, dtype=float)
    self.actual_beta = np.ndarray(self.n-1, dtype=float)
    self.eigen_vec = np.ndarray((self.n, self.n), dtype=float)

    np.copyto(self.actual_alfa, a_alfa)
    np.copyto(self.actual_beta, a_beta)
    np.copyto(self.eigen_vec, Ht)


  def _wilkinson_coeficient(self):
    """
    Calcula o coeficiente da heurística de Wilkinson para o caso de deslocamento espectral.

    Returns:
      float: Coeficiente de Wilkinson para matriz atual.
    """
    dk = (self.actual_alfa[self.n-2] - self.actual_alfa[self.n-1]) / 2
    return self.actual_alfa[self.n-1] + dk - np.sign(dk)*np.sqrt(dk**2 + (self.actual_beta[self.n-2])**2)


  def _givens_coeficients(self, k):
    """
    Calcula os coeficientes da rotação de givens para linha atual considerada.

    Args:
      k (int): Linha da matriz considerada para calculo dos coeficientes.

    Returns:
      (float, float): Retorna um tuple com os coeficientes ck e sk.
    """
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
    """
    Retorna a matriz 2x2 para a rotação de Givens.

    Args:
      ci (float): Coeficiente de Givens de cosseno.
      si (float): Coeficiente de Givens de seno.

    Returns:
      numpy.ndarray: Matriz 2x2 para rotação de Givens dados os coeficientes dos parâmetros.
    """
    Qi = np.identity(2) * ci
    Qi[0,1] = -si
    Qi[1,0] = si

    return Qi


  def _iterate_once(self):
    """
    Método que representa uma iteração do algorítmo QR.
    """
    # Heuristica de Wilkinson (coeficiente uk)
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

    # Atualização da matriz A(k+1)
    for i in range(self.n-1):
      self.actual_alfa[i] = self.actual_alfa[i] * ck[i] - gama[i] * sk[i]
      self.actual_beta[i] = -self.actual_alfa[i+1] * sk[i]
      self.actual_alfa[i+1] = self.actual_alfa[i+1] * ck[i]

    # Correção por deslocamente espectral
    self.actual_alfa += uk

    # Atualização dos autovetores V(k+1)
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
    """
    Método público que realiza as iterações do algorítmo até que a condição de parada dada por
    epsilon seja satisfeita.

    Args:
        epsilon (float): Valor máximo que os betas podem ter para que sejam considerados nulos.
                         Quanto menor esse valor, mais iterações vão ocorrer. Um bom valor inicial é 1e-6.

    Returns:
        (numpy.ndarray, numpy.ndarray): Tuple contendo o array com os autovalores e a matriz com os
                                        autovetores correspondentes em colunas.
    """
    for m in range(self.n-1, 0, -1):
      while abs(self.actual_beta[m-1]) >= epsilon:
        self._iterate_once()
      self.n -= 1

    if self.print_final:
      print(f'\nResultados finais:')
      print(f'Numero de iteracoes: {self.k}')
      print(f'Auto valores:\n{self.actual_alfa}')
      print(f'Auto vetores:\n{self.eigen_vec}')

    return self.actual_alfa, self.eigen_vec


if __name__ == '__main__':
  print('Esse arquivo contem apenas o algoritmo QR, mas não os testes.')
  print('Para rodar os testes corretamente, por favor consulte o README.txt')
